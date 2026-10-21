"""Phase 13 — what-if sandbox application orchestrator.

Pipeline (mirrors dev blueprint §4 + test-phase edge matrix):

1. Validate mapping → load feature matrix (float32, ``feature_index`` order)
   via :class:`TypeCastingEngine`, restricted to ``selected_row_indices``.
2. Chunked (≤2,500-row) batch predict over the queue + a subsampled
   reference pool (``<= 5_000`` rows) for the KDTree candidate search.
3. Build ``X/Y/Z`` :class:`AxisScaleMeta` world scales so ghost + mutated
   spheres land exactly on the Stage3D world cube (clamped ``[-7, +7]``).
4. ``evaluate`` answers a slider mutation with one extra predict call;
   ``counterfactual`` runs :class:`CounterfactualFinderService` honoring
   immutable locks, then re-evaluates the recipe for the 3D delta vector.

Memory (§5.1): one queue matrix ``(N, D)`` + one reference pool
``(<=5_000, D)`` float32 + prediction vectors per session; LRU-capped.
"""

from __future__ import annotations

import asyncio
import gc
import uuid
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.core.math_3d import WorldCoordinateScaler
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import TaskType
from app.domains.whatif.entities import (
    CounterfactualSearchResultEntity,
    WhatIfEvaluationResultEntity,
)
from app.domains.whatif.services import (
    CounterfactualFinderService,
    detect_feature_kinds,
)
from app.domains.whatif.value_objects import FeatureChangeDeltaVO, FeatureRangeVO
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)
from app.infrastructure.storage.parquet_dataset_repository import (
    ParquetDatasetRepository,
)

PREDICT_CHUNK_SIZE = 2_500
MAX_QUEUE_POINTS = 10_000
MAX_SESSIONS = 32
REFERENCE_POOL_MAX_ROWS = 5_000


class WhatIfSessionBuildRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    selected_row_indices: list[int] = Field(default_factory=list)
    axis_x_feature_index: int | None = None
    axis_y_feature_index: int | None = None
    session_id: str | None = None


class WhatIfEvaluateRequest(BaseModel):
    mutated_feature_values: dict[str, float] = Field(default_factory=dict)
    target_threshold: float | None = None


class WhatIfCounterfactualRequest(BaseModel):
    target_class_or_threshold: float | None = None
    locked_feature_indices: list[int] = Field(default_factory=list)
    locked_feature_names: list[str] = Field(default_factory=list)


class WhatIfSessionState(BaseModel):
    session_id: str
    dataset_id: str
    model_id: str
    mapping_id: str
    row_indices: list[int]
    feature_names: list[str]
    is_classification: bool
    axis_x_feature_index: int
    axis_y_feature_index: int
    feature_kinds: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


_SESSIONS: dict[str, WhatIfSessionState] = {}
_SESSION_MATRICES: dict[str, dict[str, Any]] = {}
_SESSION_ORDER: list[str] = []
MAX_SESSIONS_STATE = 32


def _store_session(session_id: str, state: WhatIfSessionState, matrices: dict[str, Any]) -> None:
    _SESSIONS[session_id] = state
    _SESSION_MATRICES[session_id] = matrices
    if session_id in _SESSION_ORDER:
        _SESSION_ORDER.remove(session_id)
    _SESSION_ORDER.append(session_id)
    while len(_SESSION_ORDER) > MAX_SESSIONS_STATE:
        old = _SESSION_ORDER.pop(0)
        _SESSIONS.pop(old, None)
        _SESSION_MATRICES.pop(old, None)


def _world_triplet(x: float, y: float, z: float, matrices: dict[str, Any]) -> tuple[float, float, float]:
    wx = float(WorldCoordinateScaler.to_world_coords(np.asarray([x], dtype=np.float32), matrices["x_scale"])[0])
    wy = float(WorldCoordinateScaler.to_world_coords(np.asarray([y], dtype=np.float32), matrices["y_scale"])[0])
    wz = float(WorldCoordinateScaler.to_world_coords(np.asarray([z], dtype=np.float32), matrices["z_scale"])[0])
    return (
        float(np.clip(wx, -7.5, 7.5)),
        float(np.clip(wy, -7.5, 7.5)),
        float(np.clip(wz, -7.5, 7.5)),
    )


class WhatIfOrchestrator:
    def __init__(
        self,
        dataset_repo: ParquetDatasetRepository | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
    ) -> None:
        self._datasets = dataset_repo or ParquetDatasetRepository()
        self._models = model_use_case or ModelIngestUseCase()
        self._mappings = mapping_repo or InMemoryMappingRepository()

    # -- session build -------------------------------------------------
    async def build_session(self, req: WhatIfSessionBuildRequest) -> WhatIfSessionState:
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                details={"mapping_id": req.mapping_id},
                http_status=422,
            )
        selected = [int(i) for i in (req.selected_row_indices or []) if int(i) >= 0]
        if len(selected) > MAX_QUEUE_POINTS:
            raise DomainException(
                code="WHATIF_TOO_MANY_FRAMES",
                message=f"Select at most {MAX_QUEUE_POINTS} rows per what-if session",
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        feature_names = [
            i.model_feature_name or i.dataset_column_name or f"feature_{i.feature_index}"
            for i in items
        ]
        if not feature_cols:
            raise DomainException(
                code="WHATIF_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )
        n_features = len(items)
        x_idx = int(req.axis_x_feature_index) if req.axis_x_feature_index is not None else 0
        y_idx = int(req.axis_y_feature_index) if req.axis_y_feature_index is not None else min(1, n_features - 1)
        if not (0 <= x_idx < n_features) or not (0 <= y_idx < n_features):
            raise DomainException(
                code="WHATIF_AXIS_OUT_OF_RANGE",
                message=f"Axis feature index out of range (model has {n_features} features)",
                http_status=422,
            )
        if x_idx == y_idx and n_features > 1:
            raise DomainException(
                code="WHATIF_AXIS_DUPLICATE",
                message="X and Y axes must be different features",
                http_status=422,
            )

        parquet_path = self._datasets._require(req.dataset_id)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        available = set(pf.schema.names)
        for col in feature_cols:
            if col not in available:
                raise DomainException(
                    code="WHATIF_COLUMN_MISSING",
                    message=f"Column '{col}' not found in cached dataset",
                    http_status=422,
                )
        for idx in selected:
            if idx >= total_rows:
                raise DomainException(
                    code="WHATIF_ROW_OUT_OF_RANGE",
                    message=f"Row index {idx} exceeds dataset size ({total_rows} rows)",
                    http_status=422,
                )

        model_obj = self._models.get_model(req.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")
        is_classification = adapter.task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )

        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X_full = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df
            gc.collect()

        queue_idx = np.asarray(selected, dtype=np.int64) if selected else np.empty((0,), dtype=np.int64)
        X_queue = (
            np.ascontiguousarray(X_full[queue_idx], dtype=np.float32)
            if queue_idx.size
            else np.empty((0, n_features), dtype=np.float32)
        )

        # Reference pool: deterministic stride subsample (bounded RAM).
        if total_rows > REFERENCE_POOL_MAX_ROWS:
            stride = max(1, total_rows // REFERENCE_POOL_MAX_ROWS)
            ref_idx = np.arange(0, total_rows, stride, dtype=np.int64)[:REFERENCE_POOL_MAX_ROWS]
        else:
            ref_idx = np.arange(0, total_rows, dtype=np.int64)
        X_ref = np.ascontiguousarray(X_full[ref_idx], dtype=np.float32)
        finite_full = X_full[np.isfinite(X_full).all(axis=1)] if len(X_full) else X_full
        del X_full
        gc.collect()

        queue_preds, queue_proba = await asyncio.to_thread(
            self._chunked_predict, adapter, model_obj, X_queue
        )
        ref_preds, _ = await asyncio.to_thread(self._chunked_predict, adapter, model_obj, X_ref)

        feature_kinds = detect_feature_kinds(X_ref if len(X_ref) else X_queue)
        col_min, col_max, col_mean = _column_stats(X_ref, X_queue, n_features)

        x_scale = WorldCoordinateScaler.build_axis_scale(
            feature_names[x_idx], (X_ref[:, x_idx] if len(X_ref) else np.array([0.0, 1.0], dtype=np.float32)),
            feature_idx=x_idx,
        )
        y_scale = WorldCoordinateScaler.build_axis_scale(
            feature_names[y_idx], (X_ref[:, y_idx] if len(X_ref) else np.array([0.0, 1.0], dtype=np.float32)),
            feature_idx=y_idx,
        )
        z_pool = [a[np.isfinite(a)] for a in (queue_preds, ref_preds) if a.size]
        z_values = (
            np.concatenate([a.astype(np.float32, copy=False) for a in z_pool if a.size])
            if any(a.size for a in z_pool)
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        z_scale = WorldCoordinateScaler.build_axis_scale("Predicted value", z_values, feature_idx=None)

        queue_classes: list[int] | None = None
        queue_conf: list[float] = []
        if is_classification and queue_proba is not None and queue_proba.size:
            proba = np.where(np.isfinite(queue_proba), queue_proba, 0.0)
            queue_classes = [int(v) for v in np.argmax(proba, axis=1)]
            queue_conf = [float(np.max(proba[r])) for r in range(proba.shape[0])]
        else:
            queue_conf = [1.0 if np.isfinite(v) else 0.0 for v in queue_preds.ravel()]

        _ = finite_full  # freed above; kept for clarity of the float32 pipeline
        gc.collect()

        session_id = req.session_id or f"whatif_{uuid.uuid4().hex[:12]}"
        state = WhatIfSessionState(
            session_id=session_id,
            dataset_id=req.dataset_id,
            model_id=req.model_id,
            mapping_id=req.mapping_id,
            row_indices=selected,
            feature_names=feature_names,
            is_classification=is_classification,
            axis_x_feature_index=x_idx,
            axis_y_feature_index=y_idx,
            feature_kinds=feature_kinds,
        )
        _store_session(
            session_id,
            state,
            {
                "X_queue": X_queue,
                "X_ref": X_ref,
                "ref_preds": np.asarray(ref_preds).ravel(),
                "queue_preds": np.asarray(queue_preds).ravel(),
                "queue_proba": queue_proba,
                "queue_classes": queue_classes,
                "queue_conf": queue_conf,
                "col_min": col_min,
                "col_max": col_max,
                "col_mean": col_mean,
                "x_scale": x_scale,
                "y_scale": y_scale,
                "z_scale": z_scale,
            },
        )
        return state

    # -- evaluation ----------------------------------------------------
    def evaluate_frame(
        self,
        session_id: str,
        frame_index: int,
        mutated_feature_values: dict[str, float] | None = None,
    ) -> WhatIfEvaluationResultEntity:
        try:
            state = _SESSIONS[session_id]
            matrices = _SESSION_MATRICES[session_id]
        except KeyError as exc:
            raise DomainException(
                code="WHATIF_SESSION_NOT_FOUND",
                message=f"What-If session '{session_id}' not found — build it via POST /api/v1/whatif/sessions/build first",
                http_status=404,
            ) from exc
        n = len(state.row_indices)
        if frame_index < 0 or frame_index >= n:
            raise DomainException(
                code="WHATIF_FRAME_OUT_OF_RANGE",
                message=f"Frame {frame_index} out of range (0..{n - 1})",
                http_status=422,
            )
        X_queue: np.ndarray = matrices["X_queue"]
        base_row = np.ascontiguousarray(X_queue[frame_index], dtype=np.float32).copy()
        mutated_row = base_row.copy()
        name_to_idx = {name: i for i, name in enumerate(state.feature_names)}
        for key, raw_val in (mutated_feature_values or {}).items():
            try:
                fval = float(raw_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fval):
                continue
            idx: int | None = None
            if key in name_to_idx:
                idx = name_to_idx[key]
            else:
                try:
                    maybe = int(key)
                    if 0 <= maybe < len(state.feature_names):
                        idx = maybe
                except (TypeError, ValueError):
                    idx = None
            if idx is None:
                continue
            kind = state.feature_kinds[idx] if idx < len(state.feature_kinds) else "float"
            if kind in ("integer", "binary"):
                fval = float(np.round(fval))
                if kind == "binary":
                    fval = float(np.clip(fval, 0.0, 1.0))
            mutated_row[idx] = np.float32(fval)

        model_obj = self._models.get_model(state.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")

        def _predict_with_proba(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
            result = adapter.predict_batch(model_obj, np.ascontiguousarray(batch, dtype=np.float32))
            preds = np.asarray(list(result.predictions), dtype=np.float64).ravel()
            proba = None
            if result.probabilities is not None:
                try:
                    proba = np.asarray(result.probabilities, dtype=np.float64)
                except (TypeError, ValueError):
                    proba = None
            return preds, proba

        base_preds, base_proba = _predict_with_proba(base_row.reshape(1, -1))
        mut_preds, mut_proba = _predict_with_proba(mutated_row.reshape(1, -1))
        base_pred_raw = float(base_preds[0]) if base_preds.size and np.isfinite(base_preds[0]) else 0.0
        mut_pred_raw = float(mut_preds[0]) if mut_preds.size and np.isfinite(mut_preds[0]) else 0.0

        base_cls: int | None = None
        mut_cls: int | None = None
        base_conf = 1.0
        mut_conf = 1.0
        if state.is_classification:
            if base_proba is not None and base_proba.size:
                base_cls = int(np.argmax(base_proba[0]))
                base_conf = float(np.clip(np.max(base_proba[0]), 0.0, 1.0))
            else:
                base_cls = int(round(base_pred_raw))
            if mut_proba is not None and mut_proba.size:
                mut_cls = int(np.argmax(mut_proba[0]))
                mut_conf = float(np.clip(np.max(mut_proba[0]), 0.0, 1.0))
            else:
                mut_cls = int(round(mut_pred_raw))
            orig_pred, mut_pred = float(base_cls), float(mut_cls)
            flipped = base_cls != mut_cls
        else:
            orig_pred, mut_pred = base_pred_raw, mut_pred_raw
            flipped = False

        x_idx, y_idx = state.axis_x_feature_index, state.axis_y_feature_index
        z_mid = float((matrices["z_scale"].raw_min + matrices["z_scale"].raw_max) / 2.0)
        # Classification Z rides the winning-class probability; regression Z
        # rides the raw prediction — both on the session z_scale so ghost +
        # mutated spheres stay consistent with the 3D stage cube.
        base_z = base_conf if state.is_classification else base_pred_raw
        mut_z = mut_conf if state.is_classification else mut_pred_raw
        if not np.isfinite(base_z):
            base_z = z_mid
        if not np.isfinite(mut_z):
            mut_z = z_mid
        orig_world = _world_triplet(
            float(base_row[x_idx]) if x_idx < base_row.size else 0.0,
            float(base_row[y_idx]) if y_idx < base_row.size else 0.0,
            float(base_z),
            matrices,
        )
        mut_world = _world_triplet(
            float(mutated_row[x_idx]) if x_idx < mutated_row.size else 0.0,
            float(mutated_row[y_idx]) if y_idx < mutated_row.size else 0.0,
            float(mut_z),
            matrices,
        )
        world_dist = float(np.linalg.norm(np.subtract(mut_world, orig_world)))

        col_min: np.ndarray = matrices["col_min"]
        col_max: np.ndarray = matrices["col_max"]
        changed: list[FeatureChangeDeltaVO] = []
        for i, name in enumerate(state.feature_names):
            if i >= base_row.size:
                break
            delta = float(mutated_row[i] - base_row[i])
            if abs(delta) <= 1e-9:
                continue
            ood = bool(
                (i < col_min.size and float(mutated_row[i]) < float(col_min[i]))
                or (i < col_max.size and float(mutated_row[i]) > float(col_max[i]))
            )
            denom = abs(float(base_row[i]))
            pct = float(delta / denom * 100.0) if denom > 1e-12 else 0.0
            changed.append(
                FeatureChangeDeltaVO(
                    feature_index=i,
                    feature_name=name,
                    original_value=float(base_row[i]),
                    mutated_value=float(mutated_row[i]),
                    delta_value=delta,
                    delta_percent=pct,
                    is_out_of_distribution=ood,
                )
            )

        clamped = bool(
            any(abs(c) >= 7.5 - 1e-6 for c in (*orig_world, *mut_world))
        )
        trace: dict[str, Any] = {
            "model_algorithm": getattr(adapter, "algorithm_key", ""),
            "is_classification": state.is_classification,
            "axis_x_feature_index": x_idx,
            "axis_y_feature_index": y_idx,
            "world_clamped": clamped,
        }
        return WhatIfEvaluationResultEntity(
            base_frame_index=int(frame_index),
            dataset_row_index=int(state.row_indices[frame_index]),
            original_position_world=orig_world,
            mutated_position_world=mut_world,
            original_prediction=float(orig_pred),
            mutated_prediction=float(mut_pred),
            original_confidence=float(np.clip(base_conf, 0.0, 1.0)),
            mutated_confidence=float(np.clip(mut_conf, 0.0, 1.0)),
            original_class=base_cls,
            mutated_class=mut_cls,
            decision_flipped=bool(flipped),
            world_distance=world_dist,
            changed_features=changed,
            internal_model_trace=trace,
        )

    # -- counterfactual ------------------------------------------------
    def find_counterfactual(
        self,
        session_id: str,
        frame_index: int,
        target_class_or_threshold: float | None,
        locked_feature_indices: list[int],
        locked_feature_names: list[str] | None = None,
    ) -> CounterfactualSearchResultEntity:
        try:
            state = _SESSIONS[session_id]
            matrices = _SESSION_MATRICES[session_id]
        except KeyError as exc:
            raise DomainException(
                code="WHATIF_SESSION_NOT_FOUND",
                message=f"What-If session '{session_id}' not found — build it via POST /api/v1/whatif/sessions/build first",
                http_status=404,
            ) from exc
        n = len(state.row_indices)
        if frame_index < 0 or frame_index >= n:
            raise DomainException(
                code="WHATIF_FRAME_OUT_OF_RANGE",
                message=f"Frame {frame_index} out of range (0..{n - 1})",
                http_status=422,
            )
        name_to_idx = {name: i for i, name in enumerate(state.feature_names)}
        locked_set = {int(i) for i in locked_feature_indices if 0 <= int(i) < len(state.feature_names)}
        for name in locked_feature_names or []:
            if name in name_to_idx:
                locked_set.add(name_to_idx[name])
        locked = sorted(locked_set)
        if len(locked) >= len(state.feature_names) and state.feature_names:
            return CounterfactualSearchResultEntity(
                found=False,
                status_message="กรุณาปลดล็อกอย่างน้อย 1 Feature เพื่อค้นหา Counterfactual",
                evaluation=None,
                locked_feature_indices=locked,
                target_description="",
            )

        X_queue: np.ndarray = matrices["X_queue"]
        base_row = np.ascontiguousarray(X_queue[frame_index], dtype=np.float32).copy()

        model_obj = self._models.get_model(state.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")

        def _predict_fn(batch: np.ndarray) -> np.ndarray:
            result = adapter.predict_batch(model_obj, np.ascontiguousarray(batch, dtype=np.float32))
            if state.is_classification and result.probabilities is not None:
                try:
                    mat = np.asarray(result.probabilities, dtype=np.float64)
                    if mat.ndim == 2 and mat.shape[0] == np.asarray(batch).shape[0]:
                        # For candidate validation the finder compares rounded
                        # raw predictions; map winning class -> label.
                        classes = np.argmax(np.where(np.isfinite(mat), mat, 0.0), axis=1)
                        return classes.astype(np.float32)
                except (TypeError, ValueError):
                    pass
            return np.asarray(
                [float(v) if np.isfinite(float(v)) else 0.0 for v in result.predictions],
                dtype=np.float32,
            )

        # Resolve the target when the caller did not pick one.
        target: float
        target_desc: str
        if target_class_or_threshold is not None:
            target = float(target_class_or_threshold)
            target_desc = (
                f"Flip to Class {int(round(target))}"
                if state.is_classification
                else f"Reach prediction {target:g}"
            )
        elif state.is_classification:
            ref_preds = np.asarray(matrices["ref_preds"]).ravel()
            base_cls = int(round(float(_predict_fn(base_row.reshape(1, -1))[0])))
            options = sorted({int(round(float(v))) for v in ref_preds.tolist() if np.isfinite(v)})
            options = [c for c in options if c != base_cls]
            if not options:
                options = [1 - base_cls] if base_cls in (0, 1) else [base_cls + 1]
            target = float(options[0])
            target_desc = f"Flip to Class {int(round(target))}"
        else:
            raise DomainException(
                code="WHATIF_TARGET_REQUIRED",
                message="Regression counterfactuals need target_class_or_threshold (e.g. Target Prediction >= 500.0)",
                http_status=422,
            )

        X_ref: np.ndarray = matrices["X_ref"]
        ref_preds = np.asarray(matrices["ref_preds"]).ravel()
        found, recipe_row, message = CounterfactualFinderService.find_minimum_change_counterfactual(
            predict_fn=_predict_fn,
            X_dataset=np.ascontiguousarray(X_ref, dtype=np.float32),
            y_predictions=np.asarray(ref_preds),
            base_row=base_row,
            locked_feature_indices=locked,
            target_class_or_threshold=float(target),
            is_classification=bool(state.is_classification),
            feature_kinds=list(state.feature_kinds),
        )
        if not found:
            return CounterfactualSearchResultEntity(
                found=False,
                status_message=message,
                evaluation=None,
                locked_feature_indices=locked,
                target_description=target_desc,
            )
        mutated_values = {
            state.feature_names[i]: float(recipe_row[i])
            for i in range(len(state.feature_names))
            if abs(float(recipe_row[i] - base_row[i])) > 1e-9
        }
        evaluation = self.evaluate_frame(session_id, frame_index, mutated_values)
        for change in evaluation.changed_features:
            if change.feature_index in locked:
                change.is_locked = True
        return CounterfactualSearchResultEntity(
            found=True,
            status_message=message,
            evaluation=evaluation,
            locked_feature_indices=locked,
            target_description=target_desc,
        )

    # -- helpers -------------------------------------------------------
    @staticmethod
    def session_feature_ranges(session_id: str) -> list[FeatureRangeVO]:
        try:
            state = _SESSIONS[session_id]
            matrices = _SESSION_MATRICES[session_id]
        except KeyError as exc:
            raise DomainException(
                code="WHATIF_SESSION_NOT_FOUND",
                message=f"What-If session '{session_id}' not found",
                http_status=404,
            ) from exc
        col_min: np.ndarray = matrices["col_min"]
        col_max: np.ndarray = matrices["col_max"]
        col_mean: np.ndarray = matrices["col_mean"]
        X_queue: np.ndarray = matrices["X_queue"]
        ranges: list[FeatureRangeVO] = []
        for i, name in enumerate(state.feature_names):
            lo = float(col_min[i]) if i < col_min.size else 0.0
            hi = float(col_max[i]) if i < col_max.size else 1.0
            if not (np.isfinite(lo) and np.isfinite(hi)) or abs(hi - lo) < 1e-9:
                if i < X_queue.shape[1] and len(X_queue):
                    col = X_queue[:, i]
                    finite = col[np.isfinite(col)]
                    if finite.size:
                        lo, hi = float(np.min(finite)), float(np.max(finite))
                if abs(hi - lo) < 1e-9:
                    lo, hi = lo - 1.0, hi + 1.0
            kind = state.feature_kinds[i] if i < len(state.feature_kinds) else "float"
            step = 1.0 if kind in ("integer", "binary") else max((hi - lo) / 200.0, 1e-6)
            ranges.append(
                FeatureRangeVO(
                    feature_index=i,
                    feature_name=name,
                    min_value=float(lo),
                    max_value=float(hi),
                    mean_value=float(col_mean[i]) if i < col_mean.size and np.isfinite(col_mean[i]) else 0.0,
                    step=float(step),
                    kind=kind,
                )
            )
        return ranges

    @staticmethod
    def get_session(session_id: str) -> WhatIfSessionState:
        try:
            return _SESSIONS[session_id]
        except KeyError as exc:
            raise DomainException(
                code="WHATIF_SESSION_NOT_FOUND",
                message=f"What-If session '{session_id}' not found",
                http_status=404,
            ) from exc

    @staticmethod
    def clear_sessions() -> int:
        n = len(_SESSIONS)
        _SESSIONS.clear()
        _SESSION_MATRICES.clear()
        _SESSION_ORDER.clear()
        return n

    @staticmethod
    def _chunked_predict(
        adapter: Any, model_obj: Any, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        n = int(X.shape[0])
        preds = np.empty((n,), dtype=np.float64)
        preds[:] = np.nan
        proba_rows: list[list[float] | None] = []
        has_proba = True
        offset = 0
        while offset < n:
            chunk = np.ascontiguousarray(X[offset : offset + PREDICT_CHUNK_SIZE], dtype=np.float32)
            result = adapter.predict_batch(model_obj, chunk)
            vals = np.asarray(list(result.predictions), dtype=np.float64)
            preds[offset : offset + vals.shape[0]] = vals
            if result.probabilities is not None and has_proba:
                for row in result.probabilities:
                    proba_rows.append([float(v) for v in row])
            else:
                has_proba = False
            del chunk, result, vals
            gc.collect()
            offset += PREDICT_CHUNK_SIZE
        proba: np.ndarray | None = None
        if has_proba and proba_rows and len(proba_rows) == n:
            try:
                proba = np.asarray(proba_rows, dtype=np.float64)
                if proba.ndim != 2 or proba.shape[0] != n or proba.shape[1] == 0:
                    proba = None
            except (TypeError, ValueError):
                proba = None
        return preds, proba


def _column_stats(
    X_ref: np.ndarray, X_queue: np.ndarray, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = [a for a in (X_ref, X_queue) if a.size]
    if not parts:
        z = np.zeros((n_features,), dtype=np.float64)
        return z.copy(), z.copy() + 1.0, z.copy()
    stacked = np.concatenate(
        [np.asarray(a, dtype=np.float64).reshape(-1, n_features) for a in parts], axis=0
    )
    with np.errstate(invalid="ignore"):
        col_min = np.nanmin(stacked, axis=0)
        col_max = np.nanmax(stacked, axis=0)
        col_mean = np.nanmean(stacked, axis=0)
    col_min = np.where(np.isfinite(col_min), col_min, 0.0)
    col_max = np.where(np.isfinite(col_max), col_max, col_min + 1.0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    return col_min.astype(np.float64), col_max.astype(np.float64), col_mean.astype(np.float64)


def session_to_response(state: WhatIfSessionState, ranges: list[FeatureRangeVO]) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "total_frames": len(state.row_indices),
        "is_classification": state.is_classification,
        "axis_x_feature_index": state.axis_x_feature_index,
        "axis_y_feature_index": state.axis_y_feature_index,
        "feature_names": state.feature_names,
        "feature_kinds": state.feature_kinds,
        "row_indices": state.row_indices,
        "feature_ranges": [r.model_dump(mode="json") for r in ranges],
    }


def evaluation_to_response(entity: WhatIfEvaluationResultEntity) -> dict[str, Any]:
    return entity.model_dump(mode="json")


def counterfactual_to_response(entity: CounterfactualSearchResultEntity) -> dict[str, Any]:
    return entity.model_dump(mode="json")

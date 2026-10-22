"""Phase 12 — explainability application orchestrator.

Pipeline (mirrors dev blueprint §3 + test-phase edge matrix):

1. Validate mapping → load feature matrix (float32, ``feature_index`` order)
   via :class:`TypeCastingEngine`, restricted to ``selected_row_indices``.
2. Chunked (≤2,500-row) batch predict over the queue only.
3. Resolve the attribution path per model family:
   - Linear (exact weights + intercept available): precompute
     ``feature_means`` over the queue and answer every frame with
     ``compute_linear_contributions`` (µs, exact additivity).
   - Anything else: cache medians/means/stds/sorted columns and answer
     per frame with single-batch median perturbation
     (``compute_fast_perturbation_contributions``), rescaled to
     ``current_pred - base`` where ``base = mean(predictions)``.
   - Classification with probabilities: attributions explain the winning
     class probability (edge #5).
4. Per frame: rank by ``|contribution|``, slice ``top_k``, fold the tail
   into ``other_contributions_sum``, attach percentiles (searchsorted),
   z-scores and raw values. ``base + top_k_sum + other == output``.

Memory (§5.1): only the queue matrix ``(N, D)`` float32 + one sorted
reference ``(min(N,5000), D)`` float32 are cached per session.
"""

from __future__ import annotations

import asyncio
import gc
import uuid
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.domains.explainability.entities import FrameExplanationEntity
from app.domains.explainability.services import FastLocalAttributionService
from app.domains.explainability.value_objects import FeatureAttributionItemVO
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import TaskType
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)
from app.infrastructure.storage.parquet_dataset_repository import (
    ParquetDatasetRepository,
)

PREDICT_CHUNK_SIZE = 2_500
MAX_QUEUE_POINTS = 10_000
MAX_SESSIONS = 32
SORTED_REF_MAX_ROWS = 5_000
DEFAULT_TOP_K = 10
BATCH_MAX_FRAMES = 2_000


class ExplainabilitySessionBuildRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    selected_row_indices: list[int] = Field(default_factory=list)
    session_id: str | None = None


class ExplainabilityFrameBatchRequest(BaseModel):
    frame_indices: list[int] = Field(default_factory=list, max_length=BATCH_MAX_FRAMES)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=500)


class ExplainabilitySessionState(BaseModel):
    session_id: str
    dataset_id: str
    model_id: str
    mapping_id: str
    row_indices: list[int]
    feature_names: list[str]
    is_classification: bool
    base_value: float
    output_values: list[float]
    winning_classes: list[int] | None = None
    linear_weights: list[float] | None = None
    linear_intercept: float | None = None

    model_config = {"arbitrary_types_allowed": True}


_SESSIONS: dict[str, ExplainabilitySessionState] = {}
_SESSION_MATRICES: dict[str, dict[str, Any]] = {}
_SESSION_ORDER: list[str] = []
MAX_SESSIONS_STATE = 32


def _store_session(session_id: str, state: ExplainabilitySessionState, matrices: dict[str, Any]) -> None:
    _SESSIONS[session_id] = state
    _SESSION_MATRICES[session_id] = matrices
    if session_id in _SESSION_ORDER:
        _SESSION_ORDER.remove(session_id)
    _SESSION_ORDER.append(session_id)
    while len(_SESSION_ORDER) > MAX_SESSIONS_STATE:
        old = _SESSION_ORDER.pop(0)
        _SESSIONS.pop(old, None)
        _SESSION_MATRICES.pop(old, None)


def _resolve_linear_weights(final_est: Any, n_features: int) -> tuple[np.ndarray | None, float | None]:
    """Best-effort exact linear weights (coef_ + intercept_)."""
    try:
        coef = getattr(final_est, "coef_", None)
        if coef is None:
            return None, None
        weights = np.asarray(coef, dtype=np.float32)
        if weights.ndim == 2:
            # Binary/multiclass (C, D): winning-class handled per row, but
            # the linear fast path only supports single-output models.
            if weights.shape[0] != 1:
                return None, None
            weights = weights[0]
        weights = np.ascontiguousarray(weights.ravel(), dtype=np.float32)
        if weights.size != n_features or not bool(np.all(np.isfinite(weights))):
            return None, None
        intercept_raw: Any = getattr(final_est, "intercept_", 0.0)
        arr = np.asarray(intercept_raw, dtype=np.float64).ravel()
        intercept = float(arr[0]) if arr.size and bool(np.isfinite(arr[0])) else 0.0
        return weights, intercept
    except (TypeError, ValueError):
        return None, None


class ExplainabilityOrchestrator:
    def __init__(
        self,
        dataset_repo: ParquetDatasetRepository | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
    ) -> None:
        self._datasets = dataset_repo or ParquetDatasetRepository()
        self._models = model_use_case or ModelIngestUseCase()
        self._mappings = mapping_repo or InMemoryMappingRepository()

    # -- public API ----------------------------------------------------
    async def build_session(self, req: ExplainabilitySessionBuildRequest) -> ExplainabilitySessionState:
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
                code="EXPLAIN_TOO_MANY_FRAMES",
                message=f"Select at most {MAX_QUEUE_POINTS} rows per explainability session",
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        # Display names: prefer the human dataset column (e.g. sepal_length)
        # over the positional model name (e.g. feature_3) — models trained
        # from bare ndarrays carry no names.
        feature_names = [
            i.dataset_column_name or i.model_feature_name or f"feature_{i.feature_index}"
            for i in items
        ]
        if not feature_cols:
            raise DomainException(
                code="EXPLAIN_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
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
                    code="EXPLAIN_COLUMN_MISSING",
                    message=f"Column '{col}' not found in cached dataset",
                    http_status=422,
                )
        for idx in selected:
            if idx >= total_rows:
                raise DomainException(
                    code="EXPLAIN_ROW_OUT_OF_RANGE",
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

        queue_idx = np.asarray(selected, dtype=np.int64)
        X = (
            np.ascontiguousarray(X_full[queue_idx], dtype=np.float32)
            if selected
            else np.empty((0, len(feature_cols)), dtype=np.float32)
        )
        del X_full
        gc.collect()

        preds, proba = await asyncio.to_thread(self._chunked_predict, adapter, model_obj, X)

        # Output scale: winning-class probability for classifiers (edge #5),
        # raw regression prediction otherwise.
        output_values: list[float]
        winning: list[int] | None = None
        if is_classification and proba is not None and proba.size:
            proba = np.where(np.isfinite(proba), proba, 0.0)
            winning = [int(v) for v in np.argmax(proba, axis=1)]
            output_values = [float(proba[r, c]) for r, c in enumerate(winning)]
        else:
            output_values = [float(v) if np.isfinite(v) else 0.0 for v in preds.ravel()]

        base_value = float(np.mean(output_values)) if output_values else 0.0

        weights, intercept = (None, None)
        if not is_classification:
            weights, intercept = _resolve_linear_weights(final_est, X.shape[1])
            if weights is not None and intercept is not None:
                means = X.mean(axis=0).astype(np.float32) if len(X) else np.zeros((X.shape[1],), dtype=np.float32)
                base_value = float(intercept + np.dot(means, weights))

        with np.errstate(invalid="ignore"):
            col_means = np.nanmean(X.astype(np.float64, copy=False), axis=0) if len(X) else np.zeros((X.shape[1],), dtype=np.float64)
        with np.errstate(invalid="ignore"):
            col_stds = np.nanstd(X.astype(np.float64, copy=False), axis=0) if len(X) else np.ones((X.shape[1],), dtype=np.float64)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        col_stds = np.where(np.isfinite(col_stds) & (col_stds >= 1e-9), col_stds, 1.0)
        medians = (
            np.nanmedian(X.astype(np.float64, copy=False), axis=0).astype(np.float32)
            if len(X)
            else np.zeros((X.shape[1],), dtype=np.float32)
        )
        medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)

        order = np.argsort(queue_idx, kind="stable") if len(queue_idx) else np.empty((0,), dtype=np.int64)
        ref_rows = order[: min(len(order), SORTED_REF_MAX_ROWS)]
        sorted_ref = np.sort(X[ref_rows], axis=0).astype(np.float32) if len(ref_rows) else np.empty((0, X.shape[1]), dtype=np.float32)

        session_id = req.session_id or f"exp_{uuid.uuid4().hex[:12]}"
        state = ExplainabilitySessionState(
            session_id=session_id,
            dataset_id=req.dataset_id,
            model_id=req.model_id,
            mapping_id=req.mapping_id,
            row_indices=selected,
            feature_names=feature_names,
            is_classification=is_classification,
            base_value=float(base_value),
            output_values=[float(v) for v in output_values],
            winning_classes=winning,
            linear_weights=[float(v) for v in weights] if weights is not None else None,
            linear_intercept=float(intercept) if intercept is not None else None,
        )
        _store_session(
            session_id,
            state,
            {
                "X": X,
                "means": col_means.astype(np.float64),
                "stds": col_stds.astype(np.float64),
                "medians": medians,
                "sorted_ref": sorted_ref,
                "proba": proba,
            },
        )
        return state

    def get_frame_explanation(
        self, session_id: str, frame_index: int, top_k: int = DEFAULT_TOP_K
    ) -> FrameExplanationEntity:
        try:
            state = _SESSIONS[session_id]
            matrices = _SESSION_MATRICES[session_id]
        except KeyError as exc:
            raise DomainException(
                code="EXPLAIN_SESSION_NOT_FOUND",
                message=f"Explainability session '{session_id}' not found — build it via POST /api/v1/explainability/sessions/build first",
                http_status=404,
            ) from exc
        n = len(state.row_indices)
        if frame_index < 0 or frame_index >= n:
            raise DomainException(
                code="EXPLAIN_FRAME_OUT_OF_RANGE",
                message=f"Frame {frame_index} out of range (0..{n - 1})",
                http_status=422,
            )
        top_k = max(1, min(int(top_k), len(state.feature_names)))
        X: np.ndarray = matrices["X"]
        x_row = np.ascontiguousarray(X[frame_index], dtype=np.float32)
        output_value = float(state.output_values[frame_index])
        d = len(state.feature_names)

        if state.linear_weights is not None and state.linear_intercept is not None:
            weights = np.asarray(state.linear_weights, dtype=np.float32)
            stacked = x_row.reshape(1, -1)
            queue_means = X.mean(axis=0).astype(np.float32) if len(X) else np.zeros((d,), dtype=np.float32)
            contribs, _ = FastLocalAttributionService.compute_linear_contributions(
                X_batch=stacked,
                feature_means=queue_means,
                weights=weights,
                intercept=float(state.linear_intercept),
            )
            contribs_vec = contribs[0]
        else:
            medians: np.ndarray = matrices["medians"]
            model_obj = self._models.get_model(state.model_id)
            _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
            adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")

            target_class: int | None = None
            if state.winning_classes is not None:
                target_class = int(state.winning_classes[frame_index])

            def _predict_fn(batch: np.ndarray) -> np.ndarray:
                clean = np.ascontiguousarray(batch, dtype=np.float32)
                result = adapter.predict_batch(model_obj, clean)
                if target_class is not None and result.probabilities is not None:
                    try:
                        mat = np.asarray(result.probabilities, dtype=np.float64)
                        if mat.ndim == 2 and mat.shape[1] > target_class:
                            col = mat[:, target_class]
                            return np.where(np.isfinite(col), col, 0.0).astype(np.float32)
                    except (TypeError, ValueError):
                        pass
                return np.asarray(
                    [float(v) if np.isfinite(float(v)) else 0.0 for v in result.predictions],
                    dtype=np.float32,
                )

            contribs_vec = FastLocalAttributionService.compute_fast_perturbation_contributions(
                predict_fn=_predict_fn,
                x_row=x_row,
                feature_medians=np.asarray(medians, dtype=np.float32),
                base_prediction=float(state.base_value),
            )

        pcts = FastLocalAttributionService.compute_percentiles_for_row(
            x_row, np.asarray(matrices["sorted_ref"], dtype=np.float32)
        )
        means: np.ndarray = np.asarray(matrices["means"], dtype=np.float64)
        stds: np.ndarray = np.asarray(matrices["stds"], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            z_scores = (x_row.astype(np.float64) - means) / stds
        z_scores = np.where(np.isfinite(z_scores), z_scores, 0.0)

        order = np.argsort(np.abs(contribs_vec), kind="stable")[::-1]
        top_idx = order[:top_k]
        tail_idx = order[top_k:]
        other_sum = float(np.sum(contribs_vec[tail_idx])) if tail_idx.size else 0.0

        attributions: list[FeatureAttributionItemVO] = []
        for fi in (int(v) for v in top_idx):
            attributions.append(
                FeatureAttributionItemVO(
                    feature_index=fi,
                    feature_name=state.feature_names[fi] if fi < len(state.feature_names) else f"feature_{fi}",
                    raw_value=float(x_row[fi]),
                    percentile=float(np.clip(pcts[fi] if fi < pcts.size else 50.0, 0.0, 100.0)),
                    z_score=float(z_scores[fi]) if fi < z_scores.size else 0.0,
                    contribution=float(contribs_vec[fi]),
                )
            )
        return FrameExplanationEntity(
            frame_index=int(frame_index),
            dataset_row_index=int(state.row_indices[frame_index]),
            base_value=float(state.base_value),
            output_value=float(output_value),
            attributions=attributions,
            other_contributions_sum=float(other_sum),
        )

    def get_frame_explanations(
        self, session_id: str, frame_indices: list[int], top_k: int = DEFAULT_TOP_K
    ) -> list[FrameExplanationEntity]:
        """Batch variant for prefetch: skips out-of-range frames instead of
        raising, so the frontend can request a blind window around playback."""
        try:
            state = _SESSIONS[session_id]
        except KeyError as exc:
            raise DomainException(
                code="EXPLAIN_SESSION_NOT_FOUND",
                message=f"Explainability session '{session_id}' not found — build it via POST /api/v1/explainability/sessions/build first",
                http_status=404,
            ) from exc
        n = len(state.row_indices)
        out: list[FrameExplanationEntity] = []
        seen: set[int] = set()
        for raw in frame_indices:
            try:
                frame_index = int(raw)
            except (TypeError, ValueError):
                continue
            if frame_index < 0 or frame_index >= n or frame_index in seen:
                continue
            seen.add(frame_index)
            out.append(self.get_frame_explanation(session_id, frame_index, top_k=top_k))
        return out

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


def session_to_response(state: ExplainabilitySessionState) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "total_frames": len(state.row_indices),
        "is_classification": state.is_classification,
        "base_value": state.base_value,
        "feature_names": state.feature_names,
        "row_indices": state.row_indices,
    }


def frame_to_response(entity: FrameExplanationEntity) -> dict[str, Any]:
    return entity.model_dump(mode="json")


def frames_to_response(entities: list[FrameExplanationEntity]) -> dict[str, Any]:
    return {"explanations": [e.model_dump(mode="json") for e in entities]}

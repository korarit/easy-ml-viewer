"""Phase 07 — Evaluation application use-case (chunked batch evaluation).

Pipeline (spec §3.1 / §5.1):
1. Load mapping (feature matrix order + target column + label map).
2. Stream parquet in ``CHUNK_SIZE = 25_000`` row windows; per chunk build a
   single ``float32`` matrix via :class:`TypeCastingEngine` and run
   ``adapter.predict_batch`` — only 1-D ``y_pred`` / ``max_z`` accumulators
   survive across iterations (``X_chunk`` is freed every loop).
3. Harmonize the raw target column against ``model.classes_``
   (``Target Label Harmonizer``, edge cases #1/#2).
4. Aggregate with :class:`GlobalEvaluationService` using ``np.argpartition``.

Memory: peak extra RAM is one chunk (25k x d float32) + three float64
vectors of length N — flat profile for 200k-row files.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import TaskType
from app.domains.simulation.evaluation_service import GlobalEvaluationService
from app.domains.simulation.evaluation_value_objects import (
    GlobalEvaluationSummary,
    PredictionDistribution,
)
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)
from app.infrastructure.storage.parquet_dataset_repository import (
    ParquetDatasetRepository,
)

CHUNK_SIZE = 25_000
MAX_CLASSES_FOR_MATRIX = 10
MAX_TOP_K = 500
_MISMATCH_PROBE_LIMIT = 32


class ComputeEvaluationRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    top_k: int = Field(default=50, ge=1, le=MAX_TOP_K)
    # Optional user-confirmed CSV-label -> model-class mapping
    # (TargetLabelMapperModal). Keys are raw CSV strings.
    target_label_map: dict[str, Any] | None = None


def _coerce_label_key(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _normalize_token(value: Any) -> str:
    import re as _re

    return _re.sub(r"[^a-z0-9]", "", str(value).lower().strip())


def build_class_index(
    raw: Any,
    classes: list[Any],
    label_map: dict[str, Any] | None,
) -> int | None:
    """Map one raw CSV target value to a ``classes_`` index (None = unmapped)."""
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    # 1. Explicit user / stored mapping wins.
    if label_map:
        key = _coerce_label_key(raw)
        if key in label_map:
            mapped = label_map[key]
            for i, cls in enumerate(classes):
                if mapped == cls or _coerce_label_key(mapped) == _coerce_label_key(cls):
                    return i
            try:
                return int(mapped) if int(mapped) in range(len(classes)) else None
            except (TypeError, ValueError):
                return None
    # 2. Direct equality (handles int 0/1 vs int64, "Churn" vs "Churn").
    for i, cls in enumerate(classes):
        try:
            if raw == cls:
                return i
        except Exception:
            pass
        if _coerce_label_key(raw) == _coerce_label_key(cls):
            return i
    # 3. Numeric-string coercion ("1" -> 1, "0.0" -> 0).
    try:
        f = float(str(raw).strip())
        for i, cls in enumerate(classes):
            try:
                if float(cls) == f:  # type: ignore[arg-type]
                    return i
            except (TypeError, ValueError):
                continue
    except (TypeError, ValueError):
        pass
    # 4. Normalized token match ("yes" -> "Yes").
    norm = _normalize_token(raw)
    for i, cls in enumerate(classes):
        if _normalize_token(cls) == norm:
            return i
    return None


class EvaluationUseCase:
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
    def compute(self, req: ComputeEvaluationRequest) -> GlobalEvaluationSummary:
        top_k = int(max(1, min(req.top_k, MAX_TOP_K)))
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                details={
                    "mapping_id": req.mapping_id,
                    "mapping_dataset_id": mapping.dataset_id,
                    "mapping_model_id": mapping.model_id,
                },
                http_status=422,
            )

        model_obj = self._models.get_model(req.model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        schema = UniversalSchemaDetector.extract_schema(final_est)
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")
        task_type = adapter.task_type
        classes: list[Any] = list(schema.classes) if schema.classes else []

        feature_cols = [
            i.dataset_column_name
            for i in sorted(mapping.items, key=lambda x: x.feature_index)
            if i.dataset_column_name
        ]
        if not feature_cols:
            raise DomainException(
                code="EVALUATION_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )
        target_col = mapping.target_column_name
        label_map: dict[str, Any] | None = dict(mapping.target_label_map or {})
        if req.target_label_map:
            label_map.update({str(k): v for k, v in req.target_label_map.items()})
        if not label_map:
            label_map = None

        parquet_path = self._datasets._require(req.dataset_id)
        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        all_columns = set(pf.schema.names)
        wanted_cols = list(feature_cols)
        if target_col:
            wanted_cols.append(target_col)
        for col in wanted_cols:
            if col not in all_columns:
                raise DomainException(
                    code="EVALUATION_COLUMN_MISSING",
                    message=f"Column '{col}' not found in cached dataset",
                    http_status=422,
                )

        # -- pass 1: chunked predict + Welford-free sum/sumsq accumulation ----
        n_features = len(feature_cols)
        sum_vec = np.zeros((n_features,), dtype=np.float64)
        sumsq_vec = np.zeros((n_features,), dtype=np.float64)
        count_total = 0
        y_pred_all: list[float] = []
        proba_all: list[list[float] | None] = []
        has_proba = True
        pred_sum = 0.0
        pred_sumsq = 0.0
        pred_min = float("inf")
        pred_max = float("-inf")
        pred_count = 0

        offset = 0
        while offset < total_rows:
            chunk = self._datasets.read_columns_mmap(
                req.dataset_id, columns=feature_cols, offset=offset, limit=CHUNK_SIZE
            )
            # Reorder to mapping order (read_columns_mmap preserves request order,
            # but guard anyway) and build the single float32 chunk matrix.
            chunk = chunk.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(chunk, list(mapping.items))
            xf64 = X.astype(np.float64, copy=False)
            # NaN-aware accumulation (PASS_NAN models may emit NaNs).
            sum_vec += np.nansum(xf64, axis=0)
            with np.errstate(invalid="ignore"):
                sumsq_vec += np.nansum(xf64 * xf64, axis=0)
            count_total += int(X.shape[0])

            result = adapter.predict_batch(model_obj, np.ascontiguousarray(X, dtype=np.float32))
            preds = list(result.predictions)
            y_pred_all.extend([float(p) if isinstance(p, (int, float)) else float("nan") for p in preds])
            if result.probabilities is not None and has_proba:
                proba_all.extend([list(map(float, r)) for r in result.probabilities])
            else:
                has_proba = False
                proba_all.extend([None] * len(preds))
            for p in preds:
                try:
                    f = float(p)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if f != f or f in (float("inf"), float("-inf")):
                    continue
                pred_sum += f
                pred_sumsq += f * f
                pred_count += 1
                if f < pred_min:
                    pred_min = f
                if f > pred_max:
                    pred_max = f
            del chunk, X, xf64, result
            gc.collect()
            offset += CHUNK_SIZE

        if count_total == 0:
            raise DomainException(
                code="EVALUATION_EMPTY_DATASET",
                message="Dataset has no rows to evaluate",
                http_status=422,
            )

        y_pred = np.asarray(y_pred_all, dtype=np.float64)
        del y_pred_all
        mean = sum_vec / max(count_total, 1)
        var = np.maximum(sumsq_vec / max(count_total, 1) - mean * mean, 0.0)
        std = np.sqrt(var)
        std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)

        # -- pass 2: OOD max-z per row (chunked, bounded RAM) ----------------
        max_z = np.empty((count_total,), dtype=np.float64)
        offset = 0
        pos = 0
        while offset < total_rows:
            chunk = self._datasets.read_columns_mmap(
                req.dataset_id, columns=feature_cols, offset=offset, limit=CHUNK_SIZE
            )
            chunk = chunk.reindex(columns=feature_cols)
            X = TypeCastingEngine.build_float32_matrix(chunk, list(mapping.items))
            zmax = GlobalEvaluationService.max_z_scores(X, mean, std)
            n = int(zmax.shape[0])
            max_z[pos : pos + n] = zmax
            pos += n
            del chunk, X, zmax
            gc.collect()
            offset += CHUNK_SIZE

        ood_order = np.where(max_z > 3.0)[0]
        if ood_order.size:
            ranked = ood_order[np.argsort(-max_z[ood_order], kind="mergesort")][:top_k]
            ood_indices = [int(i) for i in ranked]
        else:
            ood_indices = []

        if pred_count:
            pmean = pred_sum / pred_count
            pvar = max(pred_sumsq / pred_count - pmean * pmean, 0.0)
            distribution = PredictionDistribution(
                pred_min=float(pred_min),
                pred_max=float(pred_max),
                pred_mean=float(pmean),
                pred_std=float(np.sqrt(pvar)),
            )
        else:
            distribution = PredictionDistribution(
                pred_min=0.0, pred_max=0.0, pred_mean=0.0, pred_std=0.0
            )

        if not target_col:
            return GlobalEvaluationSummary(
                task_type=task_type.value,
                evaluated_row_count=count_total,
                ood_outlier_indices=ood_indices,
                ood_outlier_count=int(np.count_nonzero(max_z > 3.0)),
                prediction_distribution=distribution,
            )

        # -- pass 3: target column (chunked raw read, harmonized) ------------
        is_classification = task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )
        if not is_classification:
            y_true = np.empty((count_total,), dtype=np.float64)
            y_true[:] = np.nan
            offset = 0
            pos = 0
            while offset < total_rows:
                chunk = self._datasets.read_columns_mmap(
                    req.dataset_id, columns=[target_col], offset=offset, limit=CHUNK_SIZE
                )
                numeric = pd.to_numeric(chunk[target_col], errors="coerce").to_numpy(
                    dtype=np.float64, copy=True
                )
                n = int(numeric.shape[0])
                y_true[pos : pos + n] = numeric
                pos += n
                del chunk, numeric
                gc.collect()
                offset += CHUNK_SIZE
            report = GlobalEvaluationService.evaluate_regression(y_true, y_pred, top_k=top_k)
            return GlobalEvaluationSummary(
                task_type=task_type.value,
                evaluated_row_count=count_total,
                ood_outlier_indices=ood_indices,
                ood_outlier_count=int(np.count_nonzero(max_z > 3.0)),
                prediction_distribution=distribution,
                regression=report,
            )

        # -- classification: harmonize labels --------------------------------
        if not classes:
            raise DomainException(
                code="EVALUATION_NO_CLASSES",
                message="Model exposes no classes_ — classification evaluation unavailable",
                http_status=422,
            )
        expected = [_coerce_label_key(c) for c in classes]
        y_true_idx = np.full((count_total,), -1, dtype=np.int64)
        raw_uniques: list[str] = []
        seen_uniques: set[str] = set()
        offset = 0
        pos = 0
        while offset < total_rows:
            chunk = self._datasets.read_columns_mmap(
                req.dataset_id, columns=[target_col], offset=offset, limit=CHUNK_SIZE
            )
            series = chunk[target_col]
            for j, raw in enumerate(series.tolist()):
                idx = build_class_index(raw, classes, label_map)
                y_true_idx[pos + j] = int(idx) if idx is not None else -1
                key = _coerce_label_key(raw)
                if key and key not in seen_uniques and len(raw_uniques) < _MISMATCH_PROBE_LIMIT:
                    seen_uniques.add(key)
                    raw_uniques.append(key)
            pos += int(len(series))
            del chunk
            gc.collect()
            offset += CHUNK_SIZE

        valid = y_true_idx >= 0
        if int(np.count_nonzero(valid)) == 0:
            return GlobalEvaluationSummary(
                task_type=task_type.value,
                evaluated_row_count=0,
                ood_outlier_indices=ood_indices,
                ood_outlier_count=int(np.count_nonzero(max_z > 3.0)),
                needs_label_mapping=True,
                label_mapping_hint=(
                    f"Target column '{target_col}' labels {raw_uniques} do not match "
                    f"model classes {expected}. Map each CSV label to a model class, "
                    "then re-run evaluation."
                ),
                raw_target_labels=raw_uniques,
                expected_classes=expected,
                prediction_distribution=distribution,
            )

        valid_positions = np.where(valid)[0]
        # Predicted class -> index (with the same coercion as ground truth).
        pred_idx_full = np.full((count_total,), -1, dtype=np.int64)
        for i in range(count_total):
            p = y_pred[i]
            best: int | None = None
            for ci, cls in enumerate(classes):
                try:
                    if float(cls) == float(p):  # type: ignore[arg-type]
                        best = ci
                        break
                except (TypeError, ValueError):
                    if _coerce_label_key(p) == _coerce_label_key(cls):
                        best = ci
                        break
            # Fallback: nearest class for numeric predictions outside the set.
            if best is None:
                try:
                    f = float(p)
                    numeric_classes = [(ci, float(c)) for ci, c in enumerate(classes)]
                    best = min(numeric_classes, key=lambda t: abs(t[1] - f))[0]
                except (TypeError, ValueError):
                    best = -1
            pred_idx_full[i] = int(best) if best is not None else -1

        # Rows with invalid ground truth are excluded from the confusion math
        # (edge case #2) but every stored index stays in original row space.
        yt_valid = y_true_idx[valid]
        yp_valid = pred_idx_full[valid]
        # Guard: predicted -1 (unmappable) -> fall back to argmax-free nearest.
        yp_valid = np.where(yp_valid < 0, 0, yp_valid)

        proba_matrix: np.ndarray | None = None
        if has_proba and len(proba_all) == count_total and all(
            r is not None for r in proba_all
        ):
            try:
                proba_matrix = np.asarray(proba_all, dtype=np.float64)
                if proba_matrix.ndim != 2 or proba_matrix.shape[0] != int(np.count_nonzero(valid)):
                    # proba covers ALL rows; slice to valid rows.
                    proba_matrix = proba_matrix[valid] if proba_matrix.shape[0] == count_total else None
                else:
                    proba_matrix = proba_matrix[valid] if proba_matrix.shape[0] == count_total else proba_matrix
            except (TypeError, ValueError):
                proba_matrix = None
        del proba_all
        gc.collect()

        class_labels = [_coerce_label_key(c) for c in classes]
        cls_report = GlobalEvaluationService.evaluate_classification(
            yt_valid, yp_valid, class_labels, y_proba=proba_matrix, top_k=top_k
        )
        # Remap service-relative positions back to original row indices.
        remap = {pos: int(orig) for pos, orig in enumerate(valid_positions.tolist())}
        cls_report.all_misclassified_indices = [
            remap.get(int(i), int(i)) for i in cls_report.all_misclassified_indices
        ]
        cls_report.most_uncertain_indices = [
            remap.get(int(i), int(i)) for i in cls_report.most_uncertain_indices
        ]
        for cell in cls_report.cells:
            cell.sample_row_indices = [
                remap.get(int(i), int(i)) for i in cell.sample_row_indices
            ]

        # Multiclass guard (edge case #6): >10 classes -> top-15 confused pairs.
        if len(classes) > MAX_CLASSES_FOR_MATRIX:
            off_diag = [c for c in cls_report.cells if c.actual_class_idx != c.predicted_class_idx]
            off_diag.sort(key=lambda c: c.row_count, reverse=True)
            cls_report.cells = [c for c in cls_report.cells if c.actual_class_idx == c.predicted_class_idx] + off_diag[:15]

        return GlobalEvaluationSummary(
            task_type=task_type.value,
            evaluated_row_count=int(np.count_nonzero(valid)),
            ood_outlier_indices=ood_indices,
            ood_outlier_count=int(np.count_nonzero(max_z > 3.0)),
            prediction_distribution=distribution,
            classification=cls_report,
        )

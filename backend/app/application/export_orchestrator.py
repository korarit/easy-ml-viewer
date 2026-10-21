"""Phase 16 — export application orchestrator (enriched-table builder).

Pipeline:
1. Validate the mapping belongs to the dataset/model pair.
2. Resolve the queue (explicit ``selected_row_indices`` or the timeline
   session's stored queue) and apply ``scope`` (all / anomaly+bookmarked /
   sub-range ``[A..B]``).
3. Chunked (≤2,500-row) batch predictions over the queue only, with max
   class probability confidences (or residual fallback) and ground-truth
   actuals for ``residual_error`` / ``is_misclassified``.
4. Vectorized attributions for ``top_1 +/-`` drivers:
   - Linear models: exact ``w_i * (x_i - mean_i)`` over the whole queue.
   - Others: one median-ablated full-queue pass per feature (``D+1``
     chunked predicts), rescaled per row so contributions sum to
     ``pred - base`` — no per-row predict loop, so 50k rows stay fast.
5. Best-effort bookmarks (timeline session), auto anomaly flags
   (``TimelineEventScannerService``), and class-alias labels.

Memory (§5.1): only the queue matrix ``(N, D)`` float32 plus ``(N, D)``
contribution matrix float32 are held transiently; ``del`` + ``gc.collect()``
after each stage. The CSV path streams in 5,000-row chunks.
"""

from __future__ import annotations

import asyncio
import gc
import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.domains.export_reporting.value_objects import ExportFormat, RowScope
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
MAX_EXPORT_ROWS = 100_000
ATTRIBUTION_CHUNK = 2_500


class ExportEnrichedTableRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    session_id: str | None = None
    format: ExportFormat = "csv"
    scope: RowScope = "all_queued"
    range_start: int = Field(default=0, ge=0)
    range_end: int = Field(default=-1, ge=-1)
    selected_row_indices: list[int] = Field(default_factory=list)
    include_attribution_columns: bool = True
    include_bookmarks: bool = True


def _chunked_predict(
    adapter: Any, model_obj: Any, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """Batch predictions (+ optional probability matrix) in ≤2,500-row chunks."""
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


def _residual_confidences(preds: np.ndarray, actuals: np.ndarray | None) -> np.ndarray:
    n = int(preds.shape[0])
    if actuals is None or int(actuals.shape[0]) != n:
        return np.full((n,), 0.85, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        residuals = np.abs(preds - actuals)
        finite = residuals[np.isfinite(residuals)]
    span = float(np.max(finite) - np.min(finite)) if finite.size else 0.0
    if not np.isfinite(span) or span <= 0:
        return np.where(np.isfinite(residuals), 0.9, 0.5)
    scale = max(span * 0.25, 1e-9)
    return np.clip(1.0 / (1.0 + np.where(np.isfinite(residuals), residuals, span) / scale), 0.0, 1.0)


class ExportOrchestrator:
    def __init__(
        self,
        dataset_repo: ParquetDatasetRepository | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
    ) -> None:
        self._datasets = dataset_repo or ParquetDatasetRepository()
        self._models = model_use_case or ModelIngestUseCase()
        self._mappings = mapping_repo or InMemoryMappingRepository()

    # -- public ------------------------------------------------------
    async def build_export_data(self, req: ExportEnrichedTableRequest) -> dict[str, Any]:
        mapping = self._mappings.get(req.mapping_id)
        if mapping.dataset_id != req.dataset_id or mapping.model_id != req.model_id:
            raise DomainException(
                code="MAPPING_MISMATCH",
                message="Mapping does not belong to the given dataset/model pair",
                details={"mapping_id": req.mapping_id},
                http_status=422,
            )
        queue = self._resolve_queue(req)
        if len(queue) > MAX_EXPORT_ROWS:
            raise DomainException(
                code="EXPORT_TOO_MANY_ROWS",
                message=f"Select at most {MAX_EXPORT_ROWS} rows per export",
                http_status=422,
            )
        if not queue:
            raise DomainException(
                code="EXPORT_EMPTY_QUEUE",
                message="Export queue is empty — select at least one row",
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
                code="EXPORT_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )

        import pyarrow.parquet as pq

        parquet_path = self._datasets._require(req.dataset_id)
        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        available = set(pf.schema.names)
        for col in feature_cols:
            if col not in available:
                raise DomainException(
                    code="EXPORT_COLUMN_MISSING",
                    message=f"Column '{col}' not found in cached dataset",
                    http_status=422,
                )
        for idx in queue:
            if idx >= total_rows:
                raise DomainException(
                    code="EXPORT_ROW_OUT_OF_RANGE",
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

        # Raw feature matrix (queue only) — float32.
        df = self._datasets.load_dataframe(req.dataset_id, columns=feature_cols)
        try:
            df = df.reindex(columns=feature_cols)
            X_full = TypeCastingEngine.build_float32_matrix(df, items)
        finally:
            del df
            gc.collect()
        queue_idx = np.asarray(queue, dtype=np.int64)
        X = (
            np.ascontiguousarray(X_full[queue_idx], dtype=np.float32)
            if queue
            else np.empty((0, len(feature_cols)), dtype=np.float32)
        )
        del X_full
        gc.collect()

        # Ground-truth actuals (optional).
        actuals: np.ndarray | None = None
        if mapping.target_column_name and mapping.target_column_name in available:
            tdf = self._datasets.load_dataframe(req.dataset_id, columns=[mapping.target_column_name])
            try:
                numeric = pd.to_numeric(tdf[mapping.target_column_name], errors="coerce").to_numpy(
                    dtype=np.float64, copy=True
                )
                if np.isfinite(numeric).any():
                    actuals = numeric[queue_idx]
            finally:
                del tdf
                gc.collect()

        preds, proba = await asyncio.to_thread(_chunked_predict, adapter, model_obj, X)

        if proba is not None:
            confidences = np.max(proba, axis=1)
            confidences = np.where(np.isfinite(confidences), confidences, 0.85)
            winning = np.argmax(np.where(np.isfinite(proba), proba, 0.0), axis=1)
        else:
            confidences = _residual_confidences(preds, actuals)
            winning = None
        confidences = np.clip(confidences, 0.0, 1.0)

        # Attributions → top +/- drivers per row (vectorized, §4 above).
        top_pos_feat, top_pos_score, top_neg_feat, top_neg_score = await asyncio.to_thread(
            self._top_drivers, adapter, model_obj, X, preds, proba, winning, feature_names,
            bool(req.include_attribution_columns),
        )

        # Bookmarks + auto anomaly flags (best-effort session lookups).
        bookmarks_by_frame = self._bookmarks_by_frame(req, queue)
        anomaly_flags = self._anomaly_flags(
            X, preds, confidences, actuals, is_classification, queue
        )

        # Class-alias labels (best-effort).
        aliases = self._class_aliases(req)

        # Raw original values for the exported base columns.
        raw_rows = self._raw_rows(req.dataset_id, queue)
        base_columns = list(raw_rows[0].keys()) if raw_rows else []

        enriched_rows: list[dict[str, Any]] = []
        for t in range(len(queue)):
            pred_val = float(preds[t]) if np.isfinite(preds[t]) else 0.0
            pred_key = str(int(round(pred_val))) if is_classification else str(pred_val)
            label = aliases.get(pred_key, f"Class {pred_key}" if is_classification else pred_key)
            actual_val = float(actuals[t]) if actuals is not None and np.isfinite(actuals[t]) else None
            if actual_val is not None:
                residual = float(pred_val - actual_val)
                if is_classification:
                    mis = bool(int(round(pred_val)) != int(round(actual_val)))
                else:
                    mis = None
            else:
                residual, mis = None, None
            enriched_rows.append(
                {
                    "model_prediction": round(pred_val, 6),
                    "model_prediction_label": label,
                    "model_confidence": round(float(confidences[t]), 6),
                    "residual_error": (round(residual, 6) if residual is not None else None),
                    "is_misclassified": mis,
                    "top_positive_feature": top_pos_feat[t],
                    "top_positive_impact": round(float(top_pos_score[t]), 6),
                    "top_negative_feature": top_neg_feat[t],
                    "top_negative_impact": round(float(top_neg_score[t]), 6),
                    "auto_anomaly_flags": anomaly_flags[t],
                    "user_bookmark_note": bookmarks_by_frame.get(t, "") if req.include_bookmarks else "",
                }
            )

        del X
        gc.collect()
        return {
            "base_columns": base_columns,
            "raw_rows": raw_rows,
            "enriched_rows": enriched_rows,
            "row_indices": list(queue),
            "total_frames": len(queue),
            "is_classification": is_classification,
        }

    # -- internals -----------------------------------------------------
    def _resolve_queue(self, req: ExportEnrichedTableRequest) -> list[int]:
        if req.selected_row_indices:
            queue = [int(i) for i in req.selected_row_indices if int(i) >= 0]
        elif req.session_id:
            from app.application.timeline_orchestrator import _SESSIONS as _TL

            state = _TL.get(req.session_id)
            if state is None:
                raise DomainException(
                    code="EXPORT_SESSION_NOT_FOUND",
                    message=f"Timeline session '{req.session_id}' not found — pass selected_row_indices",
                    http_status=404,
                )
            queue = [int(i) for i in state.row_indices]
        else:
            raise DomainException(
                code="EXPORT_NO_QUEUE",
                message="Provide selected_row_indices or a timeline session_id",
                http_status=422,
            )
        if req.scope == "sub_range":
            end = req.range_end if req.range_end >= 0 else len(queue) - 1
            start = max(0, req.range_start)
            queue = queue[start : max(start, end + 1)]
        elif req.scope == "anomaly_bookmarked":
            keep = set(self._anomaly_bookmark_frames(req, queue))
            queue = [q for t, q in enumerate(queue) if t in keep]
        return queue

    def _anomaly_bookmark_frames(
        self, req: ExportEnrichedTableRequest, queue: list[int]
    ) -> set[int]:
        frames: set[int] = set()
        if req.session_id:
            try:
                from app.application.timeline_orchestrator import _SESSIONS as _TL

                state = _TL.get(req.session_id)
                if state is not None:
                    for b in state.entity.bookmarks:
                        if 0 <= int(b.frame_index) < len(queue):
                            frames.add(int(b.frame_index))
                    for e in state.entity.auto_events:
                        if 0 <= int(e.frame_index) < len(queue):
                            frames.add(int(e.frame_index))
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        return frames

    def _bookmarks_by_frame(
        self, req: ExportEnrichedTableRequest, queue: list[int]
    ) -> dict[int, str]:
        if not req.session_id:
            return {}
        try:
            from app.application.timeline_orchestrator import _SESSIONS as _TL

            state = _TL.get(req.session_id)
            if state is None:
                return {}
            out: dict[int, str] = {}
            for b in state.entity.bookmarks:
                fi = int(b.frame_index)
                if 0 <= fi < len(queue) and b.note_text:
                    out[fi] = b.note_text
            return out
        except (AttributeError, KeyError, TypeError, ValueError):
            return {}

    def _anomaly_flags(
        self,
        X: np.ndarray,
        preds: np.ndarray,
        confidences: np.ndarray,
        actuals: np.ndarray | None,
        is_classification: bool,
        queue: list[int],
    ) -> list[str]:
        try:
            from app.domains.timeline.services import TimelineEventScannerService

            events = TimelineEventScannerService.scan_events(
                X=X,
                predictions=preds,
                confidences=np.asarray(confidences, dtype=np.float64),
                actual_targets=actuals,
                is_classification=is_classification,
                row_indices=list(queue),
            )
            by_frame: dict[int, list[str]] = {}
            for e in events:
                by_frame.setdefault(int(e.frame_index), []).append(str(e.event_type))
            return ["|".join(by_frame.get(t, [])) for t in range(len(queue))]
        except (AttributeError, TypeError, ValueError):
            return ["" for _ in range(len(queue))]

    def _class_aliases(self, req: ExportEnrichedTableRequest) -> dict[str, str]:
        if not req.session_id:
            return {}
        try:
            from app.application.alias_orchestrator import _STORE as _AL

            cfg = _AL.get(req.session_id)
            if cfg is None:
                return {}
            return {str(k): str(v.label) for k, v in cfg.class_aliases.items()}
        except (AttributeError, KeyError, TypeError, ValueError):
            return {}

    def _raw_rows(self, dataset_id: str, queue: list[int]) -> list[dict[str, Any]]:
        df = self._datasets.load_dataframe(dataset_id)
        try:
            cols = list(df.columns)
            out: list[dict[str, Any]] = []
            for idx in queue:
                row = df.iloc[int(idx)]
                clean: dict[str, Any] = {}
                for c in cols:
                    v = row[c]
                    try:
                        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                            clean[str(c)] = None
                        elif isinstance(v, (np.integer,)):
                            clean[str(c)] = int(v)
                        elif isinstance(v, (np.floating,)):
                            clean[str(c)] = float(v)
                        else:
                            clean[str(c)] = v
                    except (TypeError, ValueError):
                        clean[str(c)] = str(v)
                out.append(clean)
            return out
        finally:
            del df
            gc.collect()

    @staticmethod
    def _top_drivers(
        adapter: Any,
        model_obj: Any,
        X: np.ndarray,
        preds: np.ndarray,
        proba: np.ndarray | None,
        winning: np.ndarray | None,
        feature_names: list[str],
        enabled: bool,
    ) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
        n, d = int(X.shape[0]), int(X.shape[1])
        zeros = np.zeros((n,), dtype=np.float64)
        blanks = ["" for _ in range(n)]
        if not enabled or n == 0 or d == 0:
            return blanks, zeros, list(blanks), zeros.copy()

        # Output scale mirrors the explainability orchestrator: winning-class
        # probability for classifiers, raw regression prediction otherwise.
        if winning is not None and proba is not None:
            output = np.asarray([float(proba[r, c]) for r, c in enumerate(np.asarray(winning).ravel())])
        else:
            output = np.where(np.isfinite(preds), preds, 0.0).astype(np.float64)

        contribs: np.ndarray | None = None
        try:
            from app.domains.model_engine.schema_detector import UniversalSchemaDetector as _USD

            _, final_est, _ = _USD.unwrap_model_container(model_obj)
            coef = getattr(final_est, "coef_", None)
            if coef is not None:
                w = np.asarray(coef, dtype=np.float64)
                if w.ndim == 2:
                    w = w[0] if w.shape[0] == 1 else None
                if w is not None:
                    w = np.ascontiguousarray(np.asarray(w).ravel(), dtype=np.float64)
                    if w.size == d and bool(np.all(np.isfinite(w))):
                        means = np.nanmean(X.astype(np.float64, copy=False), axis=0)
                        means = np.where(np.isfinite(means), means, 0.0)
                        contribs = (X.astype(np.float64) - means.reshape(1, -1)) * w.reshape(1, -1)
        except (TypeError, ValueError, AttributeError):
            contribs = None

        if contribs is None:
            # Black-box: one median-ablated full-queue pass per feature.
            medians = np.nanmedian(X.astype(np.float64, copy=False), axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            base = float(np.mean(output)) if n else 0.0
            contribs = np.zeros((n, d), dtype=np.float64)

            def _predict_out(batch: np.ndarray) -> np.ndarray:
                clean = np.ascontiguousarray(batch, dtype=np.float32)
                out_parts: list[np.ndarray] = []
                off = 0
                while off < clean.shape[0]:
                    chunk = clean[off : off + ATTRIBUTION_CHUNK]
                    res = adapter.predict_batch(model_obj, chunk)
                    if winning is not None and res.probabilities is not None:
                        try:
                            mat = np.asarray(res.probabilities, dtype=np.float64)
                            cols = np.asarray(winning).ravel()
                            out_parts.append(np.asarray([float(mat[r, int(c)]) for r, c in enumerate(cols)]))
                        except (TypeError, ValueError, IndexError):
                            out_parts.append(
                                np.asarray([float(v) for v in res.predictions], dtype=np.float64)
                            )
                    else:
                        out_parts.append(
                            np.asarray(
                                [float(v) if np.isfinite(float(v)) else 0.0 for v in res.predictions],
                                dtype=np.float64,
                            )
                        )
                    del chunk, res
                    gc.collect()
                    off += ATTRIBUTION_CHUNK
                return np.concatenate(out_parts) if out_parts else np.zeros((clean.shape[0],))

            try:
                for j in range(d):
                    ablated = X.astype(np.float64, copy=True)
                    ablated[:, j] = medians[j]
                    pert = _predict_out(ablated)
                    contribs[:, j] = output - np.where(np.isfinite(pert), pert, output)
                    del ablated, pert
                    gc.collect()
                # Rescale each row so contributions sum to (output - base).
                totals = contribs.sum(axis=1)
                targets = output - base
                scale = np.divide(
                    targets, totals, out=np.zeros_like(targets), where=np.abs(totals) > 1e-9
                )
                contribs = contribs * scale.reshape(-1, 1)
            except (TypeError, ValueError, MemoryError):
                contribs = np.zeros((n, d), dtype=np.float64)

        contribs = np.where(np.isfinite(contribs), contribs, 0.0)
        pos_idx = np.argmax(contribs, axis=1)
        neg_idx = np.argmin(contribs, axis=1)
        top_pos_feat = [feature_names[int(i)] if int(i) < len(feature_names) else "" for i in pos_idx]
        top_neg_feat = [feature_names[int(i)] if int(i) < len(feature_names) else "" for i in neg_idx]
        top_pos_score = contribs[np.arange(n), pos_idx].astype(np.float64)
        top_neg_score = contribs[np.arange(n), neg_idx].astype(np.float64)
        # A row with no positive (all ≤ 0) driver reports blank + 0.
        for t in range(n):
            if not bool(top_pos_score[t] > 0):
                top_pos_feat[t] = ""
                top_pos_score[t] = 0.0
            if not bool(top_neg_score[t] < 0):
                top_neg_feat[t] = ""
                top_neg_score[t] = 0.0
        del contribs
        gc.collect()
        return top_pos_feat, top_pos_score, top_neg_feat, top_neg_score

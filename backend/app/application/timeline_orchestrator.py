"""Phase 11 — timeline application orchestrator (multi-track analysis).

Pipeline:
1. Validate the mapping belongs to the dataset/model pair.
2. Chunked (≤2,500-row) feature load + batch predict over the selected
   queue rows only — bounded RAM regardless of dataset size (§5.1).
3. Derive confidences (max class probability when the adapter exposes
   ``probabilities``, else residual-based fallback) and ground-truth
   actuals (``None`` for unlabeled queues — edge case #2).
4. Vectorized :class:`TimelineEventScannerService` scan, severity-ranked
   cap at ``MAX_AUTO_EVENTS`` pins, mean-downsampled curves at
   ``MAX_TRACK_BUCKETS`` buckets.
5. Cache the payload under ``session_id``; bookmarks live on the session
   and are rebased by ``dataset_row_index`` on rebuild (edge case #6).
"""

from __future__ import annotations

import asyncio
import gc
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import TaskType
from app.domains.timeline.entities import (
    BookmarkColor,
    MultiTrackTimelineEntity,
    UserBookmarkEntity,
)
from app.domains.timeline.services import TimelineEventScannerService
from app.domains.timeline.value_objects import AutoEventPinVO
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)
from app.infrastructure.storage.parquet_dataset_repository import (
    ParquetDatasetRepository,
)

PREDICT_CHUNK_SIZE = 2_500
MAX_QUEUE_POINTS = 10_000
MAX_AUTO_EVENTS = 2_500
MAX_TRACK_BUCKETS = 2_048


class TimelineTracksBuildRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    selected_row_indices: list[int] = Field(default_factory=list)
    session_id: str | None = None


class TimelineBookmarkCreateRequest(BaseModel):
    frame_index: int = Field(ge=0)
    dataset_row_index: int = Field(ge=0)
    color: BookmarkColor = "amber"
    note_text: str = Field(default="", max_length=280)


def _mean_downsample(values: np.ndarray, max_buckets: int) -> tuple[list[float], int]:
    """Mean-bin a 1-D series to ``<= max_buckets`` buckets (NaN-aware)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = int(arr.shape[0])
    if n == 0:
        return [], 1
    bucket_size = max(1, int(np.ceil(n / max_buckets)))
    if bucket_size == 1:
        return [float(v) if np.isfinite(v) else 0.0 for v in arr], 1
    out: list[float] = []
    for start in range(0, n, bucket_size):
        block = arr[start : start + bucket_size]
        finite = block[np.isfinite(block)]
        out.append(float(np.mean(finite)) if finite.size else 0.0)
    return out, bucket_size


def _residual_confidences(
    preds: np.ndarray, actuals: np.ndarray | None
) -> np.ndarray:
    """Fallback confidences when the adapter exposes no probabilities."""
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
    conf = 1.0 / (1.0 + np.where(np.isfinite(residuals), residuals, span) / scale)
    return np.clip(conf, 0.0, 1.0)


class TimelineSessionState(BaseModel):
    entity: MultiTrackTimelineEntity
    dataset_id: str
    model_id: str
    mapping_id: str
    row_indices: list[int]


_SESSIONS: dict[str, TimelineSessionState] = {}
_SESSION_ORDER: list[str] = []
MAX_SESSIONS = 32


def _store_session(session_id: str, state: TimelineSessionState) -> None:
    _SESSIONS[session_id] = state
    if session_id in _SESSION_ORDER:
        _SESSION_ORDER.remove(session_id)
    _SESSION_ORDER.append(session_id)
    while len(_SESSION_ORDER) > MAX_SESSIONS:
        _SESSIONS.pop(_SESSION_ORDER.pop(0), None)


class TimelineOrchestrator:
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
    async def build_tracks(self, req: TimelineTracksBuildRequest) -> MultiTrackTimelineEntity:
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
                code="TIMELINE_TOO_MANY_FRAMES",
                message=f"Select at most {MAX_QUEUE_POINTS} rows per timeline",
                http_status=422,
            )
        items = sorted(mapping.items, key=lambda x: x.feature_index)
        feature_cols = [i.dataset_column_name for i in items if i.dataset_column_name]
        if not feature_cols:
            raise DomainException(
                code="TIMELINE_NO_FEATURES",
                message="Mapping has no feature columns — map at least one feature first",
                http_status=422,
            )

        parquet_path = self._datasets._require(req.dataset_id)
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path, memory_map=True)
        total_rows = int(pf.metadata.num_rows)
        all_columns = set(pf.schema.names)
        for col in feature_cols:
            if col not in all_columns:
                raise DomainException(
                    code="TIMELINE_COLUMN_MISSING",
                    message=f"Column '{col}' not found in cached dataset",
                    http_status=422,
                )
        for idx in selected:
            if idx >= total_rows:
                raise DomainException(
                    code="TIMELINE_ROW_OUT_OF_RANGE",
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

        actuals_full: np.ndarray | None = None
        if mapping.target_column_name and mapping.target_column_name in all_columns:
            tdf = self._datasets.load_dataframe(
                req.dataset_id, columns=[mapping.target_column_name]
            )
            try:
                numeric = pd.to_numeric(
                    tdf[mapping.target_column_name], errors="coerce"
                ).to_numpy(dtype=np.float64, copy=True)
                if np.isfinite(numeric).any():
                    actuals_full = numeric
            finally:
                del tdf
                gc.collect()

        queue_idx = np.asarray(selected, dtype=np.int64)
        X = np.ascontiguousarray(X_full[queue_idx], dtype=np.float32) if selected else np.empty((0, len(feature_cols)), dtype=np.float32)
        del X_full
        gc.collect()
        actuals = actuals_full[queue_idx] if actuals_full is not None else None

        preds, proba = await asyncio.to_thread(
            self._chunked_predict, adapter, model_obj, X
        )

        if proba is not None:
            confidences = np.max(proba, axis=1)
            confidences = np.where(np.isfinite(confidences), confidences, 0.85)
        else:
            confidences = _residual_confidences(preds, actuals)
        confidences = np.clip(confidences, 0.0, 1.0)

        events = TimelineEventScannerService.scan_events(
            X=X,
            predictions=preds,
            confidences=confidences,
            actual_targets=actuals,
            is_classification=is_classification,
            row_indices=selected,
        )
        # Event cap (§5.1): keep the most severe pins only.
        if len(events) > MAX_AUTO_EVENTS:
            events = sorted(events, key=lambda e: (-e.severity, e.frame_index))[
                :MAX_AUTO_EVENTS
            ]
            events.sort(key=lambda e: e.frame_index)
        event_vos: list[AutoEventPinVO] = list(events)

        pred_curve, bucket_size = _mean_downsample(preds, MAX_TRACK_BUCKETS)
        conf_curve, _ = _mean_downsample(confidences, MAX_TRACK_BUCKETS)
        actual_curve: list[float] | None = None
        if actuals is not None:
            actual_curve, _ = _mean_downsample(actuals, MAX_TRACK_BUCKETS)

        del X
        gc.collect()

        session_id = req.session_id or f"tls_{uuid.uuid4().hex[:12]}"
        # Edge case #6: rebuilds on the same session keep bookmarks, rebased
        # onto the new queue order by dataset_row_index (orphans → frame -1).
        previous = _SESSIONS.get(session_id)
        bookmarks: list[UserBookmarkEntity] = []
        if previous is not None and req.session_id is not None:
            row_to_frame = {row: t for t, row in enumerate(selected)}
            for bm in previous.entity.bookmarks:
                new_frame = row_to_frame.get(int(bm.dataset_row_index), -1)
                bookmarks.append(bm.model_copy(update={"frame_index": int(new_frame)}))

        entity = MultiTrackTimelineEntity(
            session_id=session_id,
            total_frames=len(selected),
            is_classification=is_classification,
            is_labeled=actuals is not None,
            bucket_size=bucket_size,
            prediction_curve=pred_curve,
            actual_curve=actual_curve,
            confidence_scores=conf_curve,
            auto_events=event_vos,
            bookmarks=bookmarks,
        )
        _store_session(
            session_id,
            TimelineSessionState(
                entity=entity,
                dataset_id=req.dataset_id,
                model_id=req.model_id,
                mapping_id=req.mapping_id,
                row_indices=selected,
            ),
        )
        return entity

    @staticmethod
    def get_tracks(session_id: str) -> MultiTrackTimelineEntity:
        try:
            return _SESSIONS[session_id].entity
        except KeyError as exc:
            raise DomainException(
                code="TIMELINE_SESSION_NOT_FOUND",
                message=f"Timeline session '{session_id}' not found — build it via POST /api/v1/timeline/tracks/build first",
                http_status=404,
            ) from exc

    @staticmethod
    def create_bookmark(
        session_id: str, body: TimelineBookmarkCreateRequest
    ) -> UserBookmarkEntity:
        try:
            state = _SESSIONS[session_id]
        except KeyError as exc:
            raise DomainException(
                code="TIMELINE_SESSION_NOT_FOUND",
                message=f"Timeline session '{session_id}' not found — build it via POST /api/v1/timeline/tracks/build first",
                http_status=404,
            ) from exc
        if int(body.frame_index) >= state.entity.total_frames:
            raise DomainException(
                code="TIMELINE_BOOKMARK_FRAME_OUT_OF_RANGE",
                message=(
                    f"Frame {body.frame_index} exceeds session size "
                    f"({state.entity.total_frames} frames)"
                ),
                http_status=422,
            )
        bookmark = UserBookmarkEntity(
            bookmark_id=f"bm_{uuid.uuid4().hex[:6]}",
            frame_index=int(body.frame_index),
            dataset_row_index=int(body.dataset_row_index),
            color=body.color,
            note_text=body.note_text,
            created_at_iso=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        state.entity.bookmarks.append(bookmark)
        return bookmark

    @staticmethod
    def delete_bookmark(session_id: str, bookmark_id: str) -> str:
        try:
            state = _SESSIONS[session_id]
        except KeyError as exc:
            raise DomainException(
                code="TIMELINE_SESSION_NOT_FOUND",
                message=f"Timeline session '{session_id}' not found",
                http_status=404,
            ) from exc
        remaining = [b for b in state.entity.bookmarks if b.bookmark_id != bookmark_id]
        if len(remaining) == len(state.entity.bookmarks):
            raise DomainException(
                code="TIMELINE_BOOKMARK_NOT_FOUND",
                message=f"Bookmark '{bookmark_id}' not found in session '{session_id}'",
                http_status=404,
            )
        state.entity.bookmarks = remaining
        return bookmark_id

    @staticmethod
    def clear_sessions() -> int:
        n = len(_SESSIONS)
        _SESSIONS.clear()
        _SESSION_ORDER.clear()
        return n

    # -- internals -----------------------------------------------------
    @staticmethod
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


def entity_to_response(entity: MultiTrackTimelineEntity) -> dict[str, Any]:
    return entity.model_dump(mode="json")


def bookmark_to_response(bookmark: UserBookmarkEntity) -> dict[str, Any]:
    return bookmark.model_dump(mode="json")

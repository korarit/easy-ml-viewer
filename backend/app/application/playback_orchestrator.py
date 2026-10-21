"""Phase 10 — playback application orchestrator (sequence-manifest use-case).

Pipeline:
1. Validate the mapping belongs to the dataset/model pair.
2. Reuse :class:`Stage3DUseCase` (chunked ≤2,500-row batch predictions,
   world-cube normalization) so the manifest trajectory matches the 3D
   stage exactly — no duplicated prediction math.
3. Convert trajectory points to flat :class:`FrameTrajectoryPointVO`
   records (float32 triples, 12 bytes/frame) and detect spatial
   discontinuities via :class:`TrajectoryInterpolationService`.
4. Cache the manifest under ``session_id`` so
   ``GET /api/v1/playback/{session_id}/manifest`` answers instantly.

Memory: one manifest holds ``N`` small pydantic records; a 10,000-frame
queue stays well under 5 MB. Peak extra RAM during build is bounded by
the Stage3D chunked pipeline.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.application.stage3d_use_case import InitializeSceneRequest, Stage3DUseCase
from app.core.exceptions import DomainException
from app.domains.playback.entities import PlaybackSequenceManifestEntity
from app.domains.playback.services import TrajectoryInterpolationService
from app.domains.playback.value_objects import FrameTrajectoryPointVO
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)

DEFAULT_FPS = 5.0
MAX_QUEUE_POINTS = 10_000

# Process-wide session store: session_id -> manifest (LRU-ish, capped).
_SESSION_MANIFESTS: dict[str, PlaybackSequenceManifestEntity] = {}
_SESSION_ORDER: list[str] = []
MAX_SESSIONS = 32


class PlaybackManifestRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    mapping_id: str = Field(min_length=1)
    selected_row_indices: list[int] = Field(default_factory=list)
    session_id: str | None = None
    default_fps: float = Field(default=DEFAULT_FPS, gt=0.0)


def _confidence_for_frame(
    residual: float | None, span: float, fallback: float = 0.942
) -> float:
    if residual is None or not np.isfinite(residual):
        return float(np.clip(fallback, 0.0, 1.0))
    scale = max(float(span) * 0.25, 1e-9)
    conf = 1.0 / (1.0 + abs(float(residual)) / scale)
    return float(np.clip(conf, 0.0, 1.0))


class PlaybackOrchestrator:
    def __init__(
        self,
        stage3d_use_case: Stage3DUseCase | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
    ) -> None:
        self._stage3d = stage3d_use_case or Stage3DUseCase(
            model_use_case=model_use_case,
            mapping_repo=mapping_repo,
        )
        self._mappings = mapping_repo or InMemoryMappingRepository()

    async def build_manifest(self, req: PlaybackManifestRequest) -> PlaybackSequenceManifestEntity:
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
                code="PLAYBACK_TOO_MANY_FRAMES",
                message=f"Select at most {MAX_QUEUE_POINTS} rows per playback queue",
                http_status=422,
            )

        items = sorted(mapping.items, key=lambda x: x.feature_index)
        n_features = len(items)
        if n_features < 2:
            raise DomainException(
                code="PLAYBACK_NOT_ENOUGH_FEATURES",
                message="Mapping needs at least 2 mapped features for 3D playback",
                http_status=422,
            )
        x_idx = next((i for i, it in enumerate(items) if it.dataset_column_name), 0)
        y_idx = next(
            (i for i, it in enumerate(items) if it.dataset_column_name and i != x_idx),
            min(1, n_features - 1),
        )
        if x_idx == y_idx:
            y_idx = min(1, n_features - 1)

        scene_req = InitializeSceneRequest(
            dataset_id=req.dataset_id,
            model_id=req.model_id,
            mapping_id=req.mapping_id,
            axis_x_feature_index=x_idx,
            axis_y_feature_index=y_idx,
            selected_row_indices=selected,
            grid_resolution=10,  # minimal surface: we only need trajectory points
        )
        scene = await asyncio.to_thread(self._stage3d.initialize_scene, scene_req)

        preds = np.asarray(
            [p.raw_z_pred for p in scene.trajectory_points], dtype=np.float64
        )
        finite = preds[np.isfinite(preds)]
        span = float(np.max(finite) - np.min(finite)) if finite.size else 1.0
        if not np.isfinite(span) or span <= 0:
            span = 1.0

        frames: list[FrameTrajectoryPointVO] = []
        positions = np.empty((len(scene.trajectory_points), 3), dtype=np.float32)
        for t, p in enumerate(scene.trajectory_points):
            wx = float(np.clip(p.world_x, -7.0, 7.0)) if np.isfinite(p.world_x) else 0.0
            wy = float(np.clip(p.world_y, -7.0, 7.0)) if np.isfinite(p.world_y) else 0.0
            wz = float(np.clip(p.world_z_pred, -7.0, 7.0)) if np.isfinite(p.world_z_pred) else 0.0
            positions[t] = (wx, wy, wz)
            raw_pred = float(p.raw_z_pred) if np.isfinite(p.raw_z_pred) else 0.0
            raw_actual = (
                float(p.raw_z_actual)
                if p.raw_z_actual is not None and np.isfinite(p.raw_z_actual)
                else None
            )
            frames.append(
                FrameTrajectoryPointVO(
                    frame_index=t,
                    dataset_row_index=int(p.original_row_index),
                    position_world=(wx, wy, wz),
                    prediction=raw_pred,
                    confidence=_confidence_for_frame(p.residual_error, span),
                    actual_target=raw_actual,
                )
            )

        jumps = TrajectoryInterpolationService.detect_discontinuity_jumps(positions)
        session_id = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        entity = PlaybackSequenceManifestEntity(
            session_id=session_id,
            total_frames=len(frames),
            default_fps=float(req.default_fps),
            frames_summary=frames,
            discontinuity_frames=jumps,
        )
        _SESSION_MANIFESTS[session_id] = entity
        if session_id in _SESSION_ORDER:
            _SESSION_ORDER.remove(session_id)
        _SESSION_ORDER.append(session_id)
        while len(_SESSION_ORDER) > MAX_SESSIONS:
            oldest = _SESSION_ORDER.pop(0)
            _SESSION_MANIFESTS.pop(oldest, None)
        return entity

    @staticmethod
    def get_manifest(session_id: str) -> PlaybackSequenceManifestEntity:
        try:
            return _SESSION_MANIFESTS[session_id]
        except KeyError as exc:
            raise DomainException(
                code="PLAYBACK_SESSION_NOT_FOUND",
                message=f"Playback session '{session_id}' not found — build it via POST /api/v1/playback/manifest first",
                http_status=404,
            ) from exc

    @staticmethod
    def clear_sessions() -> int:
        n = len(_SESSION_MANIFESTS)
        _SESSION_MANIFESTS.clear()
        _SESSION_ORDER.clear()
        return n


def manifest_to_response(entity: PlaybackSequenceManifestEntity) -> dict[str, Any]:
    return entity.model_dump(mode="json")

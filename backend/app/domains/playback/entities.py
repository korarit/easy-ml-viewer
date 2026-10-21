"""Phase 10 — playback sequence manifest entity."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domains.playback.value_objects import FrameTrajectoryPointVO


class PlaybackSequenceManifestEntity(BaseModel):
    session_id: str = Field(min_length=1)
    total_frames: int = Field(ge=0)
    default_fps: float = Field(default=5.0, gt=0.0)
    frames_summary: list[FrameTrajectoryPointVO] = Field(default_factory=list)
    discontinuity_frames: list[int] = Field(default_factory=list)

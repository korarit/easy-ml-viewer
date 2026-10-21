"""Phase 10 — playback value objects (per-frame trajectory point).

Memory: frames are materialized as flat ``float32`` triples (12 bytes/frame)
so a 100,000-frame queue costs ~1.2 MB of RAM (§5.1).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FrameTrajectoryPointVO(BaseModel):
    frame_index: int = Field(ge=0)
    dataset_row_index: int = Field(ge=0)
    position_world: tuple[float, float, float]
    prediction: float
    confidence: float = Field(ge=0.0, le=1.0)
    actual_target: float | None = None

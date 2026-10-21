"""Phase 11 — timeline value objects (auto-detected event pins)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal["misclassified", "decision_flip", "low_confidence", "input_outlier"]


class AutoEventPinVO(BaseModel):
    frame_index: int = Field(ge=0)
    dataset_row_index: int = Field(ge=0)
    event_type: EventType
    severity: float = Field(ge=0.0, le=1.0)
    title: str
    detail: str

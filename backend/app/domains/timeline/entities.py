"""Phase 11 — timeline entities (bookmarks + multi-track payload)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domains.timeline.value_objects import AutoEventPinVO

BookmarkColor = Literal["amber", "cyan", "emerald", "rose", "purple"]


class UserBookmarkEntity(BaseModel):
    bookmark_id: str = Field(min_length=1)
    frame_index: int = Field(ge=-1)
    dataset_row_index: int = Field(ge=0)
    color: BookmarkColor = "amber"
    note_text: str = Field(max_length=280)
    created_at_iso: str


class MultiTrackTimelineEntity(BaseModel):
    session_id: str = Field(min_length=1)
    total_frames: int = Field(ge=0)
    is_classification: bool = False
    is_labeled: bool = False
    bucket_size: int = Field(default=1, ge=1)
    prediction_curve: list[float] = Field(default_factory=list)
    actual_curve: list[float] | None = None
    confidence_scores: list[float] = Field(default_factory=list)
    auto_events: list[AutoEventPinVO] = Field(default_factory=list)
    bookmarks: list[UserBookmarkEntity] = Field(default_factory=list)

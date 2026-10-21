"""Phase 15 — recording_mlv value objects (bookmarks + auto-events)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MlvBookmarkVO(BaseModel):
    bookmark_id: str = Field(min_length=1)
    frame_index: int = Field(ge=-1)
    dataset_row_index: int = Field(ge=0)
    color: str = Field(default="amber", max_length=16)
    note_text: str = Field(default="", max_length=280)
    created_at_iso: str = ""


class MlvAutoEventVO(BaseModel):
    frame_index: int = Field(ge=0)
    kind: str = Field(min_length=1, max_length=32)
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = Field(default="", max_length=120)

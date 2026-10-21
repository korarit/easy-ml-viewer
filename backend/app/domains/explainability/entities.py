"""Phase 12 — explainability entities (per-frame local explanation)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domains.explainability.value_objects import FeatureAttributionItemVO


class FrameExplanationEntity(BaseModel):
    frame_index: int = Field(ge=0)
    dataset_row_index: int = Field(ge=0)
    base_value: float
    output_value: float
    attributions: list[FeatureAttributionItemVO] = Field(default_factory=list)
    other_contributions_sum: float = 0.0

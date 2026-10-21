"""Phase 12 — explainability value objects (local feature attribution items)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureAttributionItemVO(BaseModel):
    feature_index: int = Field(ge=0)
    feature_name: str = Field(min_length=1)
    raw_value: float
    percentile: float = Field(ge=0.0, le=100.0)
    z_score: float = 0.0
    contribution: float

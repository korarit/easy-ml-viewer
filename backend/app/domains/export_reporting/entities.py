"""Phase 16 — export_reporting entities (one enriched row per frame)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EnrichedRowEntity(BaseModel):
    frame_index: int = Field(ge=0)
    dataset_row_index: int = Field(ge=0)
    raw_values: list[float | str | None] = Field(default_factory=list)
    model_prediction: float = 0.0
    model_prediction_label: str = ""
    model_confidence: float = 0.0
    residual_error: float | None = None
    is_misclassified: bool | None = None
    top_positive_feature: str = ""
    top_positive_impact: float = 0.0
    top_negative_feature: str = ""
    top_negative_impact: float = 0.0
    auto_anomaly_flags: str = ""
    user_bookmark_note: str = ""

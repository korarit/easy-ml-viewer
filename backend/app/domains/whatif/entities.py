"""Phase 13 — what-if / counterfactual entities."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domains.whatif.value_objects import FeatureChangeDeltaVO


class WhatIfEvaluationResultEntity(BaseModel):
    base_frame_index: int = Field(ge=0)
    dataset_row_index: int = Field(ge=0)
    original_position_world: tuple[float, float, float]
    mutated_position_world: tuple[float, float, float]
    original_prediction: float
    mutated_prediction: float
    original_confidence: float = Field(ge=0.0, le=1.0)
    mutated_confidence: float = Field(ge=0.0, le=1.0)
    original_class: int | None = None
    mutated_class: int | None = None
    decision_flipped: bool = False
    world_distance: float = Field(ge=0.0)
    changed_features: list[FeatureChangeDeltaVO] = Field(default_factory=list)
    internal_model_trace: dict = Field(default_factory=dict)


class CounterfactualSearchResultEntity(BaseModel):
    found: bool
    status_message: str
    evaluation: WhatIfEvaluationResultEntity | None = None
    locked_feature_indices: list[int] = Field(default_factory=list)
    target_description: str = ""

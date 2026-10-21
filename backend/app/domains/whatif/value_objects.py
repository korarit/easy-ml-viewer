"""Phase 13 — what-if / counterfactual value objects."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureChangeDeltaVO(BaseModel):
    feature_index: int = Field(ge=0)
    feature_name: str = Field(min_length=1)
    original_value: float
    mutated_value: float
    delta_value: float
    delta_percent: float = 0.0
    is_out_of_distribution: bool = False
    is_locked: bool = False


class FeatureRangeVO(BaseModel):
    feature_index: int = Field(ge=0)
    feature_name: str = Field(min_length=1)
    min_value: float
    max_value: float
    mean_value: float = 0.0
    step: float = 0.01
    kind: str = "float"  # float | integer | binary
    locked: bool = False

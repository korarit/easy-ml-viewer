"""Model-engine bounded-context entities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domains.model_engine.value_objects import FeatureRequirement


class ModelMetadata(BaseModel):
    model_id: str
    filename: str
    sha256_hash: str
    algorithm_key: str
    display_name: str
    family_name: str
    task_type: str
    visual_3d_type: str
    is_pipeline: bool = False
    pipeline_steps: list[str] = Field(default_factory=list)
    n_features_in: int = Field(ge=1)
    classes: list[str | int | float] | None = None
    intercept_or_base: float | None = None
    features: list[FeatureRequirement] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

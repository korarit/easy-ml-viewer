"""Phase 09 — projection value objects (Pydantic schemas)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProjectionMode(str, Enum):
    AXIS_EXPLORER = "axis_explorer"
    PCA_3D = "pca_3d"


class ZMode(str, Enum):
    PREDICTION = "prediction"
    PROBABILITY = "probability"
    RESIDUAL_ERROR = "residual_error"
    FEATURE_3 = "feature_3"


class PinningMode(str, Enum):
    MEDIAN = "median"
    MEAN = "mean"
    ZERO = "zero"
    CURRENT_ROW = "current_row"


class LodLevel(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"

    @property
    def grid_resolution(self) -> int:
        return {"low": 30, "standard": 60, "high": 100}[self.value]


class ProjectionConfigVO(BaseModel):
    mode: ProjectionMode = ProjectionMode.AXIS_EXPLORER
    x_feature: str = Field(min_length=1)
    y_feature: str = Field(min_length=1)
    z_mode: ZMode = ZMode.PREDICTION
    target_class_index: int = Field(default=0, ge=0)
    pinning_mode: PinningMode = PinningMode.MEDIAN
    lod_level: LodLevel = LodLevel.STANDARD
    current_frame_index: int | None = None
    use_robust_range: bool = False

    @property
    def grid_resolution(self) -> int:
        return self.lod_level.grid_resolution

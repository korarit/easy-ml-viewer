"""Phase 09 — surface mesh entity (serializable projection result)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SurfaceMesh3DEntity(BaseModel):
    grid_resolution: int
    x_feature: str
    y_feature: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    z_heights_world: list[float] = Field(default_factory=list)
    raw_predictions: list[float] = Field(default_factory=list)
    point_positions_world: list[list[float]] = Field(default_factory=list)
    point_row_indices: list[int] = Field(default_factory=list)
    trajectory_points: list[dict[str, Any]] = Field(default_factory=list)
    pca_variance_ratio: list[float] | None = None
    pca_loadings: list[list[float]] | None = None
    resolved_z_mode: str = "prediction"
    has_constant_axis_warning: bool = False
    computation_time_ms: float = 0.0
    dataset_row_count: int = 0

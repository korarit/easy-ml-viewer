"""Phase 49 — Universal 2D Point Intelligence DTO schemas (spec §3).

Foundational point-intelligence payload for the 2D planar studio:
hexagonal density/error bins plus empirical density-contour levels.
Frontend renders comet trails / dual-rings / spokes client-side from
the trajectory queue; the backend owns the O(N) aggregation (hexbin +
KDE-lite contours) under the <15 ms DoD budget.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HexbinCell2D(BaseModel):
    center_x: float
    center_y: float
    radius: float
    count: int = Field(ge=0)
    dominant_class: int | None = None
    mean_residual: float = 0.0
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class DensityContourLevel2D(BaseModel):
    density_threshold: float
    segments: list[list[float]] = Field(
        default_factory=list,
        description="Isoline segments as [x1, y1, x2, y2] in world coords",
    )


class PointIntelligenceResponse(BaseModel):
    hexbin_cells: list[HexbinCell2D] = Field(default_factory=list)
    density_contours: list[DensityContourLevel2D] = Field(default_factory=list)
    max_bin_count: int = 0
    computation_time_ms: float = 0.0

"""Phase 09 — 3D projection REST endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.application.projection_orchestrator import (
    ProjectionContourRequest,
    ProjectionKnnFrameRequest,
    ProjectionOrchestrator,
    ProjectionSurfaceRequest,
)

router = APIRouter(prefix="/projection", tags=["projection"])


def get_projection_orchestrator() -> ProjectionOrchestrator:
    return ProjectionOrchestrator()


@router.post("/surface-mesh", summary="Phase 09: LOD decision surface + PCA manifold")
async def compute_surface_mesh(
    body: ProjectionSurfaceRequest,
    orchestrator: ProjectionOrchestrator = Depends(get_projection_orchestrator),
) -> dict[str, Any]:
    entity = await orchestrator.build_surface(body)
    return entity.model_dump(mode="json")


@router.post("/contour-2d", summary="Phase 41: dense 2D decision-boundary + isoline contours")
async def compute_contour_2d(
    body: ProjectionContourRequest,
    orchestrator: ProjectionOrchestrator = Depends(get_projection_orchestrator),
) -> dict[str, Any]:
    entity = await orchestrator.build_contour(body)
    return entity.model_dump(mode="json")


@router.post("/knn-frame-2d", summary="Phase 43: per-frame KNN radar state for the 2D viewport")
async def compute_knn_frame_2d(
    body: ProjectionKnnFrameRequest,
    orchestrator: ProjectionOrchestrator = Depends(get_projection_orchestrator),
) -> dict[str, Any]:
    entity = await orchestrator.build_knn_frame(body)
    return entity.model_dump(mode="json")

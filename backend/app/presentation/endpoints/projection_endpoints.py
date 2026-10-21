"""Phase 09 — 3D projection REST endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.application.projection_orchestrator import (
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

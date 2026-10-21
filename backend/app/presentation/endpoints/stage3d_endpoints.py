"""Phase 08 — 3D stage REST endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.application.stage3d_use_case import (
    InitializeSceneRequest,
    Stage3DUseCase,
)

router = APIRouter(prefix="/stage3d", tags=["stage3d"])


def get_stage3d_use_case() -> Stage3DUseCase:
    return Stage3DUseCase()


@router.post("/initialize-scene", summary="Phase 08: normalize axes + surface + trajectory points")
async def initialize_scene(
    body: InitializeSceneRequest,
    use_case: Stage3DUseCase = Depends(get_stage3d_use_case),
) -> dict[str, Any]:
    response = use_case.initialize_scene(body)
    return response.model_dump(mode="json")

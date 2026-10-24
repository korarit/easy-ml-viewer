"""Versioned API router (/api/v1). Phases 01-03: system + dataset + models."""

from __future__ import annotations

from fastapi import APIRouter

from app.presentation.endpoints.dataset_endpoints import router as dataset_router
from app.presentation.endpoints.dataset_query_endpoints import router as dataset_query_router
from app.presentation.endpoints.evaluation_endpoints import router as evaluation_router
from app.presentation.endpoints.explainability_endpoints import router as explainability_router
from app.presentation.endpoints.mapping_endpoints import router as mapping_router
from app.presentation.endpoints.model_endpoints import router as model_router
from app.presentation.endpoints.playback_endpoints import router as playback_router
from app.presentation.endpoints.projection_endpoints import router as projection_router
from app.presentation.endpoints.stage3d_endpoints import router as stage3d_router
from app.presentation.endpoints.alias_endpoints import router as alias_router
from app.presentation.endpoints.compute_endpoints import router as compute_router
from app.presentation.endpoints.system_endpoints import router as system_router
from app.presentation.endpoints.export_endpoints import router as export_router
from app.presentation.endpoints.video_export_endpoints import router as video_export_router
from app.presentation.endpoints.recording_endpoints import router as recording_router
from app.presentation.endpoints.timeline_endpoints import router as timeline_router
from app.presentation.endpoints.whatif_endpoints import router as whatif_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system_router)
api_router.include_router(dataset_router)
api_router.include_router(dataset_query_router)
api_router.include_router(model_router)
api_router.include_router(mapping_router)
api_router.include_router(evaluation_router)
api_router.include_router(explainability_router)
api_router.include_router(stage3d_router)
api_router.include_router(projection_router)
api_router.include_router(playback_router)
api_router.include_router(timeline_router)
api_router.include_router(recording_router)
api_router.include_router(whatif_router)
api_router.include_router(alias_router)
api_router.include_router(export_router)
api_router.include_router(video_export_router)
api_router.include_router(compute_router)

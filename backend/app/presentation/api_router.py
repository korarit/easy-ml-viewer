"""Versioned API router (/api/v1). Phases 01-03: system + dataset + models."""

from __future__ import annotations

from fastapi import APIRouter

from app.presentation.endpoints.dataset_endpoints import router as dataset_router
from app.presentation.endpoints.dataset_query_endpoints import router as dataset_query_router
from app.presentation.endpoints.evaluation_endpoints import router as evaluation_router
from app.presentation.endpoints.mapping_endpoints import router as mapping_router
from app.presentation.endpoints.model_endpoints import router as model_router
from app.presentation.endpoints.stage3d_endpoints import router as stage3d_router
from app.presentation.endpoints.system_endpoints import router as system_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system_router)
api_router.include_router(dataset_router)
api_router.include_router(dataset_query_router)
api_router.include_router(model_router)
api_router.include_router(mapping_router)
api_router.include_router(evaluation_router)
api_router.include_router(stage3d_router)

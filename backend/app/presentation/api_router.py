"""Versioned API router (/api/v1). Phases 01-02: system + dataset."""

from __future__ import annotations

from fastapi import APIRouter

from app.presentation.endpoints.dataset_endpoints import router as dataset_router
from app.presentation.endpoints.system_endpoints import router as system_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system_router)
api_router.include_router(dataset_router)

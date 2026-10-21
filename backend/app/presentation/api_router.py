"""Versioned API router (/api/v1). Phase 01 exposes system only."""

from __future__ import annotations

from fastapi import APIRouter

from app.presentation.endpoints.system_endpoints import router as system_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system_router)

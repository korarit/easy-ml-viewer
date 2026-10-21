"""System REST endpoints — /api/v1/system/*."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.application.system_service import SystemOrchestratorService
from app.core.memory_monitor import memory_monitor
from app.domains.system.value_objects import CacheClearResult, MemorySnapshot, SystemHealthStatus
from app.infrastructure.filesystem.cache_manager import CacheManager

router = APIRouter(prefix="/system", tags=["system"])


def get_system_service() -> SystemOrchestratorService:
    return SystemOrchestratorService(
        memory_monitor=memory_monitor,
        cache_manager=CacheManager(),
    )


@router.get("/health", response_model=SystemHealthStatus, summary="Backend + hardware readiness")
def get_health(service: SystemOrchestratorService = Depends(get_system_service)) -> SystemHealthStatus:
    return service.get_health_status()


@router.post("/gc", response_model=MemorySnapshot, summary="Run Python garbage collection")
def run_gc(service: SystemOrchestratorService = Depends(get_system_service)) -> MemorySnapshot:
    return service.force_garbage_collection()


@router.delete("/cache", response_model=CacheClearResult, summary="Purge .cache/ directory")
def clear_cache(
    service: SystemOrchestratorService = Depends(get_system_service),
) -> CacheClearResult:
    return service.clear_cache()

"""Application orchestrator for the System bounded context."""

from __future__ import annotations

import gc
import os
import platform

from app.core.config import settings
from app.core.events import CacheClearedEvent, GarbageCollectedEvent, event_bus
from app.core.memory_monitor import MemoryMonitor
from app.domains.system.value_objects import CacheClearResult, MemorySnapshot, SystemHealthStatus
from app.infrastructure.filesystem.cache_manager import CacheManager


class SystemOrchestratorService:
    def __init__(self, memory_monitor: MemoryMonitor, cache_manager: CacheManager) -> None:
        self._memory_monitor = memory_monitor
        self._cache_manager = cache_manager

    def get_health_status(self) -> SystemHealthStatus:
        snap = self._memory_monitor.capture_snapshot()
        cache_mb = self._cache_manager.get_total_cache_size_mb()
        return SystemHealthStatus(
            status="degraded" if snap["is_low_memory"] else "healthy",
            os_platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            python_version=platform.python_version(),
            cpu_logical_cores=os.cpu_count() or 1,
            omp_threads=settings.OMP_NUM_THREADS,
            memory=MemorySnapshot(**snap),  # type: ignore[arg-type]
            cache_size_mb=cache_mb,
        )

    def force_garbage_collection(self) -> MemorySnapshot:
        gc.collect()
        snap = self._memory_monitor.capture_snapshot()
        event_bus.publish(GarbageCollectedEvent(process_rss_mb=float(snap["process_rss_mb"])))
        return MemorySnapshot(**snap)  # type: ignore[arg-type]

    def clear_cache(self) -> CacheClearResult:
        freed_mb, removed = self._cache_manager.clear_cache()
        event_bus.publish(CacheClearedEvent(freed_mb=freed_mb))
        remaining = self._cache_manager.get_total_cache_size_mb()
        return CacheClearResult(freed_mb=freed_mb, remaining_mb=remaining, files_removed=removed)

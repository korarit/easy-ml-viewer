"""Process + system RAM telemetry and pre-allocation budget gate."""

from __future__ import annotations

import gc
import os

import psutil

from app.core.config import settings
from app.core.exceptions import MemoryLimitExceededError


class MemoryMonitor:
    def __init__(
        self,
        max_process_ram_mb: int | None = None,
        min_system_free_ram_mb: int | None = None,
    ) -> None:
        self._max_process_ram_mb = max_process_ram_mb or settings.MAX_PROCESS_RAM_MB
        self._min_free_ram_mb = min_system_free_ram_mb or settings.MIN_SYSTEM_FREE_RAM_MB
        self._process = psutil.Process(os.getpid())

    @property
    def budget_mb(self) -> float:
        return float(self._max_process_ram_mb)

    def process_rss_mb(self) -> float:
        return float(self._process.memory_info().rss) / (1024 * 1024)

    def system_snapshot(self) -> tuple[float, float]:
        vm = psutil.virtual_memory()
        total_mb = float(vm.total) / (1024 * 1024)
        available_mb = float(vm.available) / (1024 * 1024)
        return total_mb, available_mb

    def is_low_memory(self, available_mb: float) -> bool:
        return available_mb < float(self._min_free_ram_mb)

    def capture_snapshot(self) -> dict[str, object]:
        rss = self.process_rss_mb()
        total_mb, available_mb = self.system_snapshot()
        usage_pct = (rss / float(self._max_process_ram_mb) * 100.0) if self._max_process_ram_mb else 0.0
        return {
            "process_rss_mb": round(rss, 2),
            "process_budget_mb": float(self._max_process_ram_mb),
            "process_usage_percent": round(usage_pct, 2),
            "system_total_mb": round(total_mb, 2),
            "system_available_mb": round(available_mb, 2),
            "is_low_memory": self.is_low_memory(available_mb),
        }

    def assert_memory_budget(self, required_bytes: int) -> None:
        """Raise MemoryLimitExceededError instead of letting the OS OOM-freeze."""
        required_mb = float(required_bytes) / (1024 * 1024)
        rss = self.process_rss_mb()
        _, available_mb = self.system_snapshot()
        if rss + required_mb > float(self._max_process_ram_mb) or (
            available_mb - required_mb
        ) < float(self._min_free_ram_mb):
            gc.collect()  # one opportunistic pass before failing
            rss_after = self.process_rss_mb()
            _, avail_after = self.system_snapshot()
            if rss_after + required_mb > float(self._max_process_ram_mb) or (
                avail_after - required_mb
            ) < float(self._min_free_ram_mb):
                raise MemoryLimitExceededError(
                    required_mb=required_mb,
                    available_mb=avail_after,
                )


memory_monitor = MemoryMonitor()

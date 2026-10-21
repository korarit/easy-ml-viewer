"""Phase 17 — compute_engine entities (framework-free domain state)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComputeJobEntity:
    """A registered chunked simulation job (progress tracked by orchestrator)."""

    job_id: str
    total_rows: int
    chunk_size: int = 5000
    processed_rows: int = 0
    status: str = "running"  # running | cancelled | completed
    device_mode: str = "auto_smart_router"
    route: str = "cpu_vectorized"

    @property
    def progress_percent(self) -> float:
        if self.total_rows <= 0:
            return 100.0
        return round(min(100.0, (self.processed_rows / self.total_rows) * 100.0), 1)

    @property
    def chunks_total(self) -> int:
        if self.total_rows <= 0:
            return 0
        return (self.total_rows + self.chunk_size - 1) // self.chunk_size

    @property
    def chunks_done(self) -> int:
        if self.chunk_size <= 0:
            return 0
        return min(self.chunks_total, self.processed_rows // self.chunk_size)


@dataclass
class ComputeDeviceConfig:
    """Mutable engine preference (kept in orchestrator, not per request)."""

    device_mode: str = "auto_smart_router"
    chunk_size: int = 5000
    extra: dict[str, str] = field(default_factory=dict)

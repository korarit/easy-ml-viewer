"""System bounded-context value objects (Pydantic schemas)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MemorySnapshot(BaseModel):
    process_rss_mb: float = Field(ge=0)
    process_budget_mb: float = Field(gt=0)
    process_usage_percent: float = Field(ge=0)
    system_total_mb: float = Field(gt=0)
    system_available_mb: float = Field(ge=0)
    is_low_memory: bool


class SystemHealthStatus(BaseModel):
    status: str = Field(pattern="^(healthy|degraded)$")
    os_platform: str
    python_version: str
    cpu_logical_cores: int = Field(ge=1)
    omp_threads: int = Field(ge=1)
    memory: MemorySnapshot
    cache_size_mb: float = Field(ge=0)


class CacheClearResult(BaseModel):
    freed_mb: float = Field(ge=0)
    remaining_mb: float = Field(ge=0)
    files_removed: int = Field(ge=0)

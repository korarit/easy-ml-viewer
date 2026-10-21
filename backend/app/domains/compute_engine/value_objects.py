"""Phase 17 — compute_engine value objects (Pydantic REST schemas)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DeviceMode = Literal["auto_smart_router", "force_cpu_only", "prefer_gpu"]
ExecutionRoute = Literal["cpu_vectorized", "gpu_accelerated"]
MemoryGuardLevel = Literal["normal", "elevated", "guard_active"]
ChunkSizeOption = Literal[2500, 5000, 10000]


class ComputeTelemetryVO(BaseModel):
    physical_cores: int = Field(ge=1)
    logical_cores: int = Field(ge=1)
    optimal_workers: int = Field(ge=1)
    available_onnx_providers: list[str] = Field(default_factory=list)
    active_device_mode: DeviceMode
    process_rss_mb: float = Field(ge=0)
    system_available_mb: float = Field(ge=0)
    system_memory_percent: float = Field(ge=0, le=100)
    memory_guard_level: MemoryGuardLevel
    active_jobs_count: int = Field(ge=0)
    default_chunk_size: int = Field(default=5000, ge=1)


class ComputeJobCancelResultVO(BaseModel):
    job_id: str
    cancelled: bool
    status: str


class ComputeCachePurgeResultVO(BaseModel):
    freed_mb: float = Field(ge=0)
    remaining_cache_mb: float = Field(ge=0)
    gc_collected: bool = True


class ComputeJobEstimateRequestVO(BaseModel):
    total_rows: int = Field(ge=0)
    n_features: int = Field(default=1, ge=1)
    is_deep_model: bool = False
    device_mode: DeviceMode = "auto_smart_router"
    chunk_size: int = Field(default=5000, ge=100, le=100_000)


class ComputeJobEstimateVO(BaseModel):
    route: ExecutionRoute
    num_chunks: int = Field(ge=0)
    chunk_size: int = Field(ge=1)
    estimated_peak_extra_mb: float = Field(ge=0)

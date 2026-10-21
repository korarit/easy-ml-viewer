"""Phase 17 — compute job orchestrator (telemetry, cancel, purge, routing).

Application layer over :mod:`app.core.compute_pool` +
:mod:`app.domains.compute_engine`: keeps one process-wide
``AdaptiveComputeEngine`` singleton, a registry of cooperative
``CancellationToken`` jobs, and the operator-chosen device/chunk config.
"""

from __future__ import annotations

import gc
import threading
import uuid

import psutil
from pydantic import BaseModel, Field

from app.core.compute_pool import (
    DEFAULT_CHUNK_SIZE,
    CancellationToken,
    get_compute_engine,
)
from app.domains.compute_engine.entities import ComputeDeviceConfig, ComputeJobEntity
from app.domains.compute_engine.services import (
    estimate_chunks,
    estimate_peak_extra_mb,
    memory_guard_level,
)
from app.domains.compute_engine.value_objects import (
    ComputeCachePurgeResultVO,
    ComputeJobCancelResultVO,
    ComputeJobEstimateVO,
    ComputeTelemetryVO,
    DeviceMode,
)

_VALID_MODES: tuple[str, ...] = ("auto_smart_router", "force_cpu_only", "prefer_gpu")
_VALID_CHUNKS: tuple[int, ...] = (2500, 5000, 10000)

_MODE_TO_ENGINE: dict[str, str] = {
    "auto_smart_router": "auto",
    "force_cpu_only": "cpu_only",
    "prefer_gpu": "prefer_gpu",
}


class ComputeConfigRequest(BaseModel):
    device_mode: DeviceMode = "auto_smart_router"
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=100, le=100_000)


class ComputeJobRegisterRequest(BaseModel):
    total_rows: int = Field(ge=0)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=100, le=100_000)
    device_mode: DeviceMode = "auto_smart_router"
    is_deep_model: bool = False
    job_id: str | None = None


class ComputeJobOrchestrator:
    def __init__(self) -> None:
        self._engine = get_compute_engine()
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}
        self._config = ComputeDeviceConfig()

    # -- config ------------------------------------------------------
    def update_config(self, device_mode: DeviceMode, chunk_size: int) -> ComputeDeviceConfig:
        if device_mode not in _VALID_MODES:
            from app.core.exceptions import DomainException as _DE

            raise _DE(
                code="COMPUTE_INVALID_DEVICE_MODE",
                message=f"Unknown device mode '{device_mode}'",
                details={"valid": list(_VALID_MODES)},
                http_status=422,
            )
        if chunk_size <= 0:
            from app.core.exceptions import DomainException as _DE

            raise _DE(
                code="COMPUTE_INVALID_CHUNK_SIZE",
                message="chunk_size must be a positive integer",
                http_status=422,
            )
        with self._lock:
            self._config.device_mode = device_mode
            self._config.chunk_size = int(chunk_size)
            return self._config

    # -- telemetry ---------------------------------------------------
    def get_telemetry(self) -> ComputeTelemetryVO:
        engine = self._engine
        mem = engine.check_memory_budget(required_extra_bytes=1 * 1024 * 1024)
        with self._lock:
            active = sum(
                1 for j in self._jobs.values() if getattr(j["entity"], "status", "") == "running"
            )
            mode: DeviceMode = self._config.device_mode  # type: ignore[assignment]
            chunk = int(self._config.chunk_size)
        return ComputeTelemetryVO(
            physical_cores=engine.physical_cores,
            logical_cores=engine.logical_cores,
            optimal_workers=engine.optimal_workers,
            available_onnx_providers=list(engine.available_onnx_providers),
            active_device_mode=mode,
            process_rss_mb=float(mem["process_rss_mb"]),
            system_available_mb=float(mem["system_available_mb"]),
            system_memory_percent=float(mem["system_percent"]),
            memory_guard_level=memory_guard_level(float(mem["system_percent"])),
            active_jobs_count=int(active),
            default_chunk_size=chunk,
        )

    # -- job registry (cooperative cancellation) ---------------------
    def register_job(
        self,
        total_rows: int,
        chunk_size: int | None = None,
        device_mode: DeviceMode | None = None,
        is_deep_model: bool = False,
        job_id: str | None = None,
    ) -> ComputeJobEntity:
        engine = self._engine
        with self._lock:
            mode = device_mode or self._config.device_mode
            size = int(chunk_size or self._config.chunk_size)
        route = engine.route_workload(
            n_rows=int(total_rows),
            is_deep_model=bool(is_deep_model),
            device_mode=_MODE_TO_ENGINE.get(str(mode), "auto"),  # type: ignore[arg-type]
        )
        entity = ComputeJobEntity(
            job_id=job_id or f"job_{uuid.uuid4().hex[:12]}",
            total_rows=int(total_rows),
            chunk_size=size,
            device_mode=str(mode),
            route=str(route),
        )
        with self._lock:
            self._jobs[entity.job_id] = {"entity": entity, "token": CancellationToken()}
        return entity

    def report_progress(self, job_id: str, processed_rows: int) -> ComputeJobEntity:
        with self._lock:
            try:
                entry = self._jobs[job_id]
            except KeyError as exc:
                from app.core.exceptions import DomainException as _DE

                raise _DE(
                    code="COMPUTE_JOB_NOT_FOUND",
                    message=f"Compute job '{job_id}' not found",
                    http_status=404,
                ) from exc
            entity = entry["entity"]
            assert isinstance(entity, ComputeJobEntity)
            entity.processed_rows = max(0, min(int(processed_rows), entity.total_rows))
            if entity.total_rows > 0 and entity.processed_rows >= entity.total_rows:
                entity.status = "completed"
            return entity

    def cancel_job(self, job_id: str) -> ComputeJobCancelResultVO:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return ComputeJobCancelResultVO(
                    job_id=job_id, cancelled=False, status="not_found"
                )
            entity = entry["entity"]
            token = entry["token"]
            assert isinstance(entity, ComputeJobEntity)
            assert isinstance(token, CancellationToken)
            if entity.status != "running":
                return ComputeJobCancelResultVO(
                    job_id=job_id, cancelled=False, status=entity.status
                )
            token.cancel()
            entity.status = "cancelled"
            return ComputeJobCancelResultVO(
                job_id=job_id, cancelled=True, status="cancelled"
            )

    def get_job_token(self, job_id: str) -> CancellationToken | None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return None
            token = entry["token"]
            return token if isinstance(token, CancellationToken) else None

    # -- estimate / purge --------------------------------------------
    def estimate(
        self, total_rows: int, n_features: int, is_deep_model: bool, chunk_size: int
    ) -> ComputeJobEstimateVO:
        with self._lock:
            mode = self._config.device_mode
        route = self._engine.route_workload(
            n_rows=int(total_rows),
            is_deep_model=bool(is_deep_model),
            device_mode=_MODE_TO_ENGINE.get(str(mode), "auto"),  # type: ignore[arg-type]
        )
        size = int(chunk_size or self._config.chunk_size)
        return ComputeJobEstimateVO(
            route=route,  # type: ignore[arg-type]
            num_chunks=estimate_chunks(int(total_rows), size),
            chunk_size=size,
            estimated_peak_extra_mb=estimate_peak_extra_mb(int(n_features), size),
        )

    def purge_caches(self) -> ComputeCachePurgeResultVO:
        proc = psutil.Process()
        before_mb = float(proc.memory_info().rss) / (1024 * 1024)
        gc.collect()
        after_mb = float(proc.memory_info().rss) / (1024 * 1024)
        freed = max(0.0, round(before_mb - after_mb, 2))
        return ComputeCachePurgeResultVO(
            freed_mb=freed,
            remaining_cache_mb=round(after_mb, 2),
            gc_collected=True,
        )


_orchestrator: ComputeJobOrchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_compute_orchestrator() -> ComputeJobOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = ComputeJobOrchestrator()
    return _orchestrator

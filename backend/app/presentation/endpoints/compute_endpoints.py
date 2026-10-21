"""Phase 17 — compute & performance REST endpoints (/api/v1/compute/*)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.application.compute_job_orchestrator import (
    ComputeConfigRequest,
    ComputeJobOrchestrator,
    ComputeJobRegisterRequest,
    get_compute_orchestrator,
)
from app.domains.compute_engine.value_objects import (
    ComputeCachePurgeResultVO,
    ComputeJobCancelResultVO,
    ComputeJobEstimateRequestVO,
    ComputeJobEstimateVO,
    ComputeTelemetryVO,
)

router = APIRouter(prefix="/compute", tags=["compute-engine"])


def get_compute_service() -> ComputeJobOrchestrator:
    return get_compute_orchestrator()


@router.get("/telemetry", response_model=ComputeTelemetryVO, summary="Phase 17: hardware telemetry")
def get_compute_telemetry(
    service: ComputeJobOrchestrator = Depends(get_compute_service),
) -> ComputeTelemetryVO:
    return service.get_telemetry()


@router.post("/config", summary="Phase 17: execution device + chunk-size preference")
def update_compute_config(
    body: ComputeConfigRequest,
    service: ComputeJobOrchestrator = Depends(get_compute_service),
) -> dict[str, object]:
    cfg = service.update_config(body.device_mode, int(body.chunk_size))
    return {"device_mode": cfg.device_mode, "chunk_size": cfg.chunk_size}


@router.post("/jobs", summary="Phase 17: register a chunked compute job")
def register_compute_job(
    body: ComputeJobRegisterRequest,
    service: ComputeJobOrchestrator = Depends(get_compute_service),
) -> dict[str, object]:
    entity = service.register_job(
        total_rows=int(body.total_rows),
        chunk_size=int(body.chunk_size),
        device_mode=body.device_mode,
        is_deep_model=bool(body.is_deep_model),
        job_id=body.job_id,
    )
    return {
        "job_id": entity.job_id,
        "total_rows": entity.total_rows,
        "chunk_size": entity.chunk_size,
        "chunks_total": entity.chunks_total,
        "device_mode": entity.device_mode,
        "route": entity.route,
        "status": entity.status,
    }


@router.post("/jobs/{job_id}/cancel", response_model=ComputeJobCancelResultVO, summary="Phase 17: cancel a chunked job")
def cancel_compute_job(
    job_id: str,
    service: ComputeJobOrchestrator = Depends(get_compute_service),
) -> ComputeJobCancelResultVO:
    return service.cancel_job(job_id)


@router.post("/estimate", response_model=ComputeJobEstimateVO, summary="Phase 17: pre-flight chunk/route estimate")
def estimate_compute_job(
    body: ComputeJobEstimateRequestVO,
    service: ComputeJobOrchestrator = Depends(get_compute_service),
) -> ComputeJobEstimateVO:
    return service.estimate(
        total_rows=int(body.total_rows),
        n_features=int(body.n_features),
        is_deep_model=bool(body.is_deep_model),
        chunk_size=int(body.chunk_size),
    )


@router.post("/cache/purge", response_model=ComputeCachePurgeResultVO, summary="Phase 17: clear inactive caches + GC")
def purge_compute_caches(
    service: ComputeJobOrchestrator = Depends(get_compute_service),
) -> ComputeCachePurgeResultVO:
    return service.purge_caches()

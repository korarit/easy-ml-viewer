"""Phase 17 acceptance: compute pool, chunking, cancel + telemetry API."""

from __future__ import annotations

import numpy as np
import psutil
from fastapi.testclient import TestClient

from app.core.compute_pool import (
    AdaptiveComputeEngine,
    CancellationToken,
    JobCancelledError,
    get_compute_engine,
)
from app.domains.compute_engine.services import estimate_chunks, memory_guard_level
from app.main import app

client = TestClient(app)


def test_engine_topology_and_workers():
    engine = AdaptiveComputeEngine()
    physical = int(psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 2)
    assert engine.physical_cores == physical
    assert engine.logical_cores >= 1
    assert engine.optimal_workers == max(1, physical - 1)
    assert "CPUExecutionProvider" in engine.available_onnx_providers
    assert engine.get_preferred_onnx_providers(prefer_gpu=False) == ["CPUExecutionProvider"]


def test_singleton_engine_shared_pool():
    assert get_compute_engine() is get_compute_engine()


def test_chunked_generator_reports_progress():
    engine = AdaptiveComputeEngine()
    X = np.arange(12_000 * 4, dtype=np.float32).reshape(12_000, 4)
    seen: list[float] = []
    out = list(
        engine.process_in_chunks(
            X,
            lambda chunk, s, e: (s, e),
            chunk_size=5_000,
            on_progress=lambda done, total, pct: seen.append(pct),
        )
    )
    assert out == [(0, 5000), (5000, 10000), (10000, 12000)]
    assert seen[-1] == 100.0
    assert estimate_chunks(50_000, 5_000) == 10


def test_chunked_generator_empty_and_cancel():
    engine = AdaptiveComputeEngine()
    empty = np.zeros((0, 3), dtype=np.float32)
    assert list(engine.process_in_chunks(empty, lambda c, s, e: e)) == []

    X = np.ones((12_000, 2), dtype=np.float32)
    token = CancellationToken()
    token.cancel()
    try:
        list(engine.process_in_chunks(X, lambda c, s, e: e, chunk_size=5_000, cancel_token=token))
        raise AssertionError("expected JobCancelledError")
    except JobCancelledError:
        pass

    # Mid-stream cancel after first chunk.
    token2 = CancellationToken()
    calls = {"n": 0}

    def _fn(chunk, s, e):
        calls["n"] += 1
        if calls["n"] == 1:
            token2.cancel()
        return e

    try:
        list(engine.process_in_chunks(X, _fn, chunk_size=5_000, cancel_token=token2))
        raise AssertionError("expected JobCancelledError on second chunk")
    except JobCancelledError:
        assert calls["n"] == 1


def test_memory_budget_and_guard_levels():
    engine = AdaptiveComputeEngine()
    snap = engine.check_memory_budget(required_extra_bytes=1024)
    assert snap["process_rss_mb"] > 0
    assert snap["system_available_mb"] > 0
    assert memory_guard_level(59.9) == "normal"
    assert memory_guard_level(70.0) == "elevated"
    assert memory_guard_level(85.0) == "guard_active"


def test_cpu_gpu_router_prefers_cpu_for_small_tabular():
    engine = AdaptiveComputeEngine()
    assert engine.route_workload(100, is_deep_model=False) == "cpu_vectorized"
    assert engine.route_workload(49_999, is_deep_model=False) == "cpu_vectorized"


def test_telemetry_contract():
    res = client.get("/api/v1/compute/telemetry")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["optimal_workers"] == max(1, body["physical_cores"] - 1)
    assert "CPUExecutionProvider" in body["available_onnx_providers"]
    assert body["active_device_mode"] == "auto_smart_router"
    assert body["process_rss_mb"] > 0
    assert body["system_available_mb"] > 0
    assert body["memory_guard_level"] in ("normal", "elevated", "guard_active")
    assert body["active_jobs_count"] >= 0


def test_job_register_cancel_lifecycle():
    reg = client.post(
        "/api/v1/compute/jobs",
        json={"total_rows": 12_000, "chunk_size": 5000, "device_mode": "auto_smart_router"},
    )
    assert reg.status_code == 200, reg.text
    job_id = reg.json()["job_id"]
    assert reg.json()["chunks_total"] == 3

    cancel = client.post(f"/api/v1/compute/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json() == {"job_id": job_id, "cancelled": True, "status": "cancelled"}

    # Second cancel is idempotent — job already terminal.
    again = client.post(f"/api/v1/compute/jobs/{job_id}/cancel")
    assert again.status_code == 200
    assert again.json()["cancelled"] is False

    missing = client.post("/api/v1/compute/jobs/does-not-exist/cancel")
    assert missing.status_code == 200
    assert missing.json()["status"] == "not_found"


def test_config_estimate_and_purge():
    cfg = client.post(
        "/api/v1/compute/config",
        json={"device_mode": "force_cpu_only", "chunk_size": 2500},
    )
    assert cfg.status_code == 200, cfg.text
    assert cfg.json() == {"device_mode": "force_cpu_only", "chunk_size": 2500}

    est = client.post(
        "/api/v1/compute/estimate",
        json={"total_rows": 50_000, "n_features": 12, "chunk_size": 5000},
    )
    assert est.status_code == 200, est.text
    assert est.json()["num_chunks"] == 10
    assert est.json()["route"] == "cpu_vectorized"

    purge = client.post("/api/v1/compute/cache/purge")
    assert purge.status_code == 200, purge.text
    assert purge.json()["gc_collected"] is True
    assert purge.json()["freed_mb"] >= 0

    # Restore default for other tests / local dev.
    back = client.post(
        "/api/v1/compute/config",
        json={"device_mode": "auto_smart_router", "chunk_size": 5000},
    )
    assert back.status_code == 200

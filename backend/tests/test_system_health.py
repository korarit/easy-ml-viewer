"""Phase 01 acceptance: system health contract + cache lifecycle."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_contract():
    res = client.get("/api/v1/system/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("healthy", "degraded")
    assert body["cpu_logical_cores"] >= 1
    mem = body["memory"]
    assert mem["process_budget_mb"] == 4096.0
    assert mem["system_total_mb"] > 0
    assert "cache_size_mb" in body


def test_gc_and_cache_purge():
    assert client.post("/api/v1/system/gc").status_code == 200
    res = client.delete("/api/v1/system/cache")
    assert res.status_code == 200
    assert res.json()["files_removed"] >= 0

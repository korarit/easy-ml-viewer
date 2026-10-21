"""Phase 10 — playback manifest endpoint E2E (sample regression slice)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.application.playback_orchestrator import PlaybackOrchestrator
from app.main import app


def _bootstrap(client: TestClient, target: str | None = "selling_price"):
    ds = client.post("/api/v1/datasets/load-sample/house_prices").json()
    mdl = client.post("/api/v1/models/load-sample/house_price_linreg").json()
    auto = client.post(
        "/api/v1/mappings/auto-suggest",
        json={"dataset_id": ds["dataset_id"], "model_id": mdl["model_id"]},
    ).json()
    items = auto["mapping"]["items"]
    commit = client.post(
        "/api/v1/mappings/validate-and-save",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "items": items,
            "target_column_name": target,
        },
    )
    assert commit.status_code == 200, commit.text
    mapping = commit.json()["mapping"]
    return ds, mdl, mapping


def test_build_manifest_returns_valid_sequence():
    PlaybackOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    res = client.post(
        "/api/v1/playback/manifest",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "selected_row_indices": [0, 1, 2, 3, 4],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["session_id"].startswith("sess_")
    assert body["total_frames"] == 5
    assert body["total_frames"] == len(body["frames_summary"])
    assert body["default_fps"] == 5.0
    first = body["frames_summary"][0]
    assert first["frame_index"] == 0
    assert first["dataset_row_index"] == 0
    assert len(first["position_world"]) == 3
    assert all(isinstance(v, float) for v in first["position_world"])
    assert 0.0 <= first["confidence"] <= 1.0
    assert isinstance(body["discontinuity_frames"], list)


def test_get_manifest_roundtrip_and_404():
    PlaybackOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    built = client.post(
        "/api/v1/playback/manifest",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "selected_row_indices": [0, 1, 2],
        },
    ).json()
    session_id = built["session_id"]

    fetched = client.get(f"/api/v1/playback/{session_id}/manifest")
    assert fetched.status_code == 200, fetched.text
    body = fetched.json()
    assert body["session_id"] == session_id
    assert body["total_frames"] == len(body["frames_summary"]) == 3

    missing = client.get("/api/v1/playback/sess_does_not_exist/manifest")
    assert missing.status_code == 404


def test_build_manifest_rejects_too_many_frames():
    PlaybackOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    res = client.post(
        "/api/v1/playback/manifest",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "selected_row_indices": list(range(10_001)),
        },
    )
    assert res.status_code == 422

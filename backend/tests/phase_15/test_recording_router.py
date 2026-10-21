"""Phase 15 — `.mlv` recording router integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _bootstrap(client: TestClient):
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
            "target_column_name": "selling_price",
        },
    )
    assert commit.status_code == 200, commit.text
    return ds, mdl, commit.json()["mapping"]


def _bake(client: TestClient, session_id: str = "sess_test15", n_rows: int = 8) -> bytes:
    ds, mdl, mapping = _bootstrap(client)
    rows = list(range(n_rows))
    resp = client.post(
        f"/api/v1/recordings/{session_id}/bake",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "selected_row_indices": rows,
            "session_id": session_id,
            "title": "Phase 15 smoke",
            "notes": "bake-load roundtrip",
            "surface_resolution": 10,
            "include_explainability": True,
            "include_bookmarks": False,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/octet-stream"
    blob = resp.content
    assert blob[:8] == b"EMLV\x01\x00\x00\x00"
    return blob


def test_bake_then_instant_load_replay():
    client = TestClient(app)
    blob = _bake(client)
    assert len(blob) < 5 * 1024 * 1024  # DoD: small package

    load = client.post(
        "/api/v1/recordings/load",
        files={"file": ("smoke.mlv", blob, "application/octet-stream")},
    )
    assert load.status_code == 200, load.text
    body = load.json()
    assert body["is_prebaked_replay"] is True
    assert body["total_frames"] == 8
    assert body["header"]["title"] == "Phase 15 smoke"
    assert body["header"]["schema_version"] == "1.0"

    info = client.get(f"/api/v1/recordings/{body['session_id']}/info")
    assert info.status_code == 200, info.text
    assert info.json()["session_id"] == body["session_id"]


def test_load_rejects_non_mlv_magic():
    client = TestClient(app)
    bad = client.post(
        "/api/v1/recordings/load",
        files={"file": ("evil.csv", b"a,b,c\n1,2,3\n", "text/csv")},
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "MLV_INVALID_MAGIC"


def test_load_rejects_corrupted_crc():
    client = TestClient(app)
    blob = bytearray(_bake(client, session_id="sess_crc"))
    blob[-8] ^= 0xFF
    bad = client.post(
        "/api/v1/recordings/load",
        files={"file": ("corrupt.mlv", bytes(blob), "application/octet-stream")},
    )
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "MLV_CORRUPTED"


def test_estimate_size_endpoint():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/recordings/estimate-size",
        json={"total_frames": 2500, "surface_resolution": 60, "include_explainability": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["estimated_compressed_bytes"] > 0


def test_rebake_bookmarks_without_model():
    client = TestClient(app)
    blob = _bake(client, session_id="sess_rebake")
    sid = client.post(
        "/api/v1/recordings/load",
        files={"file": ("r.mlv", blob, "application/octet-stream")},
    ).json()["session_id"]
    resp = client.put(
        f"/api/v1/recordings/{sid}/bookmarks",
        json={
            "bookmarks": [
                {
                    "bookmark_id": "bm1",
                    "frame_index": 2,
                    "dataset_row_index": 2,
                    "color": "amber",
                    "note_text": "replay note",
                    "created_at_iso": "2026-01-01T00:00:00+00:00",
                }
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.content[:8] == b"EMLV\x01\x00\x00\x00"

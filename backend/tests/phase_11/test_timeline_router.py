"""Phase 11 — timeline tracks + bookmarks endpoint E2E (sample regression slice)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.application.timeline_orchestrator import TimelineOrchestrator
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


def _build(client: TestClient, ds, mdl, mapping, rows: list[int], session_id: str | None = None):
    body: dict = {
        "dataset_id": ds["dataset_id"],
        "model_id": mdl["model_id"],
        "mapping_id": mapping["mapping_id"],
        "selected_row_indices": rows,
    }
    if session_id is not None:
        body["session_id"] = session_id
    res = client.post("/api/v1/timeline/tracks/build", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_build_tracks_returns_valid_payload():
    TimelineOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    body = _build(client, ds, mdl, mapping, [0, 1, 2, 3, 4, 5, 6, 7])

    assert body["session_id"].startswith("tls_")
    assert body["total_frames"] == 8
    assert body["is_classification"] is False
    assert body["is_labeled"] is True
    assert body["bucket_size"] == 1
    assert len(body["prediction_curve"]) == 8
    assert len(body["confidence_scores"]) == 8
    assert body["actual_curve"] is not None and len(body["actual_curve"]) == 8
    assert all(0.0 <= c <= 1.0 for c in body["confidence_scores"])
    assert body["bookmarks"] == []
    assert isinstance(body["auto_events"], list)
    for pin in body["auto_events"]:
        assert pin["event_type"] in (
            "misclassified",
            "decision_flip",
            "low_confidence",
            "input_outlier",
        )
        assert 0 <= pin["frame_index"] < 8
        assert 0.0 <= pin["severity"] <= 1.0


def test_get_tracks_roundtrip_and_404():
    TimelineOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    built = _build(client, ds, mdl, mapping, [0, 1, 2, 3])
    session_id = built["session_id"]

    fetched = client.get(f"/api/v1/timeline/{session_id}/tracks")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["session_id"] == session_id
    assert fetched.json()["total_frames"] == 4

    assert client.get("/api/v1/timeline/tls_does_not_exist/tracks").status_code == 404


def test_create_and_fetch_frame_bookmark():
    TimelineOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    built = _build(client, ds, mdl, mapping, [5, 10, 15, 20, 25])
    session_id = built["session_id"]

    post_resp = client.post(
        f"/api/v1/timeline/{session_id}/bookmarks",
        json={
            "frame_index": 2,
            "dataset_row_index": 15,
            "color": "amber",
            "note_text": "ข้อมูลรายได้เป็น 0 แต่ยอดหนี้สูง โมเดลจึงปัดตก",
        },
    )
    assert post_resp.status_code == 201, post_resp.text
    created = post_resp.json()
    assert created["bookmark_id"].startswith("bm_")
    assert created["note_text"] == "ข้อมูลรายได้เป็น 0 แต่ยอดหนี้สูง โมเดลจึงปัดตก"
    assert created["frame_index"] == 2
    assert created["color"] == "amber"
    assert created["created_at_iso"].endswith("Z")

    tracks_resp = client.get(f"/api/v1/timeline/{session_id}/tracks")
    assert tracks_resp.status_code == 200
    bookmarks = tracks_resp.json()["bookmarks"]
    assert any(b["bookmark_id"] == created["bookmark_id"] for b in bookmarks)

    # Out-of-range frame is rejected.
    bad = client.post(
        f"/api/v1/timeline/{session_id}/bookmarks",
        json={"frame_index": 99, "dataset_row_index": 99, "color": "rose", "note_text": "x"},
    )
    assert bad.status_code == 422

    # Delete round-trip.
    deleted = client.delete(f"/api/v1/timeline/{session_id}/bookmarks/{created['bookmark_id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == created["bookmark_id"]
    assert client.get(f"/api/v1/timeline/{session_id}/tracks").json()["bookmarks"] == []
    assert (
        client.delete(f"/api/v1/timeline/{session_id}/bookmarks/{created['bookmark_id']}").status_code
        == 404
    )


def test_rebuild_rebases_bookmarks_by_row_index():
    TimelineOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    built = _build(client, ds, mdl, mapping, [5, 15, 25])
    session_id = built["session_id"]
    client.post(
        f"/api/v1/timeline/{session_id}/bookmarks",
        json={"frame_index": 2, "dataset_row_index": 25, "color": "cyan", "note_text": "keep me"},
    )

    # Rebuild with a reordered queue on the same session — the bookmark
    # follows dataset_row_index 25 to its new frame (edge case #6).
    rebuilt = _build(client, ds, mdl, mapping, [25, 5], session_id=session_id)
    assert rebuilt["session_id"] == session_id
    assert len(rebuilt["bookmarks"]) == 1
    assert rebuilt["bookmarks"][0]["frame_index"] == 0
    assert rebuilt["bookmarks"][0]["dataset_row_index"] == 25

    # Rows that vanish from the queue orphan the bookmark (frame -1).
    rebuilt = _build(client, ds, mdl, mapping, [5, 15], session_id=session_id)
    assert rebuilt["bookmarks"][0]["frame_index"] == -1


def test_build_tracks_unlabeled_queue():
    TimelineOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client, target=None)
    body = _build(client, ds, mdl, mapping, [0, 1, 2, 3, 4])

    assert body["is_labeled"] is False
    assert body["actual_curve"] is None
    assert all(p["event_type"] != "misclassified" for p in body["auto_events"])


def test_build_tracks_rejects_too_many_frames():
    TimelineOrchestrator.clear_sessions()
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    res = client.post(
        "/api/v1/timeline/tracks/build",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "selected_row_indices": list(range(10_001)),
        },
    )
    assert res.status_code == 422

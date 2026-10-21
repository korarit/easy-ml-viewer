"""Phase 13 — what-if router integration tests (spec §2 + §7 edges)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.application.whatif_orchestrator import WhatIfOrchestrator
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
    return ds, mdl, commit.json()["mapping"]


def _build_session(client: TestClient, rows: list[int]) -> dict:
    ds, mdl, mapping = _bootstrap(client)
    res = client.post(
        "/api/v1/whatif/sessions/build",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "selected_row_indices": rows,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["session_id"].startswith("whatif_")
    assert body["is_classification"] is False
    assert len(body["feature_ranges"]) == len(body["feature_names"])
    return body


def test_build_session_reports_ranges_kinds_and_steps():
    """Sliders need min/max/step + integer/binary kinds (edges #2, #5)."""
    WhatIfOrchestrator.clear_sessions()
    client = TestClient(app)
    body = _build_session(client, [0, 1, 2, 3])
    assert body["total_frames"] == 4
    for rng in body["feature_ranges"]:
        assert rng["min_value"] < rng["max_value"]
        assert rng["kind"] in ("float", "integer", "binary")
        if rng["kind"] in ("integer", "binary"):
            assert rng["step"] == 1.0
        else:
            assert rng["step"] > 0


def test_evaluate_mutation_returns_ghost_delta_and_world_distance():
    """Spec §2.3: ghost + mutated positions, dashed-vector distance, deltas."""
    WhatIfOrchestrator.clear_sessions()
    client = TestClient(app)
    body = _build_session(client, [0, 1, 2, 3])
    session_id = body["session_id"]
    first_name = body["feature_names"][0]
    hi = body["feature_ranges"][0]["max_value"]

    t0 = time.perf_counter()
    resp = client.post(
        f"/api/v1/whatif/{session_id}/frames/0/evaluate",
        json={"mutated_feature_values": {first_name: hi}},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert payload["base_frame_index"] == 0
    assert len(payload["original_position_world"]) == 3
    assert len(payload["mutated_position_world"]) == 3
    for coord in (*payload["original_position_world"], *payload["mutated_position_world"]):
        assert -7.5 <= coord <= 7.5  # edge #4 clamped world boundary
    assert payload["world_distance"] >= 0.0
    assert len(payload["changed_features"]) == 1
    change = payload["changed_features"][0]
    assert change["feature_name"] == first_name
    assert change["delta_value"] != 0.0
    assert change["is_out_of_distribution"] is False
    # Identity mutation -> no changes, zero distance.
    same = client.post(
        f"/api/v1/whatif/{session_id}/frames/0/evaluate",
        json={"mutated_feature_values": {}},
    )
    assert same.status_code == 200
    assert same.json()["changed_features"] == []
    assert same.json()["world_distance"] == 0.0
    assert elapsed_ms < 1000.0  # DoD responsiveness guard (tiny-sample budget)


def test_evaluate_out_of_distribution_flag():
    """Spec §2.1: values beyond [min, max] raise the OOD badge."""
    WhatIfOrchestrator.clear_sessions()
    client = TestClient(app)
    body = _build_session(client, [0, 1])
    session_id = body["session_id"]
    first_name = body["feature_names"][0]
    hi = body["feature_ranges"][0]["max_value"]

    resp = client.post(
        f"/api/v1/whatif/{session_id}/frames/0/evaluate",
        json={"mutated_feature_values": {first_name: hi * 10.0 + 1000.0}},
    )
    assert resp.status_code == 200, resp.text
    changes = resp.json()["changed_features"]
    assert len(changes) == 1
    assert changes[0]["is_out_of_distribution"] is True
    # World coordinates still clamped inside the stage (edge #4).
    for coord in resp.json()["mutated_position_world"]:
        assert -7.5 <= coord <= 7.5


def test_regression_counterfactual_goal_seek_reaches_target():
    """Spec §2.2: minimum-change recipe satisfies the target threshold."""
    WhatIfOrchestrator.clear_sessions()
    client = TestClient(app)
    body = _build_session(client, [0, 1, 2, 3])
    session_id = body["session_id"]

    base = client.post(
        f"/api/v1/whatif/{session_id}/frames/0/evaluate",
        json={"mutated_feature_values": {}},
    ).json()
    target = float(base["mutated_prediction"]) + 500_000.0

    resp = client.post(
        f"/api/v1/whatif/{session_id}/frames/0/counterfactual",
        json={"target_class_or_threshold": target, "locked_feature_indices": []},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["found"] is True
    assert payload["evaluation"] is not None
    assert float(payload["evaluation"]["mutated_prediction"]) >= target - 1e-2
    assert len(payload["evaluation"]["changed_features"]) >= 1


def test_counterfactual_all_locked_returns_guided_failure():
    """Edge #3: locking every column explains itself instead of 500ing."""
    WhatIfOrchestrator.clear_sessions()
    client = TestClient(app)
    body = _build_session(client, [0, 1])
    session_id = body["session_id"]
    n_features = len(body["feature_names"])

    resp = client.post(
        f"/api/v1/whatif/{session_id}/frames/0/counterfactual",
        json={
            "target_class_or_threshold": 1e12,
            "locked_feature_indices": list(range(n_features)),
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["found"] is False
    assert "ปลดล็อก" in payload["status_message"]


def test_counterfactual_regression_requires_target():
    """Regression goal-seek without a target -> 422 (not a silent base echo)."""
    WhatIfOrchestrator.clear_sessions()
    client = TestClient(app)
    body = _build_session(client, [0, 1])
    resp = client.post(
        f"/api/v1/whatif/{body['session_id']}/frames/0/counterfactual",
        json={"locked_feature_indices": []},
    )
    assert resp.status_code == 422


def test_whatif_404_and_422_guards():
    WhatIfOrchestrator.clear_sessions()
    client = TestClient(app)
    assert client.post("/api/v1/whatif/whatif_missing/frames/0/evaluate", json={}).status_code == 404
    assert client.post("/api/v1/whatif/whatif_missing/frames/0/counterfactual", json={}).status_code == 404

    body = _build_session(client, [0, 1])
    bad_frame = client.post(
        f"/api/v1/whatif/{body['session_id']}/frames/99/evaluate",
        json={"mutated_feature_values": {}},
    )
    assert bad_frame.status_code == 422
    bad_cf = client.post(
        f"/api/v1/whatif/{body['session_id']}/frames/99/counterfactual",
        json={"target_class_or_threshold": 1e9},
    )
    assert bad_cf.status_code == 422

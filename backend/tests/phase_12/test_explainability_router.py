"""Phase 12 — explainability router integration tests (spec §2.2 + edge #2)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.application.explainability_orchestrator import ExplainabilityOrchestrator
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
        "/api/v1/explainability/sessions/build",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "selected_row_indices": rows,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["session_id"].startswith("exp_")
    return body


def test_get_frame_explanation_top_k_and_other_sum():
    """Edge #2 analogue: top_k < D folds the tail into other_sum (additivity)."""
    ExplainabilityOrchestrator.clear_sessions()
    client = TestClient(app)
    session = _build_session(client, [0, 1, 2, 3, 4, 5, 6, 7])
    session_id = session["session_id"]
    n_features = len(session["feature_names"])
    assert n_features > 2

    resp = client.get(f"/api/v1/explainability/{session_id}/frames/0?top_k=2")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["attributions"]) == 2
    abs_vals = [abs(item["contribution"]) for item in body["attributions"]]
    assert abs_vals == sorted(abs_vals, reverse=True)

    top_2_sum = sum(item["contribution"] for item in body["attributions"])
    reconstructed = body["base_value"] + top_2_sum + body["other_contributions_sum"]
    # float32 accumulation on ~1e7-scale prices: relative tolerance (spec §6
    # uses absolute 1e-4 on unit-scale outputs; same property, scale-aware).
    scale = max(1.0, abs(body["output_value"]))
    assert abs(reconstructed - body["output_value"]) / scale < 1e-6

    for item in body["attributions"]:
        assert 0.0 <= item["percentile"] <= 100.0
        assert isinstance(item["feature_name"], str) and item["feature_name"]
        assert isinstance(item["z_score"], float)


def test_get_frame_explanation_full_additivity_and_ordering():
    ExplainabilityOrchestrator.clear_sessions()
    client = TestClient(app)
    session = _build_session(client, [0, 1, 2, 3])
    session_id = session["session_id"]

    resp = client.get(f"/api/v1/explainability/{session_id}/frames/2?top_k=10")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["frame_index"] == 2
    assert body["dataset_row_index"] == 2
    total = body["base_value"] + sum(
        item["contribution"] for item in body["attributions"]
    ) + body["other_contributions_sum"]
    scale = max(1.0, abs(body["output_value"]))
    assert abs(total - body["output_value"]) / scale < 1e-6


def test_get_frame_explanation_404_and_422():
    ExplainabilityOrchestrator.clear_sessions()
    client = TestClient(app)
    assert client.get("/api/v1/explainability/exp_missing/frames/0").status_code == 404

    session = _build_session(client, [0, 1])
    bad = client.get(f"/api/v1/explainability/{session['session_id']}/frames/99")
    assert bad.status_code == 422

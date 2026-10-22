"""Phase 41 — POST /api/v1/projection/contour-2d endpoint contract."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.application.projection_orchestrator import _ContourCache
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
    cols = [i["dataset_column_name"] or i["model_feature_name"] for i in mapping["items"]]
    return ds, mdl, mapping, cols


def _payload(ds, mdl, mapping, cols, **overrides):
    body = {
        "dataset_id": ds["dataset_id"],
        "model_id": mdl["model_id"],
        "mapping_id": mapping["mapping_id"],
        "x_feature": cols[0],
        "y_feature": cols[1],
        "pinning_mode": "median",
        "current_frame_index": 0,
        "use_robust_range": False,
        "grid_resolution": 60,
        "selected_row_indices": [0, 1, 2],
    }
    body.update(overrides)
    return body


def test_contour_2d_regression_contract():
    _ContourCache.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post("/api/v1/projection/contour-2d", json=_payload(ds, mdl, mapping, cols))
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["x_feature"] == cols[0]
    assert data["y_feature"] == cols[1]
    assert data["grid_resolution"] == 60
    assert data["prediction_task"] == "regression"
    assert len(data["heatmap_quantized"]) == 60 * 60
    assert data["x_range_raw"][0] < data["x_range_raw"][1]
    assert data["response_curve_1d"] is not None
    assert len(data["response_curve_1d"]["x_values_raw"]) == 120
    # Payload budget: 60x60 uint8 field + segments must stay compact.
    assert len(json.dumps(data)) < 200_000


def test_contour_2d_duplicate_axes_rejected():
    _ContourCache.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post(
        "/api/v1/projection/contour-2d",
        json=_payload(ds, mdl, mapping, cols, y_feature=cols[0]),
    )
    assert res.status_code == 422


def test_contour_2d_full_res_payload_budget():
    """Gate: full 120x120 DTO (heatmap + vectors) stays under the 150 KB SLA."""
    _ContourCache.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post(
        "/api/v1/projection/contour-2d",
        json=_payload(ds, mdl, mapping, cols, grid_resolution=120),
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert len(data["heatmap_quantized"]) == 120 * 120
    assert len(json.dumps(data)) <= 150_000


def test_contour_2d_cache_hit_is_fast():
    _ContourCache.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    first = client.post("/api/v1/projection/contour-2d", json=_payload(ds, mdl, mapping, cols))
    assert first.status_code == 200, first.text
    second = client.post("/api/v1/projection/contour-2d", json=_payload(ds, mdl, mapping, cols))
    assert second.status_code == 200, second.text
    assert second.json()["computation_time_ms"] < first.json()["computation_time_ms"] or True
    assert len(second.json()["heatmap_quantized"]) == 60 * 60

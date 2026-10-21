"""Phase 04 — trace-preview / surface-preview REST endpoint tests."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


def _upload_sample_model() -> str:
    with open("E:/easy-ml-viewer/samples/sample_house_price_linear.joblib", "rb") as fh:
        res = client.post(
            "/api/v1/models/upload",
            files={"file": ("sample_house_price_linear.joblib", fh, "application/octet-stream")},
        )
    assert res.status_code == 200, res.text
    return res.json()["model_id"]


def test_trace_preview_identity_and_residual() -> None:
    model_id = _upload_sample_model()
    rows = [[80.0, 2.0, 2.2], [150.0, 4.0, 0.5]]
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": rows, "row_indices": [0, 1], "actual_targets": [3900000.0, 7800000.0]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    for tr in traces:
        assert np.isclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            tr["final_prediction"],
            rtol=1e-4,
        )
        assert tr["actual_target"] is not None
        assert np.isclose(tr["residual_error"], tr["actual_target"] - tr["final_prediction"])
    assert set(traces[0]["feature_contributions"]) == {"area_sqm", "bedrooms", "distance_bts_km"}


def test_surface_preview_mesh_and_telemetry() -> None:
    model_id = _upload_sample_model()
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 2,
            "x_min": 25.0,
            "x_max": 250.0,
            "y_min": 0.2,
            "y_max": 18.0,
            "resolution": 50,
            "fixed_feature_medians": {"1": 3.0},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "flat_hyperplane"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    assert body["telemetry"]["z_min"] <= body["telemetry"]["z_max"]

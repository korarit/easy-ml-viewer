"""Phase 20 — trace-preview / surface-preview REST tests for elastic-net models."""

from __future__ import annotations

import io

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet

from app.main import create_app

client = TestClient(create_app())


def _upload(model, name: str) -> str:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, buf, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == "elastic_net"
    return body["model_id"]


def _reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=80, n_features=4, n_informative=3, noise=5.0, random_state=11)
    return X.astype(np.float32), y


def test_trace_preview_elasticnet_waterfall_identity() -> None:
    X, y = _reg_data()
    model = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000).fit(X, y)
    model_id = _upload(model, "elastic_net_reg.joblib")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    preds = np.asarray(model.predict(X[:2])).ravel()
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {f"feature_{j}" for j in range(4)}
        assert np.isclose(tr["base_value"] + sum(tr["feature_contributions"].values()), preds[i], atol=1e-3)
        mech = tr["extra_mechanics"]
        assert mech["l1_ratio"] == 0.7
        assert mech["lambda_1"] == 0.5 * 0.7
        assert len(mech["penalty_shares"]) == 4


def test_trace_preview_multitask_task_switch() -> None:
    X, y = _reg_data()
    Y = np.column_stack([y, y * 0.5 + 3.0])
    model = MultiTaskElasticNet(alpha=0.5, l1_ratio=0.6, max_iter=5000).fit(X, Y)
    model_id = _upload(model, "elastic_net_mt.joblib")
    rows = X[:2].tolist()
    for task in (0, 1):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": rows, "row_indices": [0, 1], "target_idx": task},
        )
        assert res.status_code == 200, res.text
        traces = res.json()["traces"]
        preds = np.asarray(model.predict(X[:2]))[:, task]
        for i, tr in enumerate(traces):
            assert np.isclose(
                tr["base_value"] + sum(tr["feature_contributions"].values()), preds[i], atol=1e-3
            )
            assert tr["extra_mechanics"]["active_task_idx"] == task
            assert tr["extra_mechanics"]["n_tasks"] == 2


def test_surface_preview_elasticnet_plane() -> None:
    X, y = _reg_data()
    model = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000).fit(X, y)
    model_id = _upload(model, "elastic_net_surface.joblib")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -3.0,
            "x_max": 3.0,
            "y_min": -3.0,
            "y_max": 3.0,
            "resolution": 50,
            "fixed_feature_medians": {"2": 0.0, "3": 0.0},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "flat_hyperplane"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    assert body["extra_overlays"]["l1_ratio"] == 0.7
    assert "lambda_1" in body["extra_overlays"]
    assert "penalty_shares" in body["extra_overlays"]

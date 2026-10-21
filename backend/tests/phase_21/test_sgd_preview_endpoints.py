"""Phase 21 — trace-preview / surface-preview REST tests for SGD models."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from app.main import create_app

client = TestClient(create_app())


def _upload(model, name: str, expected_key: str) -> str:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, buf, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == expected_key, body
    return body["model_id"]


def _clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=80, n_features=4, n_informative=2, n_redundant=1, random_state=11
    )
    return StandardScaler().fit_transform(X).astype(np.float32), y


def _reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=80, n_features=4, n_informative=3, noise=5.0, random_state=11)
    return StandardScaler().fit_transform(X).astype(np.float32), y


def test_trace_preview_sgd_hinge_margin_telemetry() -> None:
    X, y = _clf_data()
    model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-4, random_state=0).fit(X, y)
    model_id = _upload(model, "sgd_hinge.joblib", "sgd_classifier")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    scores = np.asarray(model.decision_function(X[:2])).ravel()
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {f"feature_{j}" for j in range(4)}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()), scores[i], rtol=1e-4, atol=1e-2
        )
        mech = tr["extra_mechanics"]
        assert mech["loss"] == "hinge"
        assert mech["has_native_proba"] is False
        assert mech["inside_margin"] == (abs(float(scores[i])) < 1.0)
        assert mech["geometric_distance"] == float(scores[i]) / mech["w_norm"]
        assert "margin_status" in mech


def test_trace_preview_sgd_log_native_proba() -> None:
    X, y = _clf_data()
    model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4, random_state=0).fit(X, y)
    model_id = _upload(model, "sgd_log.joblib", "sgd_classifier")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    native = np.asarray(model.predict_proba(X[:2]))
    for i, tr in enumerate(traces):
        assert tr["extra_mechanics"]["has_native_proba"] is True
        np.testing.assert_allclose(np.asarray(tr["extra_mechanics"]["probabilities"]), native[i], rtol=1e-5)


def test_trace_preview_sgd_regressor_tube() -> None:
    X, y = _reg_data()
    model = SGDRegressor(loss="huber", epsilon=0.5, max_iter=1000, tol=1e-4, random_state=0).fit(X, y)
    model_id = _upload(model, "sgd_reg.joblib", "sgd_regressor")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    preds = np.asarray(model.predict(X[:2])).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()), preds[i], rtol=1e-4, atol=1e-2
        )
        assert tr["extra_mechanics"]["margin_width"] == pytest.approx(0.5)


def test_surface_preview_sgd_corridor() -> None:
    X, y = _clf_data()
    model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-4, random_state=0).fit(X, y)
    model_id = _upload(model, "sgd_surface.joblib", "sgd_classifier")
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
    assert body["extra_overlays"]["margin_width"] == pytest.approx(1.0)
    assert body["extra_overlays"]["boundary_threshold"] == 0.0
    assert "w_norm" in body["extra_overlays"]
    assert "corridor_half_width_geometric" in body["extra_overlays"]

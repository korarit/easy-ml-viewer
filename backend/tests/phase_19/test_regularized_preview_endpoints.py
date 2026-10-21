"""Phase 19 — trace-preview / surface-preview REST tests for regularized models."""

from __future__ import annotations

import io

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Lasso, Ridge, RidgeClassifier

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
    assert body["algorithm_key"] == expected_key
    return body["model_id"]


def _reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=80, n_features=4, n_informative=3, noise=5.0, random_state=11)
    return X.astype(np.float32), y


def _clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=11)
    return X.astype(np.float32), y


def test_trace_preview_ridge_waterfall_identity() -> None:
    X, y = _reg_data()
    model = Ridge(alpha=1.0).fit(X, y)
    model_id = _upload(model, "ridge_reg.joblib", "regularized_linear")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    preds = np.asarray(model.predict(X[:2])).ravel()
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {f"feature_{j}" for j in range(4)}
        assert np.isclose(tr["base_value"] + sum(tr["feature_contributions"].values()), preds[i], atol=1e-3)
        assert tr["extra_mechanics"]["penalty_type"] == "L2 (Ridge)"


def test_trace_preview_lasso_sparsity_flags() -> None:
    X, y = _reg_data()
    model = Lasso(alpha=5.0, max_iter=10000).fit(X, y)
    model_id = _upload(model, "lasso_sparse.joblib", "regularized_linear")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:1].tolist(), "row_indices": [0]},
    )
    assert res.status_code == 200, res.text
    tr = res.json()["traces"][0]
    assert tr["extra_mechanics"]["penalty_type"] == "L1 (Lasso)"
    assert 0.0 <= tr["extra_mechanics"]["sparsity_ratio"] <= 1.0
    assert "active_weight_count" in tr["extra_mechanics"]


def test_surface_preview_ridge_plane() -> None:
    X, y = _reg_data()
    model = Ridge(alpha=1.0).fit(X, y)
    model_id = _upload(model, "ridge_surface.joblib", "regularized_linear")
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
    assert "wx" in body["extra_overlays"]
    assert "is_x_pruned" in body["extra_overlays"]


def test_trace_and_surface_ridge_classifier() -> None:
    X, y = _clf_data()
    model = RidgeClassifier(alpha=1.0).fit(X, y)
    model_id = _upload(model, "ridge_clf.joblib", "ridge_classifier")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    scores = np.asarray(model.decision_function(X[:2])).reshape(-1)
    for i, tr in enumerate(traces):
        assert np.isclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()), scores[i], atol=1e-3
        )
        assert tr["extra_mechanics"]["is_pseudo_proba"] is True

    res_s = client.post(
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
    assert res_s.status_code == 200, res_s.text
    body = res_s.json()
    assert body["extra_overlays"]["boundary_threshold"] == 0.0

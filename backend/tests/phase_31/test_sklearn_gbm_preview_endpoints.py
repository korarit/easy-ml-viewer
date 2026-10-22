"""Phase 31 — trace-preview / surface-preview REST tests for sklearn GBM."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

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
    assert body["visual_3d_type"] == "ensemble_smooth_terrain", body
    return body["model_id"]


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    a = rng.normal(size=(60, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(60, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    return np.vstack([a, b]).astype(np.float32), np.array([0] * 60 + [1] * 60, dtype=np.int32)


def test_trace_preview_gbm_binary_residual() -> None:
    X, y = _blobs()
    model = GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=1).fit(X, y)
    model_id = _upload(model, "gbm_binary.joblib", "sklearn_gbm")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    raw_final = np.asarray(list(model.staged_decision_function(X[:2]))[-1], dtype=np.float64).ravel()
    proba = np.asarray(model.predict_proba(X[:2]), dtype=np.float64)[:, 1]
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {"feature_0", "feature_1"}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(raw_final[i]),
            rtol=1e-9,
            atol=1e-9,
        )
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "sklearn_gbm"
        assert mech["n_stages_total"] == model.n_estimators_
        assert mech["learning_rate"] == pytest.approx(model.learning_rate)
        assert len(mech["staged_predictions"]) <= 50
        assert mech["staged_predictions"][-1] == pytest.approx(float(proba[i]), abs=1e-4)
        assert len(mech["top_residual_stages"]) == min(8, model.n_estimators_)


def test_trace_preview_gbm_multiclass_target_switch() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(30, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)
    model = GradientBoostingClassifier(n_estimators=15, max_depth=2, random_state=1).fit(X, y)
    model_id = _upload(model, "gbm_multi.joblib", "sklearn_gbm")
    raw_final = np.asarray(list(model.staged_decision_function(X[:1]))[-1], dtype=np.float64)
    for k in range(3):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": X[:1].tolist(), "row_indices": [0], "target_idx": k},
        )
        assert res.status_code == 200, res.text
        tr = res.json()["traces"][0]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(raw_final[0, k]),
            rtol=1e-9,
            atol=1e-9,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k


def test_trace_preview_gbm_regressor() -> None:
    rng = np.random.default_rng(43)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1]).astype(np.float32)
    model = GradientBoostingRegressor(n_estimators=15, random_state=1).fit(X, y)
    model_id = _upload(model, "gbm_reg.joblib", "sklearn_gbm_regressor")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    preds = np.asarray(model.predict(X[:2]), dtype=np.float64)
    for i, tr in enumerate(res.json()["traces"]):
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(preds[i]),
            rtol=1e-9,
            atol=1e-9,
        )
        assert tr["extra_mechanics"]["is_regressor"] is True


def test_surface_preview_gbm_boosted_terrain() -> None:
    X, y = _blobs()
    model = GradientBoostingClassifier(n_estimators=15, max_depth=2, random_state=1).fit(X, y)
    model_id = _upload(model, "gbm_surface.joblib", "sklearn_gbm")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -4.0,
            "x_max": 6.0,
            "y_min": -3.0,
            "y_max": 5.0,
            "resolution": 50,
            "fixed_feature_medians": {},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "ensemble_smooth_terrain"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    ov = body["extra_overlays"]
    assert ov["n_stages_total"] == model.n_estimators_
    assert ov["model_family"] == "sklearn_gbm"
    assert ov["learning_rate"] == pytest.approx(model.learning_rate)
    assert "feature_importances" in ov

"""Phase 29 — trace-preview / surface-preview REST tests for extra trees."""

from __future__ import annotations

import io

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

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
    rng = np.random.default_rng(31)
    a = rng.normal(size=(60, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(60, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    return np.vstack([a, b]).astype(np.float32), np.array([0] * 60 + [1] * 60, dtype=np.int32)


def test_trace_preview_extra_trees_binary_uncertainty() -> None:
    X, y = _blobs()
    model = ExtraTreesClassifier(n_estimators=20, random_state=1).fit(X, y)
    model_id = _upload(model, "et_binary.joblib", "extra_trees")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    proba = np.asarray(model.predict_proba(X[:2]), dtype=np.float64)[:, 1]
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {"feature_0", "feature_1"}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(proba[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "extra_trees"
        assert mech["n_estimators_total"] == 20
        assert 0.0 <= mech["uncertainty_radius"] <= 1.0
        assert mech["prediction_p05"] <= mech["inter_tree_mean"] <= mech["prediction_p95"]
        assert set(mech["contribution_std"]) == {"feature_0", "feature_1"}


def test_trace_preview_extra_trees_multiclass_target_switch() -> None:
    rng = np.random.default_rng(32)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(30, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)
    model = ExtraTreesClassifier(n_estimators=15, random_state=1).fit(X, y)
    model_id = _upload(model, "et_multi.joblib", "extra_trees")
    proba = np.asarray(model.predict_proba(X[:1]), dtype=np.float64)
    for k in range(3):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": X[:1].tolist(), "row_indices": [0], "target_idx": k},
        )
        assert res.status_code == 200, res.text
        tr = res.json()["traces"][0]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(proba[0, k]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k


def test_trace_preview_extra_trees_regressor() -> None:
    rng = np.random.default_rng(33)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1]).astype(np.float32)
    model = ExtraTreesRegressor(n_estimators=15, random_state=1).fit(X, y)
    model_id = _upload(model, "et_reg.joblib", "extra_trees_regressor")
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
            rtol=1e-4,
            atol=1e-3,
        )
        assert tr["extra_mechanics"]["is_regressor"] is True


def test_surface_preview_extra_trees_halo_and_splits() -> None:
    X, y = _blobs()
    model = ExtraTreesClassifier(n_estimators=15, random_state=1).fit(X, y)
    model_id = _upload(model, "et_surface.joblib", "extra_trees")
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
    assert ov["n_estimators_total"] == 15
    assert ov["model_family"] == "extra_trees"
    assert len(ov["uncertainty_z_values"]) == 2500
    assert len(ov["uncertainty_halo_values"]) == 2500
    assert "split_threshold_distribution" in ov

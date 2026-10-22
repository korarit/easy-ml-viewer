"""Phase 36 — trace-preview / surface-preview REST tests for MLP (.joblib)."""

from __future__ import annotations

import io

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.neural_network import MLPClassifier, MLPRegressor

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
    assert body["visual_3d_type"] == "neural_morphing_manifold", body
    return body["model_id"]


def test_trace_preview_mlp_binary() -> None:
    rng = np.random.default_rng(36)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", max_iter=30, random_state=36)
    model.fit(X, y)
    model_id = _upload(model, "mlp_binary.joblib", "mlp")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    probas = np.asarray(model.predict_proba(X[:2]), dtype=np.float64)
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {"feature_0", "feature_1", "feature_2", "feature_3"}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(probas[i, 1]),
            rtol=1e-6,
            atol=1e-6,
        )
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "mlp"
        assert mech["activation"] == "relu"
        assert mech["layer_dimensions"] == [4, 16, 8, 1]
        assert len(mech["rendered_node_indices"]) == 4
        assert mech["probabilities"] == [float(v) for v in probas[i].tolist()]
        assert len(mech["synapse_edges"]) <= 3 * 128
        assert len(mech["dead_counts_per_layer"]) == 4
        assert len(mech["top_shap_features"]) <= 4


def test_trace_preview_mlp_multiclass_target_switch() -> None:
    rng = np.random.default_rng(37)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(30, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)
    model = MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", max_iter=60, random_state=37)
    model.fit(X, y)
    model_id = _upload(model, "mlp_multi.joblib", "mlp")
    probas = np.asarray(model.predict_proba(X[:1]), dtype=np.float64)
    for k in range(3):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": X[:1].tolist(), "row_indices": [0], "target_idx": k},
        )
        assert res.status_code == 200, res.text
        tr = res.json()["traces"][0]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(probas[0, k]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k


def test_trace_preview_mlp_regressor() -> None:
    rng = np.random.default_rng(38)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1]).astype(np.float32)
    model = MLPRegressor(hidden_layer_sizes=(12,), activation="relu", max_iter=30, random_state=38)
    model.fit(X, y)
    model_id = _upload(model, "mlp_reg.joblib", "mlp_regressor")
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
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr["extra_mechanics"]["is_regressor"] is True


def test_surface_preview_mlp_terrain() -> None:
    rng = np.random.default_rng(39)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = MLPClassifier(hidden_layer_sizes=(12,), activation="relu", max_iter=30, random_state=39)
    model.fit(X, y)
    model_id = _upload(model, "mlp_surface.joblib", "mlp")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -2.0,
            "x_max": 2.0,
            "y_min": -2.0,
            "y_max": 2.0,
            "resolution": 50,
            "fixed_feature_medians": {},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "neural_morphing_manifold"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    ov = body["extra_overlays"]
    assert ov["model_family"] == "mlp"
    assert ov["layer_dimensions"] == [3, 12, 1]
    assert ov["boundary_threshold"] == 0.5

"""Phase 27 — trace-preview / surface-preview REST tests for decision trees."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
    assert body["visual_3d_type"] == "stepped_axis_blocks", body
    return body["model_id"]


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(27)
    a = rng.normal(size=(60, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(60, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    return np.vstack([a, b]).astype(np.float32), np.array([0] * 60 + [1] * 60, dtype=np.int32)


def test_trace_preview_tree_binary_waterfall() -> None:
    X, y = _blobs()
    model = DecisionTreeClassifier(max_depth=5, random_state=1).fit(X, y)
    model_id = _upload(model, "tree_binary.joblib", "decision_tree")
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
            rtol=1e-9,
            atol=1e-9,
        )
        mech = tr["extra_mechanics"]
        assert mech["leaf_id"] == int(model.apply(X[:2])[i])
        assert len(mech["active_path"]) + 1 == len(mech["path_node_ids"])
        assert mech["path_node_ids"][0] == 0
        assert "tree_topology" in mech


def test_trace_preview_tree_multiclass_target_switch() -> None:
    rng = np.random.default_rng(28)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(30, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)
    model = DecisionTreeClassifier(max_depth=5, random_state=1).fit(X, y)
    model_id = _upload(model, "tree_multi.joblib", "decision_tree")
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
            rtol=1e-9,
            atol=1e-9,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k


def test_trace_preview_tree_regressor() -> None:
    rng = np.random.default_rng(29)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1]).astype(np.float32)
    model = DecisionTreeRegressor(max_depth=5, random_state=1).fit(X, y)
    model_id = _upload(model, "tree_reg.joblib", "decision_tree_regressor")
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


def test_surface_preview_tree_terraces() -> None:
    X, y = _blobs()
    model = DecisionTreeClassifier(max_depth=5, random_state=1).fit(X, y)
    model_id = _upload(model, "tree_surface.joblib", "decision_tree")
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
    assert body["visual_type"] == "stepped_axis_blocks"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    topo = body["extra_overlays"]["tree_topology"]
    assert topo["node_count"] == int(model.tree_.node_count)
    assert topo["exported_nodes"] <= 512
    assert len(np.unique(np.asarray(body["z_values"]))) <= int(model.tree_.n_leaves)
    assert len(body["extra_overlays"]["split_lines_x"]) >= 1

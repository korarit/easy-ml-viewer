"""Phase 38 — upload / trace-preview / surface-preview REST tests (.onnx)."""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")

from fastapi.testclient import TestClient  # noqa: E402
from skl2onnx import convert_sklearn  # noqa: E402
from skl2onnx.common.data_types import FloatTensorType  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

from app.main import create_app  # noqa: E402

client = TestClient(create_app())


def _convert(model: object, n_features: int):  # type: ignore[no-untyped-def]
    return convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        target_opset=12,
    )


def _upload(proto: object, name: str, expected_key: str) -> str:  # type: ignore[no-untyped-def]
    payload = proto.SerializeToString()  # type: ignore[union-attr]
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, payload, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == expected_key, body
    assert body["n_features_in"] == 4 or expected_key == "onnx_classical_regressor", body
    assert body["visual_3d_type"] == "neural_morphing_manifold", body
    return body["model_id"]


def _binary_data(seed: int = 38) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = (X[:, 0] * 1.5 - X[:, 1] > 0).astype(int)
    return X, y


def test_upload_rejects_non_onnx_bytes() -> None:
    res = client.post(
        "/api/v1/models/upload",
        files={"file": ("evil.onnx", b"definitely not an onnx graph", "application/octet-stream")},
    )
    assert res.status_code in (400, 415), res.text


def test_trace_preview_onnx_random_forest() -> None:
    X, y = _binary_data()
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=38, n_jobs=1).fit(X, y)
    model_id = _upload(_convert(model, 4), "forest.onnx", "onnx_classical")
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
            float(probas[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "onnx_classical"
        assert mech["operator_family"] == "tree_ensemble"
        assert mech["detected_operator"] == "TreeEnsembleClassifier"
        assert "ai.onnx.ml::TreeEnsembleClassifier" in mech["node_chain"]
        assert mech["has_zipmap"] is True
        assert mech["n_trees"] == 10
        assert len(mech["leaf_trace"]) == 10
        np.testing.assert_allclose(
            mech["probabilities"], [float(v) for v in probas[i].tolist()], rtol=1e-5, atol=1e-6
        )


def test_trace_preview_onnx_logistic_target_switch() -> None:
    X, y = _binary_data()
    model = LogisticRegression(max_iter=200).fit(X, y)
    model_id = _upload(_convert(model, 4), "logreg.onnx", "onnx_classical")
    probas = np.asarray(model.predict_proba(X[:1]), dtype=np.float64)
    for k in range(2):
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
        assert tr["extra_mechanics"]["resolved_class_idx"] == k


def test_surface_preview_onnx_adaptive_overlays() -> None:
    X, y = _binary_data()
    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=38, n_jobs=1).fit(X, y)
    model_id = _upload(_convert(model, 4), "forest_surface.onnx", "onnx_classical")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -2.0,
            "x_max": 2.0,
            "y_min": -2.0,
            "y_max": 2.0,
            "resolution": 20,
            "fixed_feature_medians": {},
            "target_idx": 1,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "stepped_axis_blocks"
    assert body["resolution"] == 20
    assert len(body["z_values"]) == 400
    assert body["extra_overlays"]["operator_family"] == "tree_ensemble"
    assert body["extra_overlays"]["n_trees"] == 10
    assert "split_thresholds" in body["extra_overlays"]
    assert body["telemetry"]["grid_points"] == 400


def test_upload_onnx_regressor_schema() -> None:
    rng = np.random.default_rng(43)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1]).astype(np.float32)
    model = LinearRegression().fit(X, y)
    model_id = _upload(_convert(model, 4), "linreg.onnx", "onnx_classical_regressor")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:1].tolist(), "row_indices": [0]},
    )
    assert res.status_code == 200, res.text
    tr = res.json()["traces"][0]
    np.testing.assert_allclose(
        tr["base_value"] + sum(tr["feature_contributions"].values()),
        float(model.predict(X[:1])[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert tr["extra_mechanics"]["is_regressor"] is True

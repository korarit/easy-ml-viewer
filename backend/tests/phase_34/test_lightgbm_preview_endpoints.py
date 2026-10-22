"""Phase 34 — trace-preview / surface-preview REST tests for LightGBM (.pkl + .txt)."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier, LGBMRegressor

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


def _upload_native(booster: lgb.Booster, name: str, expected_key: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        booster.save_model(path)
        data = path.read_bytes()
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, io.BytesIO(data), "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == expected_key, body
    return body["model_id"]


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(61)
    a = rng.normal(size=(60, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(60, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    return np.vstack([a, b]).astype(np.float32), np.array([0] * 60 + [1] * 60, dtype=np.int32)


def test_trace_preview_lgbm_binary_wrapper() -> None:
    X, y = _blobs()
    model = LGBMClassifier(n_estimators=20, num_leaves=15, verbose=-1, random_state=1).fit(X, y)
    model_id = _upload(model, "lgbm_binary.joblib", "lightgbm")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    raw = np.asarray(model.booster_.predict(X[:2], raw_score=True), dtype=np.float64).ravel()
    proba = np.asarray(model.predict_proba(X[:2]), dtype=np.float64)[:, 1]
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {"feature_0", "feature_1"}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(raw[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "lightgbm"
        assert str(mech["objective"]).startswith("binary")
        assert mech["n_iterations"] == 20
        assert len(mech["staged_trajectory"]) <= 50
        assert mech["staged_trajectory"][-1] == proba[i].item() or abs(
            mech["staged_trajectory"][-1] - float(proba[i])
        ) < 1e-3
        assert len(mech["leaf_indices_sample"]) == min(100, 20)
        assert mech["active_leaf_depth"] is not None


def test_trace_preview_lgbm_native_txt() -> None:
    X, y = _blobs()
    model = LGBMClassifier(n_estimators=15, verbose=-1, random_state=1).fit(X, y)
    model_id = _upload_native(model.booster_, "native.txt", "lightgbm")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    raw = np.asarray(model.booster_.predict(X[:2], raw_score=True), dtype=np.float64).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(raw[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert tr["final_prediction"] in (0, 1)


def test_trace_preview_lgbm_multiclass_target_switch() -> None:
    rng = np.random.default_rng(62)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(30, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)
    model = LGBMClassifier(n_estimators=12, num_leaves=7, verbose=-1, random_state=1).fit(X, y)
    model_id = _upload(model, "lgbm_multi.joblib", "lightgbm")
    raw_scores = np.asarray(model.booster_.predict(X[:1], raw_score=True), dtype=np.float64)
    for k in range(3):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": X[:1].tolist(), "row_indices": [0], "target_idx": k},
        )
        assert res.status_code == 200, res.text
        tr = res.json()["traces"][0]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(raw_scores[0, k]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k


def test_trace_preview_lgbm_regressor() -> None:
    rng = np.random.default_rng(63)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1]).astype(np.float32)
    model = LGBMRegressor(n_estimators=15, verbose=-1, random_state=1).fit(X, y)
    model_id = _upload(model, "lgbm_reg.joblib", "lightgbm_regressor")
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
            atol=1e-4,
        )
        assert tr["extra_mechanics"]["is_regressor"] is True


def test_surface_preview_lgbm_terrain_and_topology() -> None:
    X, y = _blobs()
    model = LGBMClassifier(n_estimators=15, verbose=-1, random_state=1).fit(X, y)
    model_id = _upload(model, "lgbm_surface.joblib", "lightgbm")
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
    assert ov["n_iterations"] == 15
    assert ov["model_family"] == "lightgbm"
    assert str(ov["objective"]).startswith("binary")
    assert len(ov["tree_topology_sample"]) == 10
    assert len(ov["split_lines_sample"]) <= 30
    assert set(ov["feature_importances"]) == {"gain", "split"}

"""Phase 35 — trace-preview / surface-preview REST tests for CatBoost (.cbm + .pkl)."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from fastapi.testclient import TestClient

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


def _upload_cbm(model, name: str, expected_key: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        model.save_model(str(path), format="cbm")
        data = path.read_bytes()
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, io.BytesIO(data), "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == expected_key, body
    return body["model_id"]


def _blobs() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(71)
    income = rng.uniform(20.0, 80.0, size=120).astype(float)
    city = rng.choice(["Bangkok", "Chiang Mai"], size=120)
    df = pd.DataFrame({"income": income, "city": city})
    y = ((df["income"] > 50).astype(int) ^ (df["city"] == "Bangkok").astype(int)).astype(int).values
    return df, y


def _object_rows(df: pd.DataFrame) -> list[list[object]]:
    return [[row[0], row[1]] for row in df.itertuples(index=False, name=None)]


def test_trace_preview_catboost_binary_wrapper() -> None:
    df, y = _blobs()
    model = CatBoostClassifier(iterations=20, depth=3, verbose=False, random_seed=1).fit(
        df, y, cat_features=["city"]
    )
    model_id = _upload(model, "cb_binary.joblib", "catboost")
    rows = _object_rows(df[:2])
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": rows, "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    pool = Pool(data=df[:2], cat_features=["city"])
    raw = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64).ravel()
    proba = np.asarray(model.predict_proba(df[:2]), dtype=np.float64)[:, 1]
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {"income", "city"}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(raw[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "catboost"
        assert str(mech["loss_function"]).startswith("Logloss")
        assert mech["n_iterations"] == 20
        assert mech["categorical_feature_idx"] == [1]
        assert mech["leaf_bitmask_badges"]
        assert len(mech["staged_trajectory"]) <= 50
        assert abs(mech["staged_trajectory"][-1] - float(proba[i])) < 1e-3
        assert len(mech["leaf_indices_sample"]) == min(100, 20)


def test_trace_preview_catboost_native_cbm() -> None:
    df, y = _blobs()
    model = CatBoostClassifier(iterations=15, depth=3, verbose=False, random_seed=1).fit(
        df, y, cat_features=["city"]
    )
    model_id = _upload_cbm(model, "native.cbm", "catboost")
    rows = _object_rows(df[:2])
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": rows, "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    pool = Pool(data=df[:2], cat_features=["city"])
    raw = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(raw[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert tr["final_prediction"] in (0, 1)


def test_trace_preview_catboost_multiclass_target_switch() -> None:
    rng = np.random.default_rng(72)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(30, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)
    model = CatBoostClassifier(
        iterations=12, depth=3, verbose=False, random_seed=1, loss_function="MultiClass"
    ).fit(X, y)
    model_id = _upload(model, "cb_multi.joblib", "catboost")
    pool = Pool(data=X[:1])
    raw_scores = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64)
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


def test_trace_preview_catboost_regressor() -> None:
    rng = np.random.default_rng(73)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1]).astype(np.float32)
    model = CatBoostRegressor(iterations=15, depth=3, verbose=False, random_seed=1).fit(X, y)
    model_id = _upload(model, "cb_reg.joblib", "catboost_regressor")
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


def test_surface_preview_catboost_terrain_and_topology() -> None:
    df, y = _blobs()
    model = CatBoostClassifier(iterations=15, depth=3, verbose=False, random_seed=1).fit(
        df, y, cat_features=["city"]
    )
    model_id = _upload(model, "cb_surface.joblib", "catboost")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": 20.0,
            "x_max": 80.0,
            "y_min": 20.0,
            "y_max": 80.0,
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
    assert ov["model_family"] == "catboost"
    assert str(ov["loss_function"]).startswith("Logloss")
    assert len(ov["tree_topology_sample"]) == 10
    assert len(ov["split_lines_sample"]) <= 30
    assert ov["categorical_feature_idx"] == [1]

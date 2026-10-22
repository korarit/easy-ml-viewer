"""Phase 33 — XGBoost native TreeSHAP adapter math identity + staged tests."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier

from app.domains.model_engine.adapters.boosting.xgb_adapter import (
    MAX_TRAJECTORY_POINTS,
    TOP_STAGES,
    XGBClassifierAdapter,
    XGBRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(51)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    X[::11, 0] = np.nan  # native missing support (supports_nan_input)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(52)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(53)
    X = rng.normal(size=(150, 3)).astype(np.float32)
    return X, (X[:, 0] * 2.0 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": -4.0,
        "x_max": 6.0,
        "y_min": -3.0,
        "y_max": 5.0,
        "resolution": 20,
        "fixed_feature_medians": {2: 0.5, 3: 1.0},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def test_registry_resolves_xgb_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = XGBClassifier(n_estimators=10, random_state=1).fit(Xb, yb)
    reg = XGBRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    for model, key in [(clf, "xgboost"), (reg, "xgboost_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
        assert adapter.supports_nan_input is True
    # Native booster resolves by objective.
    clf.get_booster().save_model("/tmp/t33_reg_ubj.ubj")
    native = xgb.Booster()
    native.load_model("/tmp/t33_reg_ubj.ubj")
    assert ModelRegistry.resolve_adapter(native).algorithm_key == "xgboost"


def test_binary_shap_margin_identity_with_nan(binary_data) -> None:
    Xb, yb = binary_data
    model = XGBClassifier(n_estimators=25, max_depth=3, random_state=1).fit(Xb, yb)
    adapter = XGBClassifierAdapter()
    batch = adapter.predict_batch(model, Xb[:12])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:12]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:12])), rtol=1e-9
    )
    booster = model.get_booster()
    dmat = xgb.DMatrix(np.ascontiguousarray(Xb[:12], dtype=np.float32))
    margin = np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64).ravel()
    proba = np.asarray(model.predict_proba(Xb[:12]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, Xb[:12], list(range(12)), ["a", "b"])
    assert MAX_TRAJECTORY_POINTS == 50
    assert traces[0].extra_mechanics["has_missing_values"] is True  # row 0 holds a NaN
    assert traces[1].extra_mechanics["has_missing_values"] is False
    assert traces[0].extra_mechanics["booster_type"] == "gbtree"
    for i, tr in enumerate(traces):
        # Exact TreeSHAP additivity in raw margin space (DoD atol=1e-4).
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(margin[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        mech = tr.extra_mechanics
        assert mech["model_family"] == "xgboost"
        assert mech["objective"] == "binary:logistic"
        assert mech["n_boost_rounds"] == 25
        assert mech["n_parallel_trees"] == 1
        assert len(mech["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        assert mech["staged_trajectory"][-1] == pytest.approx(float(proba[i]), abs=1e-3)
        assert len(mech["top_shap_features"]) == min(TOP_STAGES, 2)
        for step in mech["top_shap_features"]:
            assert set(step) >= {"rank", "feature_idx", "feature_name", "shap_value"}
        assert mech["sampled_round_indices"][0] == 0
        assert mech["sampled_round_indices"][-1] == 24
        assert len(mech["leaf_indices_sample"]) == min(100, 25)


def test_binary_target_zero_negated_view(binary_data) -> None:
    Xb, yb = binary_data
    model = XGBClassifier(n_estimators=10, random_state=1).fit(Xb, yb)
    booster = model.get_booster()
    dmat = xgb.DMatrix(np.ascontiguousarray(Xb[:1], dtype=np.float32))
    margin = float(np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64)[0])
    tr = XGBClassifierAdapter().trace_internal_flow(
        model, Xb[:1], [0], ["a", "b"], target_class_idx=0
    )[0]
    assert tr.extra_mechanics["negated_target_zero_view"] is True
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), -margin, rtol=1e-4, atol=1e-4
    )
    proba0 = float(np.asarray(model.predict_proba(Xb[:1]), dtype=np.float64)[0, 0])
    assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(proba0, abs=1e-3)


def test_multiclass_per_group_columns(multi_data) -> None:
    Xm, ym = multi_data
    model = XGBClassifier(n_estimators=12, max_depth=2, random_state=1).fit(Xm, ym)
    booster = model.get_booster()
    dmat = xgb.DMatrix(np.ascontiguousarray(Xm[:5], dtype=np.float32))
    margins = np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64)
    for k in range(3):
        traces = XGBClassifierAdapter().trace_internal_flow(
            model, Xm[:5], list(range(5)), FEATURES_4, target_class_idx=k
        )
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(margins[i, k]),
                rtol=1e-4,
                atol=1e-4,
            )
            assert tr.extra_mechanics["resolved_class_idx"] == k
        proba_k = np.asarray(model.predict_proba(Xm[:5]), dtype=np.float64)[:, k]
        for i, tr in enumerate(traces):
            assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(
                float(proba_k[i]), abs=1e-3
            )


def test_regressor_objectives_and_ttr(reg_data) -> None:
    X, y = reg_data
    for objective, yy in [
        ("reg:squarederror", y),
        ("reg:gamma", np.abs(y) + 0.5),
        ("count:poisson", np.abs(y) + 0.5),
    ]:
        model = XGBRegressor(n_estimators=12, objective=objective, random_state=1).fit(
            X, yy.astype(np.float32)
        )
        traces = XGBRegressorAdapter().trace_internal_flow(model, X[:5], list(range(5)), ["a", "b", "c"])
        booster = model.get_booster()
        dmat = xgb.DMatrix(np.ascontiguousarray(X[:5], dtype=np.float32))
        margins = np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64).ravel()
        preds = np.asarray(model.predict(X[:5]), dtype=np.float64)
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(margins[i]),
                rtol=1e-4,
                atol=1e-4,
            )
            assert tr.extra_mechanics["is_regressor"] is True
            assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(
                float(preds[i]), abs=1e-3
            )
    # TTR wrappers: no-op keeps SHAP, real transformer remaps (parity both).
    ttr = TransformedTargetRegressor(regressor=XGBRegressor(n_estimators=8, random_state=1))
    ttr.fit(X, y)
    assert ModelRegistry.resolve_adapter(ttr).algorithm_key == "xgboost_regressor"
    traces = XGBRegressorAdapter().trace_internal_flow(ttr, X[:2], [0, 1], ["a", "b", "c"])
    preds = np.asarray(ttr.predict(X[:2]), dtype=np.float64)
    for i, t in enumerate(traces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()), float(preds[i]), rtol=1e-5, atol=1e-6
        )


def test_native_ubj_json_roundtrip_and_schema(binary_data) -> None:
    Xb, yb = binary_data
    model = XGBClassifier(n_estimators=10, random_state=1).fit(Xb, yb)
    with tempfile.TemporaryDirectory() as tmp:
        ubj = Path(tmp) / "m.ubj"
        js = Path(tmp) / "m.json"
        model.get_booster().save_model(ubj)
        model.get_booster().save_model(js)
        for path in (ubj, js):
            booster = xgb.Booster()
            booster.load_model(path)
            schema = UniversalSchemaDetector.extract_schema(booster)
            assert schema.n_features_in == 2
            assert schema.classes == [0, 1]
            UniversalSchemaDetector.validate_is_fitted(booster)
            traces = XGBClassifierAdapter().trace_internal_flow(
                booster, Xb[:3], [0, 1, 2], [f.name for f in schema.features]
            )
            dmat = xgb.DMatrix(np.ascontiguousarray(Xb[:3], dtype=np.float32))
            margins = np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64).ravel()
            for i, tr in enumerate(traces):
                np.testing.assert_allclose(
                    tr.base_value + sum(tr.feature_contributions.values()),
                    float(margins[i]),
                    rtol=1e-4,
                    atol=1e-4,
                )
                assert tr.final_prediction in (0, 1)


def test_hinge_gblinear_rf_and_early_stopping(binary_data) -> None:
    Xb, yb = binary_data
    # binary:hinge has no probability output — labels + one-hot confidences.
    hinge = XGBClassifier(n_estimators=8, objective="binary:hinge", random_state=1).fit(Xb, yb)
    traces = XGBClassifierAdapter().trace_internal_flow(hinge, Xb[:3], [0, 1, 2], ["a", "b"])
    assert [t.final_prediction for t in traces] == [int(v) for v in hinge.predict(Xb[:3])]
    # gblinear: SHAP works, leaves/dump are skipped (edge #4).
    dtrain = xgb.DMatrix(np.ascontiguousarray(Xb, dtype=np.float32), label=yb)
    bl = xgb.train({"objective": "binary:logistic", "booster": "gblinear"}, dtrain, num_boost_round=5)
    assert ModelRegistry.resolve_adapter(bl).algorithm_key == "xgboost"
    tr = XGBClassifierAdapter().trace_internal_flow(bl, Xb[:2], [0, 1], ["a", "b"])[0]
    assert tr.extra_mechanics["booster_type"] == "gblinear"
    assert tr.extra_mechanics["leaf_indices_sample"] == []
    assert tr.extra_mechanics["tree_topology_rounds"] == []
    dmat = xgb.DMatrix(np.ascontiguousarray(Xb[:1], dtype=np.float32))
    margin = float(np.asarray(bl.predict(dmat, output_margin=True), dtype=np.float64)[0])
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), margin, rtol=1e-4, atol=1e-4
    )
    # XGBRF: single boosting round with parallel trees.
    rf = XGBRFClassifier(n_estimators=6, max_depth=2, random_state=1).fit(Xb, yb)
    assert ModelRegistry.resolve_adapter(rf).algorithm_key == "xgboost"
    tr = XGBClassifierAdapter().trace_internal_flow(rf, Xb[:2], [0, 1], ["a", "b"])[0]
    assert tr.extra_mechanics["n_boost_rounds"] == 1
    assert tr.extra_mechanics["n_parallel_trees"] == 6
    assert len(tr.extra_mechanics["leaf_indices_sample"]) == 1
    # Early stopping caps effective rounds at best_iteration + 1 (edge #3).
    # Train on structure but validate on noise: eval logloss degrades fast.
    rng_es = np.random.default_rng(55)
    X_es = rng_es.normal(size=(400, 4)).astype(np.float32)
    y_es = (X_es[:, 0] > 0).astype(int)
    X_va = rng_es.normal(size=(400, 4)).astype(np.float32)
    y_va = rng_es.integers(0, 2, size=400)
    es = XGBClassifier(n_estimators=100, early_stopping_rounds=5, eval_metric="logloss", random_state=1)
    es.fit(X_es, y_es, eval_set=[(X_va, y_va)], verbose=False)
    assert es.get_booster().num_boosted_rounds() < 100
    tr = XGBClassifierAdapter().trace_internal_flow(
        es, X_es[:1], [0], ["a", "b", "c", "d"]
    )[0]
    assert tr.extra_mechanics["n_boost_rounds"] == es.best_iteration + 1
    dmat = xgb.DMatrix(np.ascontiguousarray(X_es[:1], dtype=np.float32))
    margin = float(np.asarray(es.get_booster().predict(dmat, output_margin=True), dtype=np.float64)[0])
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), margin, rtol=1e-4, atol=1e-4
    )


def test_categorical_pipeline_and_wrappers(binary_data) -> None:
    Xb, yb = binary_data
    rng = np.random.default_rng(54)
    cat = rng.integers(0, 4, size=len(Xb)).astype(np.float64)
    import pandas as pd

    df = pd.DataFrame({"c": pd.Categorical(cat.astype(int)), "n": Xb[:, 1]})
    yy = ((cat >= 2).astype(int) ^ (yb)).astype(int)
    model = XGBClassifier(n_estimators=8, enable_categorical=True, random_state=1).fit(df, yy)
    Xc = np.column_stack([cat.astype(np.float32), Xb[:, 1]]).astype(np.float32)
    traces = XGBClassifierAdapter().trace_internal_flow(model, Xc[:4], [0, 1, 2, 3], ["c", "n"])
    booster = model.get_booster()
    dmat = xgb.DMatrix(np.ascontiguousarray(Xc[:4], dtype=np.float32), feature_names=["c", "n"])
    margins = np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(margins[i]),
            rtol=1e-4,
            atol=1e-4,
        )
    assert traces[0].extra_mechanics["categorical_feature_idx"] == [0]
    # Pipeline wrapper resolves through unwrapping.
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("xgb", XGBClassifier(n_estimators=8, random_state=1))]
    )
    Xn = np.nan_to_num(Xb).astype(np.float32)
    pipe.fit(Xn, yb)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "xgboost"
    ptraces = XGBClassifierAdapter().trace_internal_flow(pipe, Xn[:2], [0, 1], ["a", "b"])
    assert [t.final_prediction for t in ptraces] == [int(v) for v in pipe.predict(Xn[:2])]


def test_trajectory_cap_topology_and_surface(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    big = XGBClassifier(n_estimators=120, random_state=1).fit(Xb, yb)
    traces = XGBClassifierAdapter().trace_internal_flow(big, Xb[:3], [0, 1, 2], ["a", "b"])
    booster = big.get_booster()
    for tr in traces:
        assert len(tr.extra_mechanics["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        assert len(tr.extra_mechanics["sampled_round_indices"]) == min(120, MAX_TRAJECTORY_POINTS)
        assert len(tr.extra_mechanics["tree_topology_rounds"]) == 10
        dmat = xgb.DMatrix(np.ascontiguousarray(Xb[[tr.row_index]], dtype=np.float32))
        margin = float(np.asarray(booster.predict(dmat, output_margin=True), dtype=np.float64)[0])
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), margin, rtol=1e-4, atol=1e-4
        )
    mesh = XGBClassifierAdapter().compute_3d_surface(big, _spec(), n_features=2)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    grid = np.zeros((400, 2), dtype=np.float32)
    xs = np.linspace(-4.0, 6.0, 20, dtype=np.float32)
    ys = np.linspace(-3.0, 5.0, 20, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0], grid[:, 1] = xx.ravel(), yy.ravel()
    expected = np.asarray(big.predict_proba(grid), dtype=np.float64)[:, 1]
    np.testing.assert_allclose(np.asarray(mesh.z_values), expected, rtol=1e-4)
    ov = mesh.extra_overlays
    assert ov["model_family"] == "xgboost"
    assert ov["objective"] == "binary:logistic"
    assert ov["n_boost_rounds"] == 120
    assert len(ov["tree_topology_sample"]) == 10
    assert len(ov["split_lines_sample"]) <= 30
    assert set(ov["feature_importances"]) == {"gain", "weight", "cover", "total_gain", "total_cover"}
    assert len(ov["feature_importances"]["gain"]) == 2
    # Topology node shape: internal splits carry gain/cover, leaves carry weight.
    first_tree = ov["tree_topology_sample"][0]["nodes"]
    assert any("leaf" in n for n in first_tree)
    assert any(n.get("gain", 0) > 0 for n in first_tree if "leaf" not in n)

    reg = XGBRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    mesh_r = XGBRegressorAdapter().compute_3d_surface(
        reg,
        GridSpec3D(
            axis_x_feature_index=0,
            axis_y_feature_index=1,
            x_min=-2.0,
            x_max=2.0,
            y_min=-2.0,
            y_max=2.0,
            resolution=20,
            fixed_feature_medians={2: 0.0},
        ),
        n_features=3,
    )
    assert mesh_r.extra_overlays["is_regressor"] is True
    assert mesh_r.extra_overlays["boundary_threshold"] is None

    big_spec = _spec(resolution=200)
    assert XGBClassifierAdapter().compute_3d_surface(big, big_spec, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(big, buf)
    buf.seek(0)
    assert len(XGBClassifierAdapter().trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2

    # get_score parity: gain importances match the booster for the top feature.
    top_gain = max(ov["feature_importances"]["gain"])
    raw_scores = booster.get_score(importance_type="gain")
    assert top_gain == pytest.approx(max(raw_scores.values()), rel=1e-5)

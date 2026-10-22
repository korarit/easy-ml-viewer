"""Phase 35 — CatBoost symmetric oblivious-tree adapter math identity + staged tests."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.boosting.catboost_adapter import (
    MAX_TRAJECTORY_POINTS,
    TOP_STAGES,
    CatBoostClassifierAdapter,
    CatBoostRegressorAdapter,
    load_catboost_from_path,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_CAT = ["income", "city"]


def _binary_frame(seed: int = 71, n: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    income = rng.uniform(20.0, 80.0, size=n).astype(float)
    city = rng.choice(["Bangkok", "Chiang Mai"], size=n)
    df = pd.DataFrame({"income": income, "city": city})
    y = ((df["income"] > 50).astype(int) ^ (df["city"] == "Bangkok").astype(int)).astype(int).values
    return df, y


def _frame_to_object(df: pd.DataFrame) -> np.ndarray:
    X = np.empty((len(df), df.shape[1]), dtype=object)
    for j, col in enumerate(df.columns):
        X[:, j] = df[col].to_numpy()
    return X


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": 20.0,
        "x_max": 80.0,
        "y_min": 20.0,
        "y_max": 80.0,
        "resolution": 20,
        "fixed_feature_medians": {},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def test_registry_resolves_catboost_variants() -> None:
    df, y = _binary_frame()
    clf = CatBoostClassifier(iterations=10, depth=3, verbose=False, random_seed=1).fit(
        df, y, cat_features=["city"]
    )
    Xr = np.random.default_rng(73).normal(size=(60, 2))
    yr = Xr[:, 0] * 2.0 - Xr[:, 1]
    reg = CatBoostRegressor(iterations=10, depth=3, verbose=False, random_seed=1).fit(Xr, yr)
    for model, key in [(clf, "catboost"), (reg, "catboost_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
        assert adapter.supports_nan_input is True
    # .cbm 2-stage probing: regression loss switches to the regressor.
    with tempfile.TemporaryDirectory() as tmp:
        clf.save_model(str(Path(tmp) / "c.cbm"), format="cbm")
        reg.save_model(str(Path(tmp) / "r.cbm"), format="cbm")
        assert type(load_catboost_from_path(Path(tmp) / "c.cbm")).__name__ == "CatBoostClassifier"
        assert type(load_catboost_from_path(Path(tmp) / "r.cbm")).__name__ == "CatBoostRegressor"
        assert ModelRegistry.resolve_adapter(
            load_catboost_from_path(Path(tmp) / "r.cbm")).algorithm_key == "catboost_regressor"


def test_binary_shap_raw_identity_with_categorical_nan() -> None:
    df, y = _binary_frame()
    # Edge #2: NaN in the categorical column must not raise (→ __MISSING__).
    df.loc[0, "city"] = np.nan
    df.loc[1, "income"] = np.nan
    train = df.copy()
    train["city"] = train["city"].fillna("__MISSING__").astype(str)
    model = CatBoostClassifier(iterations=25, depth=3, verbose=False, random_seed=1).fit(
        train, y, cat_features=["city"]
    )
    adapter = CatBoostClassifierAdapter()
    X = _frame_to_object(df)
    batch = adapter.predict_batch(model, X[:12])
    assert np.asarray(batch.predictions).tolist() == np.asarray(model.predict(train[:12])).ravel().tolist()
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(train[:12])), rtol=1e-9
    )
    pool = Pool(data=train[:12], cat_features=["city"])
    raw = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64).ravel()
    proba = np.asarray(model.predict_proba(train[:12]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, X[:12], list(range(12)), FEATURES_CAT)
    assert MAX_TRAJECTORY_POINTS == 50
    assert traces[0].extra_mechanics["model_family"] == "catboost"
    assert traces[0].extra_mechanics["n_iterations"] == 25
    assert traces[0].extra_mechanics["native_categorical_features"] == ["city"]
    assert traces[0].extra_mechanics["categorical_feature_idx"] == [1]
    assert traces[0].extra_mechanics["has_missing_values"] is True  # row 1 holds numeric NaN
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(raw[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        mech = tr.extra_mechanics
        assert str(mech["loss_function"]).startswith("Logloss")
        assert len(mech["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        assert mech["staged_trajectory"][-1] == pytest.approx(float(proba[i]), abs=1e-3)
        assert len(mech["top_shap_features"]) == min(TOP_STAGES, 2)
        for step in mech["top_shap_features"]:
            assert set(step) >= {"rank", "feature_idx", "feature_name", "shap_value"}
        assert mech["sampled_iteration_indices"][0] == 0
        assert mech["sampled_iteration_indices"][-1] == 24
        assert len(mech["leaf_indices_sample"]) == min(100, 25)
        # Phase 35 HUD: bitwise leaf badges + CTR pill.
        assert mech["leaf_bitmask_badges"], "expected bitwise leaf badges"
        assert set(mech["leaf_bitmask_badges"][0]) >= {"tree", "leaf", "bitmask"}
        assert mech["active_leaf_depth"] is not None
        # Leaves downcast to uint16 range (depth <= 16 → < 65536).
        assert all(0 <= v <= 65535 for v in mech["leaf_indices_sample"])


def test_binary_target_zero_negated_view() -> None:
    df, y = _binary_frame()
    model = CatBoostClassifier(iterations=10, depth=3, verbose=False, random_seed=1).fit(
        df, y, cat_features=["city"]
    )
    X = _frame_to_object(df)
    pool = Pool(data=df[:1], cat_features=["city"])
    raw = float(np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64)[0])
    tr = CatBoostClassifierAdapter().trace_internal_flow(
        model, X[:1], [0], FEATURES_CAT, target_class_idx=0
    )[0]
    assert tr.extra_mechanics["negated_target_zero_view"] is True
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), -raw, rtol=1e-4, atol=1e-4
    )
    proba0 = float(np.asarray(model.predict_proba(df[:1]), dtype=np.float64)[0, 0])
    assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(proba0, abs=1e-3)


def test_multiclass_3d_shap_winning_slice() -> None:
    rng = np.random.default_rng(72)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)
    model = CatBoostClassifier(
        iterations=12, depth=3, verbose=False, random_seed=1, loss_function="MultiClass"
    ).fit(X, y)
    pool = Pool(data=X[:5])
    raw3 = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"))
    assert raw3.ndim == 3  # (N, C, F+1) — CatBoost multiclass explicit class axis
    assert raw3.shape == (5, 3, 4)
    raw_scores = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64)
    for k in range(3):
        traces = CatBoostClassifierAdapter().trace_internal_flow(
            model, X[:5], list(range(5)), ["f0", "f1", "f2"], target_class_idx=k
        )
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(raw_scores[i, k]),
                rtol=1e-4,
                atol=1e-4,
            )
            assert tr.extra_mechanics["resolved_class_idx"] == k
        proba_k = np.asarray(model.predict_proba(X[:5]), dtype=np.float64)[:, k]
        for i, tr in enumerate(traces):
            assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(
                float(proba_k[i]), abs=1e-3
            )


def test_regressor_and_pipeline_wrapper() -> None:
    rng = np.random.default_rng(73)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    model = CatBoostRegressor(iterations=12, depth=3, verbose=False, random_seed=1).fit(X, y)
    traces = CatBoostRegressorAdapter().trace_internal_flow(model, X[:5], list(range(5)), ["a", "b", "c"])
    preds = np.asarray(model.predict(X[:5]), dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(preds[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert tr.extra_mechanics["is_regressor"] is True
        assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(float(preds[i]), abs=1e-3)
    # Pipeline wrapper resolves through unwrapping (numeric-only path).
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("cb", CatBoostRegressor(iterations=8, depth=3, verbose=False, random_seed=1))]
    )
    Xn = np.nan_to_num(X).astype(np.float32)
    yn = Xn[:, 0] * 2.0 - Xn[:, 1]
    pipe.fit(Xn, yn)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "catboost_regressor"
    ptraces = CatBoostRegressorAdapter().trace_internal_flow(pipe, Xn[:2], [0, 1], ["a", "b", "c"])
    assert [float(t.final_prediction) for t in ptraces] == pytest.approx(
        [float(v) for v in pipe.predict(Xn[:2])], abs=1e-3
    )


def test_native_cbm_roundtrip_and_schema() -> None:
    df, y = _binary_frame()
    model = CatBoostClassifier(iterations=10, depth=3, verbose=False, random_seed=1).fit(
        df, y, cat_features=["city"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.cbm"
        model.save_model(str(path), format="cbm")
        loaded = load_catboost_from_path(path)
        schema = UniversalSchemaDetector.extract_schema(loaded)
        assert schema.n_features_in == 2
        assert schema.classes == [0, 1]
        assert schema.features[1].expected_dtype == "categorical"
        UniversalSchemaDetector.validate_is_fitted(loaded)
        X = _frame_to_object(df)
        traces = CatBoostClassifierAdapter().trace_internal_flow(
            loaded, X[:3], [0, 1, 2], [f.name for f in schema.features]
        )
        pool = Pool(data=df[:3], cat_features=["city"])
        raw = np.asarray(loaded.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64).ravel()
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(raw[i]),
                rtol=1e-4,
                atol=1e-4,
            )
            assert tr.final_prediction in (0, 1)


def test_oblivious_topology_symmetry_and_surface() -> None:
    df, y = _binary_frame()
    big = CatBoostClassifier(iterations=60, depth=4, verbose=False, random_seed=1).fit(
        df, y, cat_features=["city"]
    )
    X = _frame_to_object(df)
    traces = CatBoostClassifierAdapter().trace_internal_flow(big, X[:3], [0, 1, 2], FEATURES_CAT)
    pool = Pool(data=df, cat_features=["city"])
    raw = np.asarray(big.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64).ravel()
    for tr in traces:
        assert len(tr.extra_mechanics["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        assert len(tr.extra_mechanics["sampled_iteration_indices"]) == min(60, MAX_TRAJECTORY_POINTS)
        assert len(tr.extra_mechanics["tree_topology_rounds"]) == 10
        # Oblivious symmetry: every level of a tree shares one feature/threshold.
        topo = tr.extra_mechanics["oblivious_tree_topology"]
        assert topo, "expected oblivious topology"
        for tree in topo:
            assert tree["depth"] >= 1
            assert tree["num_leaves"] == 2 ** tree["depth"]
            assert len(tree["levels"]) == tree["depth"]
            assert len(tree["leaf_bitmasks"]) == tree["num_leaves"]
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(raw[tr.row_index]),
            rtol=1e-4,
            atol=1e-4,
        )
    mesh = CatBoostClassifierAdapter().compute_3d_surface(big, _spec(), n_features=2)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    ov = mesh.extra_overlays
    assert ov["model_family"] == "catboost"
    assert str(ov["loss_function"]).startswith("Logloss")
    assert ov["n_iterations"] == 60
    assert len(ov["tree_topology_sample"]) == 10
    assert len(ov["split_lines_sample"]) <= 30
    assert ov["categorical_feature_idx"] == [1]
    assert set(ov["feature_importances"]) >= {"prediction_values_change", "gain"}
    assert len(ov["feature_importances"]["gain"]) == 2

    buf = io.BytesIO()
    joblib.dump(big, buf)
    buf.seek(0)
    assert len(CatBoostClassifierAdapter().trace_internal_flow(joblib.load(buf), X[:2], [0, 1], FEATURES_CAT)) == 2

    big_spec = _spec(resolution=200)
    assert CatBoostClassifierAdapter().compute_3d_surface(big, big_spec, n_features=2).resolution == 100

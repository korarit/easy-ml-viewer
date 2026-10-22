"""Phase 34 — LightGBM leaf-wise adapter math identity + staged tests."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.boosting.lightgbm_adapter import (
    MAX_TRAJECTORY_POINTS,
    TOP_STAGES,
    LightGBMClassifierAdapter,
    LightGBMRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(61)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    X[::11, 0] = np.nan  # native missing support (supports_nan_input)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(62)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(63)
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


def test_registry_resolves_lgbm_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = LGBMClassifier(n_estimators=10, verbose=-1, random_state=1).fit(Xb, yb)
    reg = LGBMRegressor(n_estimators=10, verbose=-1, random_state=1).fit(Xr, yr)
    for model, key in [(clf, "lightgbm"), (reg, "lightgbm_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
        assert adapter.supports_nan_input is True
    # Native booster resolves by objective.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.txt"
        clf.booster_.save_model(path)
        native = lgb.Booster(model_file=str(path))
        assert ModelRegistry.resolve_adapter(native).algorithm_key == "lightgbm"


def test_binary_shap_raw_score_identity_with_nan(binary_data) -> None:
    Xb, yb = binary_data
    model = LGBMClassifier(n_estimators=25, num_leaves=15, verbose=-1, random_state=1).fit(Xb, yb)
    adapter = LightGBMClassifierAdapter()
    batch = adapter.predict_batch(model, Xb[:12])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:12]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:12])), rtol=1e-9
    )
    raw = np.asarray(model.booster_.predict(Xb[:12], raw_score=True), dtype=np.float64).ravel()
    proba = np.asarray(model.predict_proba(Xb[:12]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, Xb[:12], list(range(12)), ["a", "b"])
    assert MAX_TRAJECTORY_POINTS == 50
    assert traces[0].extra_mechanics["has_missing_values"] is True  # row 0 holds a NaN
    assert traces[1].extra_mechanics["has_missing_values"] is False
    assert traces[0].extra_mechanics["model_family"] == "lightgbm"
    assert traces[0].extra_mechanics["n_iterations"] == 25
    assert traces[0].extra_mechanics["n_trees_total"] == 25
    for i, tr in enumerate(traces):
        # Exact TreeSHAP additivity in raw-score space (DoD atol=1e-4).
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(raw[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        mech = tr.extra_mechanics
        assert str(mech["objective"]).startswith("binary")
        assert mech["n_boost_rounds"] == 25
        assert len(mech["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        assert mech["staged_trajectory"][-1] == pytest.approx(float(proba[i]), abs=1e-3)
        assert len(mech["top_shap_features"]) == min(TOP_STAGES, 2)
        for step in mech["top_shap_features"]:
            assert set(step) >= {"rank", "feature_idx", "feature_name", "shap_value"}
        assert mech["sampled_iteration_indices"][0] == 0
        assert mech["sampled_iteration_indices"][-1] == 24
        assert len(mech["leaf_indices_sample"]) == min(100, 25)
        # HUD leaf-depth gauge: active depth within the sampled tree stats.
        assert mech["active_leaf_depth"] is not None
        stats = mech["tree_depth_stats"][0]
        assert stats["min_leaf_depth"] <= mech["active_leaf_depth"] <= stats["max_leaf_depth"]


def test_binary_target_zero_negated_view(binary_data) -> None:
    Xb, yb = binary_data
    model = LGBMClassifier(n_estimators=10, verbose=-1, random_state=1).fit(Xb, yb)
    raw = float(np.asarray(model.booster_.predict(Xb[:1], raw_score=True), dtype=np.float64)[0])
    tr = LightGBMClassifierAdapter().trace_internal_flow(
        model, Xb[:1], [0], ["a", "b"], target_class_idx=0
    )[0]
    assert tr.extra_mechanics["negated_target_zero_view"] is True
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), -raw, rtol=1e-4, atol=1e-4
    )
    proba0 = float(np.asarray(model.predict_proba(Xb[:1]), dtype=np.float64)[0, 0])
    assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(proba0, abs=1e-3)


def test_multiclass_flattened_2d_reshape(multi_data) -> None:
    Xm, ym = multi_data
    model = LGBMClassifier(n_estimators=12, num_leaves=7, verbose=-1, random_state=1).fit(Xm, ym)
    booster = model.booster_
    raw_flat = np.asarray(booster.predict(Xm[:5], pred_contrib=True), dtype=np.float64)
    # Edge #1: LightGBM multiclass SHAP is a flattened 2D array.
    assert raw_flat.ndim == 2
    assert raw_flat.shape == (5, 3 * (4 + 1))
    raw_scores = np.asarray(booster.predict(Xm[:5], raw_score=True), dtype=np.float64)
    for k in range(3):
        traces = LightGBMClassifierAdapter().trace_internal_flow(
            model, Xm[:5], list(range(5)), FEATURES_4, target_class_idx=k
        )
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(raw_scores[i, k]),
                rtol=1e-4,
                atol=1e-4,
            )
            assert tr.extra_mechanics["resolved_class_idx"] == k
        proba_k = np.asarray(model.predict_proba(Xm[:5]), dtype=np.float64)[:, k]
        for i, tr in enumerate(traces):
            assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(
                float(proba_k[i]), abs=1e-3
            )
        # Multiclass leaves reshape to (n, iters, C).
        leaves = np.asarray(booster.predict(Xm[:5], pred_leaf=True))
        assert leaves.shape[1] == traces[0].extra_mechanics["n_iterations"] * 3


def test_regressor_ranker_and_pipeline(reg_data) -> None:
    X, y = reg_data
    model = LGBMRegressor(n_estimators=12, verbose=-1, random_state=1).fit(X, y)
    traces = LightGBMRegressorAdapter().trace_internal_flow(model, X[:5], list(range(5)), ["a", "b", "c"])
    preds = np.asarray(model.predict(X[:5]), dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(preds[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert tr.extra_mechanics["is_regressor"] is True
        assert tr.extra_mechanics["staged_trajectory"][-1] == pytest.approx(
            float(preds[i]), abs=1e-3
        )
    # Ranker resolves to the regressor adapter.
    from lightgbm import LGBMRanker

    rank_labels = (y > np.median(y)).astype(int)
    ranker = LGBMRanker(n_estimators=8, verbose=-1, random_state=1).fit(
        X, rank_labels, group=[50, 50, 50]
    )
    assert ModelRegistry.resolve_adapter(ranker).algorithm_key == "lightgbm_regressor"
    rank_traces = LightGBMRegressorAdapter().trace_internal_flow(
        ranker, X[:2], [0, 1], ["a", "b", "c"]
    )
    rank_preds = np.asarray(ranker.predict(X[:2]), dtype=np.float64)
    for i, tr in enumerate(rank_traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(rank_preds[i]),
            rtol=1e-4,
            atol=1e-4,
        )
    # Pipeline wrapper resolves through unwrapping.
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("lgbm", LGBMClassifier(n_estimators=8, verbose=-1, random_state=1))]
    )
    Xn = np.nan_to_num(X).astype(np.float32)
    yn = (Xn[:, 0] > 0).astype(int)
    pipe.fit(Xn, yn)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "lightgbm"
    ptraces = LightGBMClassifierAdapter().trace_internal_flow(pipe, Xn[:2], [0, 1], ["a", "b", "c"])
    assert [t.final_prediction for t in ptraces] == [int(v) for v in pipe.predict(Xn[:2])]


def test_native_txt_roundtrip_and_schema(binary_data) -> None:
    Xb, yb = binary_data
    model = LGBMClassifier(n_estimators=10, verbose=-1, random_state=1).fit(Xb, yb)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.txt"
        model.booster_.save_model(path)
        booster = lgb.Booster(model_file=str(path))
        schema = UniversalSchemaDetector.extract_schema(booster)
        assert schema.n_features_in == 2
        assert schema.classes == [0, 1]
        UniversalSchemaDetector.validate_is_fitted(booster)
        traces = LightGBMClassifierAdapter().trace_internal_flow(
            booster, Xb[:3], [0, 1, 2], [f.name for f in schema.features]
        )
        raw = np.asarray(booster.predict(Xb[:3], raw_score=True), dtype=np.float64).ravel()
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(raw[i]),
                rtol=1e-4,
                atol=1e-4,
            )
            assert tr.final_prediction in (0, 1)


def test_single_tree_categorical_and_dump_cap(binary_data) -> None:
    Xb, yb = binary_data
    import pandas as pd

    rng = np.random.default_rng(64)
    cat = rng.integers(0, 3, size=len(Xb)).astype(int)
    df = pd.DataFrame({"c": pd.Categorical(cat), "n": Xb[:, 1].astype(float)})
    yy = ((cat >= 1).astype(int) ^ (yb)).astype(int)
    params = {"objective": "binary", "verbose": -1}
    dtrain = lgb.Dataset(df, label=yy, categorical_feature=["c"])
    booster = lgb.train(params, dtrain, num_boost_round=6)
    assert ModelRegistry.resolve_adapter(booster).algorithm_key == "lightgbm"
    Xc = np.column_stack([cat.astype(np.float32), Xb[:, 1]]).astype(np.float32)
    traces = LightGBMClassifierAdapter().trace_internal_flow(booster, Xc[:4], [0, 1, 2, 3], ["c", "n"])
    raw = np.asarray(booster.predict(Xc[:4], raw_score=True), dtype=np.float64).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(raw[i]),
            rtol=1e-4,
            atol=1e-4,
        )
    # Edge #4: categorical split parsed into bitset categories.
    assert traces[0].extra_mechanics["categorical_feature_idx"] == [0]
    dump = booster.dump_model(num_iteration=100)

    def _walk(node):
        if not isinstance(node, dict) or "split_index" not in node:
            return []
        found = [node] if node.get("decision_type") == "==" else []
        return found + _walk(node.get("left_child")) + _walk(node.get("right_child"))

    cat_nodes = [n for t in dump["tree_info"] for n in _walk(t.get("tree_structure"))]
    assert cat_nodes, "expected at least one categorical split in dump_model"
    from app.domains.model_engine.adapters.boosting.lightgbm_adapter import (
        _split_categorical_threshold,
    )

    assert _split_categorical_threshold(cat_nodes[0]["threshold"]) != []
    assert _split_categorical_threshold("0||3||5") == [0, 3, 5]
    # Single iteration model: leaf extraction never collapses (edge #3 guard).
    tiny = lgb.train(params, lgb.Dataset(df, label=yy, categorical_feature=["c"]), num_boost_round=1)
    leaves = np.asarray(tiny.predict(Xc[:2], pred_leaf=True))
    assert leaves.reshape(2, -1).shape == (2, 1)
    tr = LightGBMClassifierAdapter().trace_internal_flow(tiny, Xc[:2], [0, 1], ["c", "n"])[0]
    assert len(tr.extra_mechanics["leaf_indices_sample"]) == 1


def test_trajectory_cap_topology_and_surface(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    big = LGBMClassifier(n_estimators=120, verbose=-1, random_state=1).fit(Xb, yb)
    traces = LightGBMClassifierAdapter().trace_internal_flow(big, Xb[:3], [0, 1, 2], ["a", "b"])
    booster = big.booster_
    for tr in traces:
        assert len(tr.extra_mechanics["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        assert len(tr.extra_mechanics["sampled_iteration_indices"]) == min(120, MAX_TRAJECTORY_POINTS)
        assert len(tr.extra_mechanics["tree_topology_rounds"]) == 10
        raw = float(np.asarray(booster.predict(Xb[[tr.row_index]], raw_score=True), dtype=np.float64)[0])
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), raw, rtol=1e-4, atol=1e-4
        )
        assert tr.extra_mechanics["active_leaf_depth"] is not None
    mesh = LightGBMClassifierAdapter().compute_3d_surface(big, _spec(), n_features=2)
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
    assert ov["model_family"] == "lightgbm"
    assert str(ov["objective"]).startswith("binary")
    assert ov["n_iterations"] == 120
    assert len(ov["tree_topology_sample"]) == 10
    assert len(ov["split_lines_sample"]) <= 30
    assert set(ov["feature_importances"]) == {"gain", "split"}
    assert len(ov["feature_importances"]["gain"]) == 2
    assert len(ov["tree_depth_stats"]) == 10
    # Topology node shape: internal splits carry gain/depth, leaves carry leaf_index.
    first_tree = ov["tree_topology_sample"][0]["nodes"]
    assert any("leaf_index" in n for n in first_tree)
    assert any(n.get("gain", 0) > 0 for n in first_tree if "leaf_index" not in n)

    reg = LGBMRegressor(n_estimators=10, verbose=-1, random_state=1).fit(Xr, yr)
    mesh_r = LightGBMRegressorAdapter().compute_3d_surface(
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
    assert LightGBMClassifierAdapter().compute_3d_surface(big, big_spec, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(big, buf)
    buf.seek(0)
    assert len(LightGBMClassifierAdapter().trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2

    # get_score parity: feature_importance matches the booster for the top feature.
    top_gain = max(ov["feature_importances"]["gain"])
    raw_scores = list(booster.feature_importance(importance_type="gain"))
    assert top_gain == pytest.approx(float(max(raw_scores)), rel=1e-5)

"""Phase 30 — adaboost cascade adapter math identity + staged tests."""

from __future__ import annotations

import io
import warnings

import joblib
import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.domains.model_engine.adapters.trees.adaboost import (
    MAX_TRAJECTORY_POINTS,
    AdaBoostClassifierAdapter,
    AdaBoostRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(34)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(35)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(36)
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


def test_registry_resolves_adaboost_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = AdaBoostClassifier(n_estimators=10, random_state=1).fit(Xb, yb)
    reg = AdaBoostRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    for model, key in [(clf, "adaboost"), (reg, "adaboost_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN


def test_stump_cascade_waterfall_matches_staged(binary_data, multi_data) -> None:
    Xb, yb = binary_data
    model = AdaBoostClassifier(n_estimators=25, random_state=1).fit(Xb, yb)
    adapter = AdaBoostClassifierAdapter()
    batch = adapter.predict_batch(model, Xb[:10])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:10]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:10])), rtol=1e-9
    )
    proba = np.asarray(model.predict_proba(Xb[:10]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, Xb[:10], list(range(10)), ["a", "b"])
    m_actual = len(model.estimators_)
    assert MAX_TRAJECTORY_POINTS == 50
    for i, tr in enumerate(traces):
        # Cascade identity: base + ΣΦ == final staged == predict_proba.
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(proba[i]),
            rtol=1e-9,
            atol=1e-9,
        )
        assert tr.base_value == pytest.approx(0.5)
        mech = tr.extra_mechanics
        assert mech["model_family"] == "adaboost"
        assert mech["n_estimators_actual"] == m_actual
        assert mech["early_terminated"] is False or m_actual < 25
        assert len(mech["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        # Trajectory is rounded to 4 decimals; allow rounding slack.
        assert mech["staged_trajectory"][-1] == pytest.approx(float(proba[i]), abs=1e-4)
        assert len(mech["top_influential_learners"]) == min(8, m_actual)
        for step in mech["top_influential_learners"]:
            assert set(step) >= {"stage", "weight", "error", "feature_idx", "threshold", "delta", "cumulative"}
            assert 0 <= step["stage"] < m_actual
        # sampled indices always span first..last stage.
        assert mech["sampled_stage_indices"][0] == 0
        assert mech["sampled_stage_indices"][-1] == m_actual - 1

    Xm, ym = multi_data
    multi = AdaBoostClassifier(n_estimators=15, random_state=1).fit(Xm, ym)
    traces = AdaBoostClassifierAdapter().trace_internal_flow(
        multi, Xm[:5], list(range(5)), FEATURES_4, target_class_idx=2
    )
    proba_k = np.asarray(multi.predict_proba(Xm[:5]), dtype=np.float64)[:, 2]
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(proba_k[i]),
            rtol=1e-9,
            atol=1e-9,
        )
        assert tr.base_value == pytest.approx(1.0 / 3.0)
        assert tr.extra_mechanics["resolved_class_idx"] == 2


def test_regressor_weighted_median_parity(reg_data) -> None:
    X, y = reg_data
    model = AdaBoostRegressor(n_estimators=20, random_state=1).fit(X, y)
    adapter = AdaBoostRegressorAdapter()
    preds = np.asarray(model.predict(X[:10]), dtype=np.float64)
    traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), ["a", "b", "c"])
    staged_all = list(model.staged_predict(X[:10]))
    final_staged = np.asarray(staged_all[-1], dtype=np.float64)
    staged_first = np.asarray(staged_all[0], dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(preds[i]),
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(float(preds[i]), float(final_staged[i]), rtol=1e-12)
        assert tr.extra_mechanics["is_regressor"] is True
        # Base is mean(S_1) over the chunk (blueprint S_0 definition).
        assert tr.base_value == pytest.approx(float(np.mean(staged_first)), rel=1e-9)


def test_early_stopping_deep_and_nontree_estimators(binary_data) -> None:
    Xb, yb = binary_data
    # Edge #1: perfect fit on a tiny set terminates long before n_estimators.
    # (Xb[:20] holds a single class, which also exercises the staged-generator
    # fallback: sklearn's own staged_predict_proba raises on n_classes_ == 1.)
    tiny = AdaBoostClassifier(n_estimators=200, random_state=0).fit(Xb[:20], yb[:20])
    assert len(tiny.estimators_) < 200
    tr = AdaBoostClassifierAdapter().trace_internal_flow(tiny, Xb[:2], [0, 1], ["a", "b"])[0]
    assert tr.extra_mechanics["n_estimators_actual"] == len(tiny.estimators_)
    assert tr.extra_mechanics["early_terminated"] is True
    proba = np.asarray(tiny.predict_proba(Xb[:1]), dtype=np.float64)[0, -1]
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()), float(proba), rtol=1e-9, atol=1e-9
    )
    assert tr.extra_mechanics["staged_trajectory"] == [round(float(proba), 4)] * len(
        tr.extra_mechanics["staged_trajectory"]
    )
    # Deep-tree base estimator: decision_path share still exact.
    deep = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3), n_estimators=10, random_state=1
    ).fit(Xb, yb)
    traces = AdaBoostClassifierAdapter().trace_internal_flow(deep, Xb[:5], list(range(5)), ["a", "b"])
    proba_d = np.asarray(deep.predict_proba(Xb[:5]), dtype=np.float64)[:, 1]
    for i, t in enumerate(traces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()), float(proba_d[i]), rtol=1e-9, atol=1e-9
        )
    # Edge #2: non-tree base estimator distributes via |coef·x| (parity intact).
    lin = AdaBoostClassifier(
        estimator=LogisticRegression(max_iter=500), n_estimators=5, random_state=1
    ).fit(Xb, yb)
    assert ModelRegistry.resolve_adapter(lin).algorithm_key == "adaboost"
    traces = AdaBoostClassifierAdapter().trace_internal_flow(lin, Xb[:3], [0, 1, 2], ["a", "b"])
    proba_l = np.asarray(lin.predict_proba(Xb[:3]), dtype=np.float64)[:, 1]
    for i, t in enumerate(traces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()), float(proba_l[i]), rtol=1e-9, atol=1e-9
        )


def test_pipeline_and_transformed_target_wrappers(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("ab", AdaBoostClassifier(n_estimators=10, random_state=1))]
    )
    pipe.fit(Xb, yb)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "adaboost"
    traces = AdaBoostClassifierAdapter().trace_internal_flow(pipe, Xb[:3], [0, 1, 2], ["a", "b"])
    assert [t.final_prediction for t in traces] == list(pipe.predict(Xb[:3]))
    proba = np.asarray(pipe.predict_proba(Xb[:3]), dtype=np.float64)[:, 1]
    for i, t in enumerate(traces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()), float(proba[i]), rtol=1e-9, atol=1e-9
        )
    ttr = TransformedTargetRegressor(regressor=AdaBoostRegressor(n_estimators=8, random_state=1))
    ttr.fit(Xr, yr)
    assert ModelRegistry.resolve_adapter(ttr).algorithm_key == "adaboost_regressor"
    traces = AdaBoostRegressorAdapter().trace_internal_flow(ttr, Xr[:2], [0, 1], ["a", "b", "c"])
    preds = np.asarray(ttr.predict(Xr[:2]), dtype=np.float64)
    for i, t in enumerate(traces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()), float(preds[i]), rtol=1e-9, atol=1e-9
        )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        AdaBoostClassifierAdapter().trace_internal_flow(pipe, Xb[:2], [0, 1], ["a", "b"])
    assert not [w for w in rec if "feature names" in str(w.message).lower()]


def test_trajectory_cap_and_surface_curtains(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    big = AdaBoostClassifier(n_estimators=200, random_state=1).fit(Xb, yb)
    m_actual = len(big.estimators_)
    traces = AdaBoostClassifierAdapter().trace_internal_flow(big, Xb[:3], [0, 1, 2], ["a", "b"])
    for tr in traces:
        assert len(tr.extra_mechanics["staged_trajectory"]) <= MAX_TRAJECTORY_POINTS
        # Cap only binds when the cascade is longer than the budget.
        assert len(tr.extra_mechanics["sampled_stage_indices"]) == min(m_actual, MAX_TRAJECTORY_POINTS)
    mesh = AdaBoostClassifierAdapter().compute_3d_surface(big, _spec(), n_features=2)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    grid = np.zeros((400, 2), dtype=np.float32)
    xs = np.linspace(-4.0, 6.0, 20, dtype=np.float32)
    ys = np.linspace(-3.0, 5.0, 20, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0], grid[:, 1] = xx.ravel(), yy.ravel()
    expected = np.asarray(big.predict_proba(grid), dtype=np.float64)[:, 1]
    np.testing.assert_allclose(np.asarray(mesh.z_values), expected, rtol=1e-9)
    ov = mesh.extra_overlays
    assert ov["model_family"] == "adaboost"
    assert ov["n_estimators_actual"] == m_actual
    assert len(ov["estimator_weights"]) == m_actual
    assert len(ov["estimator_errors"]) == m_actual
    assert len(ov["stump_splits"]) <= 50
    assert all("threshold" in s and "weight" in s for s in ov["stump_splits"])

    reg = AdaBoostRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    mesh_r = AdaBoostRegressorAdapter().compute_3d_surface(
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
    assert AdaBoostClassifierAdapter().compute_3d_surface(big, big_spec, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(big, buf)
    buf.seek(0)
    assert len(AdaBoostClassifierAdapter().trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2

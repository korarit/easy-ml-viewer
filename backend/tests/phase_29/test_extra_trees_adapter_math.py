"""Phase 29 — extra-trees adapter math identity + uncertainty tests."""

from __future__ import annotations

import io
import time
import warnings

import joblib
import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.trees.extra_trees import (
    MAX_SPLITS_PER_FEATURE,
    ExtraTreesClassifierAdapter,
    ExtraTreesRegressorAdapter,
    extract_split_threshold_distribution,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(32)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(33)
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


def test_registry_resolves_extra_trees_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = ExtraTreesClassifier(n_estimators=10, random_state=1).fit(Xb, yb)
    reg = ExtraTreesRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    for model, key in [(clf, "extra_trees"), (reg, "extra_trees_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN


def test_classifier_waterfall_and_uncertainty(binary_data, multi_data) -> None:
    Xb, yb = binary_data
    model = ExtraTreesClassifier(n_estimators=25, random_state=1).fit(Xb, yb)
    adapter = ExtraTreesClassifierAdapter()
    batch = adapter.predict_batch(model, Xb[:10])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:10]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:10])), rtol=1e-6
    )
    proba = np.asarray(model.predict_proba(Xb[:10]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, Xb[:10], list(range(10)), ["a", "b"])
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            tr.extra_mechanics["inter_tree_mean"],
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            float(proba[i]), tr.extra_mechanics["inter_tree_mean"], rtol=1e-4, atol=1e-4
        )
        mech = tr.extra_mechanics
        assert mech["model_family"] == "extra_trees"
        assert mech["n_estimators_total"] == 25
        # Uncertainty contract: ddof=0, clipped halo, ordered quantiles.
        assert 0.0 <= mech["inter_tree_std"] <= 0.5
        assert mech["uncertainty_radius"] == pytest.approx(
            min(max(2.0 * mech["inter_tree_std"], 0.0), 1.0), rel=1e-3, abs=1e-4
        )
        assert mech["prediction_p05"] <= mech["inter_tree_mean"] <= mech["prediction_p95"]
        assert set(mech["contribution_std"]) == {"a", "b"}
        assert all(v >= 0.0 for v in mech["contribution_std"].values())

    Xm, ym = multi_data
    multi = ExtraTreesClassifier(n_estimators=15, random_state=1).fit(Xm, ym)
    traces = ExtraTreesClassifierAdapter().trace_internal_flow(
        multi, Xm[:5], list(range(5)), FEATURES_4, target_class_idx=2
    )
    proba_k = np.asarray(multi.predict_proba(Xm[:5]), dtype=np.float64)[:, 2]
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(proba_k[i]),
            rtol=1e-4,
            atol=1e-4,
        )
        assert tr.extra_mechanics["resolved_class_idx"] == 2
        assert tr.extra_mechanics["predictive_entropy"] is not None
        assert tr.extra_mechanics["predictive_entropy"] >= 0.0


def test_single_tree_never_nan_and_regressor_span_guard(reg_data) -> None:
    X, y = reg_data
    # Edge #1: n_estimators == 1 must yield 0.0 std/halo, never NaN.
    X1 = (X[:, :2] > 0).astype(np.float32)
    y1 = (X[:, 0] > 0).astype(int)
    one = ExtraTreesClassifier(n_estimators=1, random_state=1).fit(X1, y1)
    tr = ExtraTreesClassifierAdapter().trace_internal_flow(
        one, X[:2, :2].astype(np.float32), [0, 1], ["a", "b"]
    )
    for t in tr:
        assert t.extra_mechanics["inter_tree_std"] == 0.0
        assert t.extra_mechanics["uncertainty_radius"] == 0.0
        assert t.extra_mechanics["prediction_p05"] == t.extra_mechanics["prediction_p95"]
    # Edge #2: constant-prediction chunk -> span guard keeps halo finite.
    const = ExtraTreesRegressor(n_estimators=10, max_depth=1, random_state=1).fit(X, np.full_like(y, 3.0))
    traces = ExtraTreesRegressorAdapter().trace_internal_flow(const, X[:5], list(range(5)), ["a", "b", "c"])
    for t in traces:
        assert np.isfinite(t.extra_mechanics["uncertainty_radius"])
        assert 0.0 <= t.extra_mechanics["uncertainty_radius"] <= 1.0
    # Normal regressor parity.
    model = ExtraTreesRegressor(n_estimators=20, random_state=1).fit(X, y)
    preds = np.asarray(model.predict(X[:10]), dtype=np.float64)
    traces = ExtraTreesRegressorAdapter().trace_internal_flow(model, X[:10], list(range(10)), ["a", "b", "c"])
    for i, t in enumerate(traces):
        np.testing.assert_allclose(
            t.base_value + sum(t.feature_contributions.values()),
            float(preds[i]),
            rtol=1e-4,
            atol=1e-3,
        )
        assert t.extra_mechanics["is_regressor"] is True


def test_pipeline_and_transformed_target_wrappers(reg_data) -> None:
    X, y = reg_data
    adapter = ExtraTreesRegressorAdapter()
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("et", ExtraTreesRegressor(n_estimators=10, random_state=1))]
    )
    pipe.fit(X, y)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "extra_trees_regressor"
    traces = adapter.trace_internal_flow(pipe, X[:3], [0, 1, 2], ["a", "b", "c"])
    assert [t.final_prediction for t in traces] == list(pipe.predict(X[:3]))
    ttr = TransformedTargetRegressor(regressor=ExtraTreesRegressor(n_estimators=5, random_state=1))
    ttr.fit(X, y)
    assert ModelRegistry.resolve_adapter(ttr).algorithm_key == "extra_trees_regressor"
    assert len(adapter.trace_internal_flow(ttr, X[:2], [0, 1], ["a", "b", "c"])) == 2
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        adapter.trace_internal_flow(pipe, X[:2], [0, 1], ["a", "b", "c"])
    assert not [w for w in rec if "feature names" in str(w.message).lower()]


def test_export_cap_and_split_distribution(binary_data) -> None:
    Xb, yb = binary_data
    big = ExtraTreesClassifier(n_estimators=150, random_state=1).fit(Xb, yb)
    adapter = ExtraTreesClassifierAdapter()
    t0 = time.perf_counter()
    traces = adapter.trace_internal_flow(big, Xb[:5], list(range(5)), ["a", "b"])
    assert time.perf_counter() - t0 < 5.0
    for tr in traces:
        assert tr.extra_mechanics["n_estimators_total"] == 150
        assert tr.extra_mechanics["n_estimators_exported"] == 100
        assert len(tr.extra_mechanics["tree_predictions_sample"]) == 100
    dist = extract_split_threshold_distribution(big, 2)
    assert set(dist) == {"0", "1"}
    assert all(len(v) <= MAX_SPLITS_PER_FEATURE for v in dist.values())
    assert sum(len(v) for v in dist.values()) > 0


def test_surface_mean_uncertainty_and_splits(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = ExtraTreesClassifier(n_estimators=15, random_state=1).fit(Xb, yb)
    adapter = ExtraTreesClassifierAdapter()
    mesh = adapter.compute_3d_surface(clf, _spec(), n_features=2)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    grid = np.zeros((400, 2), dtype=np.float32)
    xs = np.linspace(-4.0, 6.0, 20, dtype=np.float32)
    ys = np.linspace(-3.0, 5.0, 20, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0], grid[:, 1] = xx.ravel(), yy.ravel()
    expected = np.asarray(clf.predict_proba(grid), dtype=np.float64)[:, 1]
    np.testing.assert_allclose(np.asarray(mesh.z_values), expected, rtol=1e-6)
    ov = mesh.extra_overlays
    assert ov["model_family"] == "extra_trees"
    assert ov["n_estimators_total"] == 15
    assert ov["boundary_threshold"] == 0.5
    assert len(ov["uncertainty_z_values"]) == 400
    assert len(ov["uncertainty_halo_values"]) == 400
    assert all(0.0 <= v <= 1.0 for v in ov["uncertainty_halo_values"])
    assert set(ov["split_threshold_distribution"]) >= {"0", "1"}
    assert all(len(v) <= MAX_SPLITS_PER_FEATURE for v in ov["split_threshold_distribution"].values())

    reg = ExtraTreesRegressor(n_estimators=10, random_state=1).fit(Xr, yr)
    mesh_r = ExtraTreesRegressorAdapter().compute_3d_surface(
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
    assert len(mesh_r.extra_overlays["uncertainty_z_values"]) == 400

    big = _spec(resolution=200)
    assert adapter.compute_3d_surface(clf, big, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    assert len(adapter.trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2

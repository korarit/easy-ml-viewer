"""Phase 28 — random-forest adapter math identity tests."""

from __future__ import annotations

import io
import time

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.trees.random_forest import (
    MAX_EXPORTED_TREES,
    RandomForestClassifierAdapter,
    RandomForestRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(28)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(30)
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


def test_registry_resolves_forest_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=1).fit(Xb, yb)
    reg = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=1).fit(Xr, yr)
    for model, key in [(clf, "random_forest"), (reg, "random_forest_regressor")]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.ENSEMBLE_SMOOTH_TERRAIN


def test_classifier_waterfall_matches_forest_mean(binary_data, multi_data) -> None:
    Xb, yb = binary_data
    model = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=1).fit(Xb, yb)
    adapter = RandomForestClassifierAdapter()
    batch = adapter.predict_batch(model, Xb[:10])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:10]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:10])), rtol=1e-6
    )
    proba = np.asarray(model.predict_proba(Xb[:10]), dtype=np.float64)[:, 1]
    traces = adapter.trace_internal_flow(model, Xb[:10], list(range(10)), ["a", "b"])
    for i, tr in enumerate(traces):
        # Ensemble identity: base + contributions == inter-tree mean == soft-vote proba
        # (float32 leaf lookup + 5-decimal telemetry rounding -> 1e-4 tolerance).
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            tr.extra_mechanics["inter_tree_mean"],
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(float(proba[i]), tr.extra_mechanics["inter_tree_mean"], rtol=1e-4, atol=1e-4)
        mech = tr.extra_mechanics
        assert mech["n_estimators_total"] == 25
        assert mech["n_estimators_exported"] == 25
        assert 0.5 <= mech["consensus_ratio"] <= 1.0
        assert 0.0 <= mech["inter_tree_std"] <= 0.5
        assert len(mech["tree_predictions_sample"]) == 25
        assert len(mech["leaf_indices_sample"]) == 25
        assert mech["model_family"] == "random_forest"

    Xm, ym = multi_data
    multi = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=1).fit(Xm, ym)
    for k in range(3):
        traces = RandomForestClassifierAdapter().trace_internal_flow(
            multi, Xm[:5], list(range(5)), FEATURES_4, target_class_idx=k
        )
        proba_k = np.asarray(multi.predict_proba(Xm[:5]), dtype=np.float64)[:, k]
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(proba_k[i]),
                rtol=1e-5,
                atol=1e-5,
            )
            assert tr.extra_mechanics["resolved_class_idx"] == k


def test_regressor_waterfall_matches_prediction(reg_data) -> None:
    X, y = reg_data
    model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=1).fit(X, y)
    adapter = RandomForestRegressorAdapter()
    preds = np.asarray(model.predict(X[:10]), dtype=np.float64)
    traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), ["a", "b", "c"])
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(preds[i]),
            rtol=1e-5,
            atol=1e-4,
        )
        assert tr.extra_mechanics["is_regressor"] is True
        assert tr.extra_mechanics["consensus_ratio"] == pytest.approx(
            1.0 / (1.0 + tr.extra_mechanics["inter_tree_std"]), rel=1e-3, abs=1e-4
        )


def test_large_forest_export_cap_and_pipeline(binary_data) -> None:
    Xb, yb = binary_data
    big = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=1).fit(Xb, yb)
    adapter = RandomForestClassifierAdapter()
    assert MAX_EXPORTED_TREES == 100
    t0 = time.perf_counter()
    traces = adapter.trace_internal_flow(big, Xb[:5], list(range(5)), ["a", "b"])
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0  # generous CI guard: capped decision_path tracing
    for tr in traces:
        assert tr.extra_mechanics["n_estimators_total"] == 150
        assert tr.extra_mechanics["n_estimators_exported"] == 100
        assert len(tr.extra_mechanics["tree_predictions_sample"]) == 100
        # Summary stats still cover all 150 trees (mean over full forest).
        assert 0.0 <= tr.extra_mechanics["inter_tree_mean"] <= 1.0
    # Pipeline wrapper resolves and predicts through the full chain.
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=10, max_depth=4))])
    pipe.fit(Xb, yb)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "random_forest"
    traces_pipe = adapter.trace_internal_flow(pipe, Xb[:3], [0, 1, 2], ["a", "b"])
    assert [t.final_prediction for t in traces_pipe] == list(pipe.predict(Xb[:3]))
    # No feature_names_in UserWarning escapes: recwarn returns empty under filter.
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        adapter.trace_internal_flow(big, Xb[:2], [0, 1], ["a", "b"])
    assert not [w for w in rec if "feature names" in str(w.message).lower()]


def test_surface_consensus_parity_and_cap(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=1).fit(Xb, yb)
    adapter = RandomForestClassifierAdapter()
    mesh = adapter.compute_3d_surface(clf, _spec(), n_features=2)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    assert mesh.extra_overlays["n_estimators_total"] == 15
    assert mesh.extra_overlays["boundary_threshold"] == 0.5
    assert mesh.extra_overlays["model_family"] == "random_forest"
    assert len(mesh.extra_overlays["feature_importances"]) == 2
    # Surface z matches native soft-vote proba on the same grid.
    grid = np.zeros((400, 2), dtype=np.float32)
    xs = np.linspace(-4.0, 6.0, 20, dtype=np.float32)
    ys = np.linspace(-3.0, 5.0, 20, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0], grid[:, 1] = xx.ravel(), yy.ravel()
    expected = np.asarray(clf.predict_proba(grid), dtype=np.float64)[:, 1]
    np.testing.assert_allclose(np.asarray(mesh.z_values), expected, rtol=1e-6)

    reg = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=1).fit(Xr, yr)
    mesh_r = RandomForestRegressorAdapter().compute_3d_surface(
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

    big = _spec(resolution=200)
    assert adapter.compute_3d_surface(clf, big, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    assert len(adapter.trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2

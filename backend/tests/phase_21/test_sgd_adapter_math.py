"""Phase 21 — SGD / Perceptron / PassiveAggressive adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.linear.sgd import (
    PROBA_LOSSES,
    SgdClassifierAdapter,
    SgdRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURE_NAMES = ["feature_velocity", "feature_pressure", "feature_temp", "feature_vibration"]


@pytest.fixture()
def clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=2, n_redundant=1, random_state=0
    )
    return StandardScaler().fit_transform(X).astype(np.float32), y


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=4, n_informative=3, noise=5.0, random_state=0)
    return StandardScaler().fit_transform(X).astype(np.float32), y.astype(np.float64)


def _spec() -> GridSpec3D:
    return GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=1,
        x_min=-3.0,
        x_max=3.0,
        y_min=-3.0,
        y_max=3.0,
        resolution=40,
        fixed_feature_medians={2: 0.5, 3: -0.25},
    )


def test_registry_resolves_online_linear_family(clf_data, reg_data) -> None:
    Xc, yc = clf_data
    Xr, yr = reg_data
    clf_cases = [
        SGDClassifier(loss="hinge", max_iter=500, tol=1e-3, random_state=0).fit(Xc, yc),
        SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(Xc, yc),
        Perceptron(max_iter=500, tol=1e-3, random_state=0).fit(Xc, yc),
        PassiveAggressiveClassifier(max_iter=500, tol=1e-3, random_state=0).fit(Xc, yc),
    ]
    for model in clf_cases:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "sgd_classifier"
        assert adapter.visual_3d_type == Visual3DType.FLAT_HYPERPLANE
    reg_cases = [
        SGDRegressor(max_iter=500, tol=1e-3, random_state=0).fit(Xr, yr),
        PassiveAggressiveRegressor(max_iter=500, tol=1e-3, random_state=0).fit(Xr, yr),
    ]
    for model in reg_cases:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "sgd_regressor"


@pytest.mark.parametrize("loss", ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"])
def test_classifier_waterfall_and_margin_identity(clf_data, loss: str) -> None:
    X, y = clf_data
    model = SGDClassifier(loss=loss, max_iter=2000, tol=1e-4, random_state=0).fit(X, y)
    adapter = SgdClassifierAdapter()
    batch = adapter.predict_batch(model, X[:25])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:25]))
    traces = adapter.trace_internal_flow(model, X[:25], list(range(25)), FEATURE_NAMES)
    assert len(traces) == 25
    scores = np.asarray(model.decision_function(X[:25])).ravel()
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), scores[i], rtol=1e-4, atol=1e-2
        )
        mech = tr.extra_mechanics
        assert mech["loss"] == loss
        assert mech["has_native_proba"] == (loss in PROBA_LOSSES)
        assert mech["is_pseudo_proba"] == (loss not in PROBA_LOSSES)
        assert mech["raw_score"] == pytest.approx(float(scores[i]))
        assert mech["geometric_distance"] == pytest.approx(float(scores[i]) / mech["w_norm"])
        assert mech["inside_margin"] == (abs(float(scores[i])) < 1.0)
        assert mech["margin_width"] == pytest.approx(1.0)
        assert mech["total_weight_updates_t"] > 0
        # Geometric contributions sum to the geometric distance.
        geom_sum = float(np.sum(list(mech["geometric_contributions"])))
        assert geom_sum == pytest.approx(mech["geometric_distance"] - mech["intercept"] / mech["w_norm"], rel=1e-4)


def test_hinge_has_no_native_proba_but_pseudo_confidence(clf_data) -> None:
    X, y = clf_data
    model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-4, random_state=0).fit(X, y)
    adapter = SgdClassifierAdapter()
    traces = adapter.trace_internal_flow(model, X[:3], [0, 1, 2], FEATURE_NAMES)
    for tr in traces:
        mech = tr.extra_mechanics
        assert mech["has_native_proba"] is False
        proba = np.asarray(mech["probabilities"])
        assert proba.shape == (2,)
        assert np.isclose(proba.sum(), 1.0)
        assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_log_loss_native_proba_parity_and_modified_huber_saturation(clf_data) -> None:
    X, y = clf_data
    log = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4, random_state=0).fit(X, y)
    adapter = SgdClassifierAdapter()
    traces = adapter.trace_internal_flow(log, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    native = np.asarray(log.predict_proba(X[:5]))
    for i, tr in enumerate(traces):
        assert tr.extra_mechanics["has_native_proba"] is True
        np.testing.assert_allclose(np.asarray(tr.extra_mechanics["probabilities"]), native[i], rtol=1e-5)

    huber = SGDClassifier(loss="modified_huber", max_iter=2000, tol=1e-4, random_state=0).fit(X, y)
    traces_h = adapter.trace_internal_flow(huber, X[:10], list(range(10)), FEATURE_NAMES)
    scores = np.asarray(huber.decision_function(X[:10])).ravel()
    native_h = np.asarray(huber.predict_proba(X[:10]))
    for i, tr in enumerate(traces_h):
        np.testing.assert_allclose(np.asarray(tr.extra_mechanics["probabilities"]), native_h[i], rtol=1e-5)
        if abs(float(scores[i])) >= 1.0:
            # Piecewise-linear saturation outside [-1, +1] (blueprint edge #2).
            assert max(native_h[i]) == pytest.approx(1.0)


def test_perceptron_and_passive_aggressive_fallback_loss(clf_data) -> None:
    X, y = clf_data
    adapter = SgdClassifierAdapter()
    perc = Perceptron(max_iter=1000, tol=1e-3, random_state=0).fit(X, y)
    traces = adapter.trace_internal_flow(perc, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert traces[0].extra_mechanics["loss"] == "perceptron"
    assert traces[0].extra_mechanics["has_native_proba"] is False
    scores = np.asarray(perc.decision_function(X[:3])).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), scores[i], rtol=1e-4, atol=1e-2
        )
    pa = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3, random_state=0).fit(X, y)
    traces_pa = adapter.trace_internal_flow(pa, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert traces_pa[0].extra_mechanics["loss"] in ("hinge", "squared_hinge")
    scores_pa = np.asarray(pa.decision_function(X[:3])).ravel()
    for i, tr in enumerate(traces_pa):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), scores_pa[i], rtol=1e-4, atol=1e-2
        )


def test_multiclass_ovr_class_switch(clf_data) -> None:
    X, _ = clf_data
    rng = np.random.default_rng(5)
    y3 = rng.integers(0, 3, size=X.shape[0]).astype(np.int32)
    model = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4, random_state=0).fit(X, y3)
    assert np.asarray(model.coef_).shape[0] == 3
    adapter = SgdClassifierAdapter()
    for k in range(3):
        traces = adapter.trace_internal_flow(model, X[:4], [0, 1, 2, 3], FEATURE_NAMES, target_class_idx=k)
        scores = np.asarray(model.decision_function(X[:4]))[:, k]
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()), scores[i], rtol=1e-4, atol=1e-2
            )
            assert tr.extra_mechanics["target_class_idx"] == k
    alt = adapter.trace_internal_flow(model, X[:2], [0, 1], FEATURE_NAMES, target_class_idx=2)
    assert alt[0].extra_mechanics["target_class_idx"] == 2


@pytest.mark.parametrize(
    "loss", ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
)
def test_regressor_waterfall_and_epsilon_tube(reg_data, loss: str) -> None:
    X, y = reg_data
    model = SGDRegressor(loss=loss, max_iter=2000, tol=1e-4, random_state=0).fit(X, y)
    adapter = SgdRegressorAdapter()
    batch = adapter.predict_batch(model, X[:25])
    np.testing.assert_allclose(np.asarray(batch.predictions), model.predict(X[:25]), rtol=1e-4)
    traces = adapter.trace_internal_flow(model, X[:25], list(range(25)), FEATURE_NAMES)
    assert len(traces) == 25
    preds = np.asarray(model.predict(X[:25])).ravel()
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), preds[i], rtol=1e-4, atol=1e-2
        )
        mech = tr.extra_mechanics
        assert mech["loss"] == loss
        assert mech["margin_width"] == pytest.approx(mech["epsilon"])
        assert mech["is_classifier"] is False


def test_passive_aggressive_regressor_and_pipeline_sparse(clf_data, reg_data) -> None:
    Xr, yr = reg_data
    adapter = SgdRegressorAdapter()
    pa = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3, random_state=0).fit(Xr, yr)
    traces = adapter.trace_internal_flow(pa, Xr[:3], [0, 1, 2], FEATURE_NAMES)
    assert len(traces) == 3

    Xc, yc = clf_data
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4))]
    ).fit(Xc, yc)
    clf_adapter = SgdClassifierAdapter()
    traces_pipe = clf_adapter.trace_internal_flow(pipe, Xc[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    preds = np.asarray(pipe.decision_function(Xc[:5])).ravel()
    for i, tr in enumerate(traces_pipe):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), preds[i], rtol=1e-3, atol=1e-2
        )
    mesh = clf_adapter.compute_3d_surface(pipe, _spec(), n_features=4)
    assert mesh.extra_overlays["gpu_formula_valid"] is False
    # Pipeline surface must be evaluated through the scaler (raw-space grid).
    spec = _spec()
    res = spec.resolution
    grid = np.zeros((res * res, 4), dtype=np.float32)
    xs = np.linspace(spec.x_min, spec.x_max, res, dtype=np.float32)
    ys = np.linspace(spec.y_min, spec.y_max, res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()
    grid[:, 2] = 0.5
    grid[:, 3] = -0.25
    expected = np.asarray(pipe.decision_function(grid), dtype=np.float64).ravel()
    np.testing.assert_allclose(np.asarray(mesh.z_values), expected, rtol=1e-4, atol=1e-2)

    try:
        from scipy import sparse  # noqa: PLC0415
    except ImportError:
        pytest.skip("scipy not available")
        return
    X_sparse = sparse.csr_matrix(Xc[:10])
    plain = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-4, random_state=0).fit(Xc, yc)
    traces_sp = clf_adapter.trace_internal_flow(plain, X_sparse, list(range(5)), FEATURE_NAMES)
    assert len(traces_sp) == 5


def test_surface_corridor_overlays_and_resolution_clamp(clf_data, reg_data) -> None:
    Xc, yc = clf_data
    Xr, yr = reg_data
    clf = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-4, random_state=0).fit(Xc, yc)
    adapter = SgdClassifierAdapter()
    big = GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=1,
        x_min=-3.0,
        x_max=3.0,
        y_min=-3.0,
        y_max=3.0,
        resolution=200,
        fixed_feature_medians={},
    )
    mesh = adapter.compute_3d_surface(clf, big, n_features=4)
    assert mesh.resolution == 100
    assert len(mesh.z_values) == 100 * 100
    assert mesh.extra_overlays["margin_width"] == pytest.approx(1.0)
    assert mesh.extra_overlays["boundary_threshold"] == 0.0
    assert mesh.extra_overlays["corridor_half_width_geometric"] == pytest.approx(
        1.0 / mesh.extra_overlays["w_norm"]
    )

    spec = _spec()
    mesh2 = adapter.compute_3d_surface(clf, spec, n_features=4)
    w = np.asarray(clf.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(clf.intercept_).ravel()[0])
    off = b + float(w[2]) * 0.5 + float(w[3]) * -0.25
    xs = np.linspace(-3.0, 3.0, 40)
    ys = np.linspace(-3.0, 3.0, 40)
    assert np.isclose(mesh2.z_values[0], w[0] * xs[0] + w[1] * ys[0] + off, atol=1e-3)

    reg = SGDRegressor(loss="epsilon_insensitive", epsilon=0.3, max_iter=1000, tol=1e-4, random_state=0).fit(
        Xr, yr
    )
    mesh_r = SgdRegressorAdapter().compute_3d_surface(reg, spec, n_features=4)
    assert mesh_r.extra_overlays["margin_width"] == pytest.approx(0.3)
    assert mesh_r.extra_overlays["boundary_threshold"] is None
    assert np.all(np.isfinite(np.asarray(mesh_r.z_values)))

    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    traces = adapter.trace_internal_flow(restored, Xc[:2], [0, 1], FEATURE_NAMES)
    assert len(traces) == 2

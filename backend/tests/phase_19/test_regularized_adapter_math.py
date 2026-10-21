"""Phase 19 — Ridge/Lasso/RidgeClassifier adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.linear.regularized import (
    ZERO_TOL,
    RegularizedLinearAdapter,
    RidgeClassifierAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURE_NAMES = ["area_sqft", "rooms", "age_years", "noise_1"]


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=120, n_features=4, n_informative=3, noise=5.0, random_state=42)
    return X.astype(np.float32), y.astype(np.float64)


@pytest.fixture()
def clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3, n_redundant=0, n_classes=2, random_state=42
    )
    return X.astype(np.float32), y


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


def test_registry_resolves_all_nine_classes(reg_data) -> None:
    X, y = reg_data
    _, yc = make_classification(n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=1)
    Xc = X[:60]
    cases = [
        (Ridge(alpha=1.0), X, y, "regularized_linear"),
        (RidgeCV(alphas=[0.1, 1.0], cv=2), X, y, "regularized_linear"),
        (Lasso(alpha=0.5, max_iter=5000), X, y, "regularized_linear"),
        (LassoCV(alphas=[0.1, 1.0], cv=2, max_iter=5000), X, y, "regularized_linear"),
        (LassoLars(alpha=0.5), X, y, "regularized_linear"),
        (LassoLarsCV(cv=2), X, y, "regularized_linear"),
        (LassoLarsIC(criterion="aic"), X, y, "regularized_linear"),
        (RidgeClassifier(alpha=1.0), Xc, yc, "ridge_classifier"),
        (RidgeClassifierCV(alphas=[0.1, 1.0], cv=2), Xc, yc, "ridge_classifier"),
    ]
    for cls_or_inst, Xf, yf, expected_key in cases:
        model = cls_or_inst.fit(Xf, yf)
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == expected_key
        assert adapter.visual_3d_type == Visual3DType.FLAT_HYPERPLANE


def test_ridge_parity_and_waterfall_additivity(reg_data) -> None:
    X, y = reg_data
    model = Ridge(alpha=1.0).fit(X, y)
    adapter = RegularizedLinearAdapter()
    batch = adapter.predict_batch(model, X[:25])
    np.testing.assert_allclose(np.asarray(batch.predictions), model.predict(X[:25]), atol=1e-4)
    traces = adapter.trace_internal_flow(model, X[:25], list(range(25)), FEATURE_NAMES)
    assert len(traces) == 25
    preds = np.asarray(model.predict(X[:25])).ravel()
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds[i], atol=1e-4)
        assert tr.extra_mechanics["penalty_type"] == "L2 (Ridge)"
        assert tr.extra_mechanics["is_classifier"] is False
        assert 0.0 <= tr.extra_mechanics["sparsity_ratio"] <= 1.0


def test_lasso_sparsity_and_all_pruned_banner(reg_data) -> None:
    X, y = reg_data
    sparse = Lasso(alpha=50.0, max_iter=10000).fit(X, y)
    adapter = RegularizedLinearAdapter()
    traces = adapter.trace_internal_flow(sparse, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    assert len(traces) == 5
    w = np.asarray(sparse.coef_, dtype=np.float64)
    expected_zeroed = int(np.sum(np.abs(w) <= ZERO_TOL))
    assert traces[0].extra_mechanics["active_weight_count"] == int(np.sum(np.abs(w) > ZERO_TOL))
    assert traces[0].extra_mechanics["sparsity_ratio"] == pytest.approx(expected_zeroed / 4)
    assert traces[0].extra_mechanics["penalty_type"] == "L1 (Lasso)"
    # Zeroed features contribute exactly 0.0.
    for idx in traces[0].extra_mechanics["zeroed_indices"]:
        assert traces[0].feature_contributions[FEATURE_NAMES[idx]] == 0.0
    # Extreme alpha prunes everything -> flat plane at intercept.
    assert expected_zeroed >= 1

    over = Lasso(alpha=1e6, max_iter=10000).fit(X, y)
    traces_all = adapter.trace_internal_flow(over, X[:2], [0, 1], FEATURE_NAMES)
    assert traces_all[0].extra_mechanics["active_weight_count"] == 0
    mesh = adapter.compute_3d_surface(over, _spec(), n_features=4)
    assert mesh.extra_overlays["zero_influence_axes"] is True
    assert mesh.extra_overlays["is_x_pruned"] in (True, False)


def test_cv_alpha_and_larsic_criterion(reg_data) -> None:
    X, y = reg_data
    adapter = RegularizedLinearAdapter()
    cv_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=2).fit(X, y)
    traces = adapter.trace_internal_flow(cv_model, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert np.isfinite(traces[0].extra_mechanics["alpha"])
    assert traces[0].extra_mechanics["alpha"] in (0.1, 1.0, 10.0)

    lasso_cv = LassoCV(alphas=[0.1, 1.0], cv=2, max_iter=5000).fit(X, y)
    traces_cv = adapter.trace_internal_flow(lasso_cv, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert np.isfinite(traces_cv[0].extra_mechanics["alpha"])

    ic_model = LassoLarsIC(criterion="bic").fit(X, y)
    traces_ic = adapter.trace_internal_flow(ic_model, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert traces_ic[0].extra_mechanics["criterion"] == "bic"


def test_multioutput_target_selector(reg_data) -> None:
    X, y = reg_data
    Y = np.column_stack([y, y * 0.5 + 3.0])
    model = Ridge(alpha=1.0).fit(X, Y)
    assert np.asarray(model.coef_).shape == (2, 4)
    adapter = RegularizedLinearAdapter()
    for target in (0, 1):
        traces = adapter.trace_internal_flow(model, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES, target_idx=target)
        preds = np.asarray(model.predict(X[:5]))[:, target]
        for i, tr in enumerate(traces):
            assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds[i], atol=1e-3)
            assert tr.extra_mechanics["target_idx"] == target


def test_ridge_classifier_pseudo_proba_and_scores(clf_data) -> None:
    X, y = clf_data
    model = RidgeClassifier(alpha=1.0).fit(X, y)
    # Must not require predict_proba.
    assert not hasattr(model, "predict_proba")
    adapter = RidgeClassifierAdapter()
    batch = adapter.predict_batch(model, X[:10])
    assert batch.probabilities is not None and batch.confidences is not None
    probas = np.asarray(batch.probabilities)
    assert probas.shape == (10, 2)
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(10), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:10]))

    traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), FEATURE_NAMES)
    scores = np.asarray(model.decision_function(X[:10])).reshape(-1)
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), scores[i], atol=1e-4)
        assert tr.extra_mechanics["is_pseudo_proba"] is True
        assert tr.extra_mechanics["decision_threshold"] == 0.0
        assert np.isclose(sum(tr.extra_mechanics["probabilities"]), 1.0, atol=1e-6)

    mesh = adapter.compute_3d_surface(model, _spec(), n_features=4)
    assert mesh.extra_overlays["boundary_threshold"] == 0.0
    assert len(mesh.z_values) == 1600


def test_ridge_classifier_multiclass(clf_data) -> None:
    X, _ = clf_data
    rng = np.random.default_rng(3)
    y3 = rng.integers(0, 3, size=len(X))
    model = RidgeClassifier().fit(X, y3)
    adapter = RidgeClassifierAdapter()
    traces = adapter.trace_internal_flow(model, X[:6], list(range(6)), FEATURE_NAMES)
    assert len(traces) == 6
    scores = np.asarray(model.decision_function(X[:6]))
    assert scores.shape == (6, 3)
    for i, tr in enumerate(traces):
        k = tr.extra_mechanics["target_class_idx"]
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), scores[i, k], atol=1e-4)


def test_pipeline_and_sparse_and_no_intercept(reg_data) -> None:
    X, y = reg_data
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]).fit(X, y)
    adapter = RegularizedLinearAdapter()
    traces = adapter.trace_internal_flow(pipe, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    preds = np.asarray(pipe.predict(X[:5])).ravel()
    for i, tr in enumerate(traces):
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds[i], atol=1e-3)
    mesh = adapter.compute_3d_surface(pipe, _spec(), n_features=4)
    assert mesh.extra_overlays["gpu_formula_valid"] is False

    plain = Ridge(alpha=1.0, fit_intercept=False).fit(X, y)
    traces_plain = adapter.trace_internal_flow(plain, X[:2], [0, 1], FEATURE_NAMES)
    assert traces_plain[0].base_value == 0.0

    try:
        from scipy import sparse  # noqa: PLC0415
    except ImportError:
        pytest.skip("scipy not available")
        return
    X_sparse = sparse.csr_matrix(X[:10])
    ridge = Ridge(alpha=1.0).fit(X, y)
    traces_sp = adapter.trace_internal_flow(ridge, X_sparse, list(range(5)), FEATURE_NAMES)
    assert len(traces_sp) == 5
    preds_sp = np.asarray(ridge.predict(X[:5])).ravel()
    for i, tr in enumerate(traces_sp):
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds_sp[i], atol=1e-4)


def test_surface_analytical_plane_and_resolution_clamp(reg_data) -> None:
    X, y = reg_data
    model = Ridge(alpha=2.0).fit(X, y)
    adapter = RegularizedLinearAdapter()
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
    mesh = adapter.compute_3d_surface(model, big, n_features=4)
    assert mesh.resolution == 100
    assert len(mesh.z_values) == 100 * 100
    assert isinstance(mesh.extra_overlays["is_x_pruned"], bool)

    spec = _spec()
    mesh2 = adapter.compute_3d_surface(model, spec, n_features=4)
    w = np.asarray(model.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(model.intercept_).ravel()[0])
    off = b + float(w[2]) * 0.5 + float(w[3]) * -0.25
    xs = np.linspace(-3.0, 3.0, 40)
    ys = np.linspace(-3.0, 3.0, 40)
    assert np.isclose(mesh2.z_values[0], w[0] * xs[0] + w[1] * ys[0] + off, atol=1e-4)
    assert np.isclose(mesh2.extra_overlays["intercept_offset"], off, atol=1e-6)

    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    traces = adapter.trace_internal_flow(restored, X[:2], [0, 1], FEATURE_NAMES)
    assert len(traces) == 2

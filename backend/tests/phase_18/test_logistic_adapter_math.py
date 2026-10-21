"""Phase 18 — LogisticRegressionAdapter math identity tests.

Gates (mirrors test-phase-18 §6):
- predictions / probabilities / decision_scores parity with sklearn (atol 1e-5)
- waterfall additivity: sum(contribs) + base == decision_function (atol 1e-5)
- surface z_values in [0, 1], resolution clamp, zero-influence flag
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.linear.logistic_regression_adapter import (
    LogisticRegressionAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURE_NAMES = ["feat_income", "feat_age", "feat_debt", "feat_score"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    return X.astype(np.float32), y


@pytest.fixture()
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=150,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
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
        resolution=60,
        fixed_feature_medians={2: 0.0, 3: 0.0},
    )


def test_registry_resolves_logistic_variants(binary_data: np.ndarray) -> None:
    X, y = binary_data
    for cls, kwargs in [
        (LogisticRegression, {"solver": "lbfgs", "max_iter": 500}),
        (LogisticRegression, {"solver": "liblinear", "penalty": "l1", "C": 0.5}),
        (LogisticRegressionCV, {"cv": 2, "Cs": [0.1, 1.0]}),
    ]:
        model = cls(**kwargs).fit(X, y)
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "logistic_regression"
        assert adapter.visual_3d_type == Visual3DType.PROBABILITY_SIGMOID_SURFACE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "lbfgs", "C": 1.0},
        {"solver": "liblinear", "penalty": "l1", "C": 0.5},
        {"solver": "lbfgs", "fit_intercept": False},
    ],
)
def test_binary_parity_and_waterfall_additivity(binary_data: np.ndarray, kwargs: dict) -> None:
    X, y = binary_data
    model = LogisticRegression(max_iter=1000, random_state=42, **kwargs).fit(X, y)
    adapter = LogisticRegressionAdapter()

    batch = adapter.predict_batch(model, X[:25])
    assert batch.probabilities is not None and batch.confidences is not None
    probas = np.asarray(batch.probabilities)
    assert probas.shape == (25, 2)
    np.testing.assert_allclose(probas, model.predict_proba(X[:25]), atol=1e-5)
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:25]))
    np.testing.assert_allclose(
        np.asarray(batch.predictions),
        np.asarray(model.predict(X[:25])),
    )

    traces = adapter.trace_internal_flow(
        model_obj=model, X_float32=X[:25], row_indices=list(range(25)), feature_names=FEATURE_NAMES
    )
    assert len(traces) == 25
    scores = np.asarray(model.decision_function(X[:25])).reshape(-1)
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        reconstructed = tr.base_value + sum(tr.feature_contributions.values())
        assert np.isclose(reconstructed, scores[i], atol=1e-5), (i, reconstructed, scores[i])
        assert tr.extra_mechanics["is_binary"] is True
        assert np.isclose(tr.extra_mechanics["logit"], scores[i], atol=1e-5)
        assert np.isclose(sum(tr.extra_mechanics["probabilities"]), 1.0, atol=1e-5)


def test_multiclass_softmax_waterfall_and_surface_mesh(multiclass_data: np.ndarray) -> None:
    X, y = multiclass_data
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
    adapter = LogisticRegressionAdapter()

    traces = adapter.trace_internal_flow(
        model_obj=model, X_float32=X[:20], row_indices=list(range(20)), feature_names=FEATURE_NAMES
    )
    probas = np.asarray(model.predict_proba(X[:20]))
    scores = np.asarray(model.decision_function(X[:20]))
    assert scores.shape == (20, 3)
    np.testing.assert_allclose(np.sum(probas, axis=1), np.ones(20), atol=1e-5)

    winning = np.argmax(probas, axis=1)
    for i, tr in enumerate(traces):
        reconstructed = tr.base_value + sum(tr.feature_contributions.values())
        assert np.isclose(reconstructed, scores[i, winning[i]], atol=1e-5)

    mesh = adapter.compute_3d_surface(model, _spec(), n_features=4)
    assert mesh.visual_type == Visual3DType.PROBABILITY_SIGMOID_SURFACE
    assert mesh.resolution == 60
    assert len(mesh.z_values) == 3600
    assert all(0.0 <= v <= 1.0 for v in mesh.z_values)
    assert mesh.extra_overlays["boundary_threshold"] == 0.5


def test_logistic_cv_intercept_array_and_no_intercept(binary_data: np.ndarray) -> None:
    X, y = binary_data
    cv_model = LogisticRegressionCV(cv=2, Cs=[0.1, 1.0], max_iter=1000).fit(X, y)
    adapter = LogisticRegressionAdapter()
    traces = adapter.trace_internal_flow(
        cv_model, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES
    )
    assert len(traces) == 5
    scores = np.asarray(cv_model.decision_function(X[:5])).reshape(-1)
    for i, tr in enumerate(traces):
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), scores[i], atol=1e-5)

    plain = LogisticRegression(fit_intercept=False, max_iter=1000).fit(X, y)
    traces_plain = adapter.trace_internal_flow(plain, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert traces_plain[0].base_value == 0.0


def test_extreme_outlier_no_overflow_and_sparse_input(binary_data: np.ndarray) -> None:
    X, y = binary_data
    model = LogisticRegression(max_iter=1000).fit(X, y)
    adapter = LogisticRegressionAdapter()

    outlier = np.array([[-1.0e6, 2.0e6, -3.0e6, 4.0e6]], dtype=np.float32)
    with np.errstate(all="raise"):
        traces = adapter.trace_internal_flow(model, outlier, [0], FEATURE_NAMES)
    proba = traces[0].extra_mechanics["probabilities"]
    assert all(0.0 <= v <= 1.0 for v in proba)

    try:
        from scipy import sparse  # noqa: PLC0415
    except ImportError:
        pytest.skip("scipy not available")
        return
    X_sparse = sparse.csr_matrix(X[:10])
    traces_sp = adapter.trace_internal_flow(model, X_sparse, list(range(5)), FEATURE_NAMES)
    assert len(traces_sp) == 5
    scores = np.asarray(model.decision_function(X[:5])).reshape(-1)
    for i, tr in enumerate(traces_sp):
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), scores[i], atol=1e-5)


def test_l1_zero_axes_surface_flags_and_pipeline_parity(binary_data: np.ndarray) -> None:
    X, y = binary_data
    sparse_model = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.01, random_state=42
    ).fit(X, y)
    adapter = LogisticRegressionAdapter()
    mesh = adapter.compute_3d_surface(sparse_model, _spec(), n_features=4)
    assert len(mesh.z_values) == 3600
    assert all(0.0 <= v <= 1.0 for v in mesh.z_values)
    assert isinstance(mesh.extra_overlays["zero_influence_axes"], bool)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]).fit(
        X, y
    )
    traces = adapter.trace_internal_flow(pipe, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    preds = pipe.predict(X[:5])
    for i, tr in enumerate(traces):
        total = tr.base_value + sum(tr.feature_contributions.values())
        assert np.isclose(total, float(np.asarray(pipe.decision_function(X[:5])).ravel()[i]), atol=1e-4)
        assert tr.final_prediction == preds[i]


def test_surface_resolution_clamped_and_joblib_roundtrip(binary_data: np.ndarray) -> None:
    X, y = binary_data
    model = LogisticRegression(max_iter=500).fit(X, y)
    adapter = LogisticRegressionAdapter()
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

    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    traces = adapter.trace_internal_flow(restored, X[:2], [0, 1], FEATURE_NAMES)
    assert len(traces) == 2

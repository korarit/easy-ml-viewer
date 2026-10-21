"""Phase 24 — GaussianNB bell-hills adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.probabilistic.gaussian_nb import (
    GaussianNBAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    centers = np.array(
        [[5.0, 3.4, 1.5, 0.2], [6.0, 2.8, 4.3, 1.3], [6.6, 3.0, 5.5, 2.0]],
        dtype=np.float64,
    )
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for k, c in enumerate(centers):
        xs.append(rng.normal(loc=c, scale=[0.35, 0.3, 0.35, 0.15], size=(60, 4)))
        ys.extend([k] * 60)
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int32)


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=160, n_features=4, n_informative=2, n_redundant=1, random_state=7
    )
    return StandardScaler().fit_transform(X).astype(np.float32), y


def _spec(resolution: int = 40) -> GridSpec3D:
    return GridSpec3D(
        axis_x_feature_index=2,
        axis_y_feature_index=3,
        x_min=0.5,
        x_max=7.0,
        y_min=0.0,
        y_max=2.6,
        resolution=resolution,
        fixed_feature_medians={0: 5.8, 1: 3.0},
    )


def test_registry_resolves_gaussian_nb(multi_data) -> None:
    X, y = multi_data
    model = GaussianNB().fit(X, y)
    adapter = ModelRegistry.resolve_adapter(model)
    assert adapter.algorithm_key == "gaussian_nb"
    assert adapter.visual_3d_type == Visual3DType.GAUSSIAN_ELLIPSOID_DOMES


def test_multiclass_waterfall_matches_joint_ll(multi_data) -> None:
    X, y = multi_data
    model = GaussianNB().fit(X, y)
    adapter = GaussianNBAdapter()
    batch = adapter.predict_batch(model, X[:25])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:25]))
    np.testing.assert_allclose(
        np.asarray(batch.probabilities), np.asarray(model.predict_proba(X[:25])), rtol=1e-6
    )
    joint = np.asarray(model._joint_log_likelihood(X[:25]), dtype=np.float64)
    for k in range(3):
        traces = adapter.trace_internal_flow(
            model, X[:25], list(range(25)), FEATURE_NAMES, target_class_idx=k
        )
        assert len(traces) == 25
        for i, tr in enumerate(traces):
            assert set(tr.feature_contributions) == set(FEATURE_NAMES)
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(joint[i, k]),
                rtol=1e-6,
                atol=1e-6,
            )
            mech = tr.extra_mechanics
            assert mech["target_class_idx"] == k
            assert mech["joint_log_probs"] == pytest.approx(joint[i].tolist(), rel=1e-6)
            np.testing.assert_allclose(
                np.asarray(mech["probabilities"]),
                np.asarray(model.predict_proba(X[i : i + 1])).ravel(),
                rtol=1e-5,
                atol=1e-9,
            )
            assert np.all(np.isfinite(list(tr.feature_contributions.values())))
            # Z-scores are finite distances in sigma units.
            assert all(v >= 0.0 for v in mech["winning_class_z_scores"])


def test_binary_log_odds_ratio_identity(binary_data) -> None:
    X, y = binary_data
    model = GaussianNB().fit(X, y)
    adapter = GaussianNBAdapter()
    joint = np.asarray(model._joint_log_likelihood(X[:20]), dtype=np.float64)
    traces = adapter.trace_internal_flow(model, X[:20], list(range(20)), FEATURE_NAMES)
    for i, tr in enumerate(traces):
        expected_llr = float(joint[i, 1] - joint[i, 0])
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            expected_llr,
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr.base_value == pytest.approx(
            float(np.log(model.class_prior_[1] / model.class_prior_[0]))
        )


def test_sigma_legacy_and_custom_priors(binary_data) -> None:
    X, y = binary_data
    model = GaussianNB(priors=[0.2, 0.8]).fit(X, y)
    # Simulate a pre-1.0 pickle carrying ``sigma_`` instead of ``var_``.
    model.sigma_ = model.var_  # type: ignore[attr-defined]
    delattr(model, "var_")
    assert not hasattr(model, "var_")
    assert hasattr(model, "sigma_")
    adapter = GaussianNBAdapter()
    traces = adapter.trace_internal_flow(model, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert traces[0].extra_mechanics["class_priors"] == pytest.approx([0.2, 0.8])
    # sklearn's own _joint_log_likelihood needs ``var_`` too — compute the
    # expected joint log-likelihood manually from the legacy ``sigma_``.
    theta = np.asarray(model.theta_, dtype=np.float64)
    var = np.maximum(np.asarray(model.sigma_, dtype=np.float64), 1e-12)
    log_prior = np.log(np.asarray([0.2, 0.8], dtype=np.float64))
    diff = (X[:3].astype(np.float64)[:, None, :] - theta[None, :, :]) ** 2
    joint = log_prior[None, :] - 0.5 * (
        np.sum(np.log(2.0 * np.pi * var[None, :, :]), axis=2) + np.sum(diff / var[None, :, :], axis=2)
    )
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )


def test_outlier_flag_and_pipeline(binary_data, multi_data) -> None:
    Xb, yb = binary_data
    model = GaussianNB().fit(Xb, yb)
    adapter = GaussianNBAdapter()
    far = np.full((2, 4), 100.0, dtype=np.float32)
    traces = adapter.trace_internal_flow(model, far, [0, 1], FEATURE_NAMES)
    for tr in traces:
        assert tr.extra_mechanics["is_low_density_outlier"] is True
        assert tr.extra_mechanics["min_z_score"] > 4.0
    near = adapter.trace_internal_flow(model, Xb[:3], [0, 1, 2], FEATURE_NAMES)
    assert all(t.extra_mechanics["is_low_density_outlier"] is False for t in near)

    Xm, ym = multi_data
    pipe = Pipeline([("scaler", StandardScaler()), ("gnb", GaussianNB())]).fit(Xm, ym)
    traces_pipe = adapter.trace_internal_flow(pipe, Xm[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    preds = np.asarray(pipe.predict(Xm[:5])).ravel()
    for i, tr in enumerate(traces_pipe):
        assert tr.final_prediction == preds[i]

    assert all(v is None for v in adapter.extract_feature_weights(model, 4))


def test_surface_hills_parity_and_truncation(binary_data, multi_data) -> None:
    Xb, yb = binary_data
    clf = GaussianNB().fit(Xb, yb)
    adapter = GaussianNBAdapter()
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
    assert set(mesh.extra_overlays["bell_hills_per_class"]) == {"0", "1"}
    assert mesh.extra_overlays["boundary_threshold"] == 0.5
    assert mesh.extra_overlays["hills_truncated_to_top5"] is False
    assert np.all(np.isfinite(np.asarray(mesh.z_values)))

    spec = GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=1,
        x_min=-3.0,
        x_max=3.0,
        y_min=-3.0,
        y_max=3.0,
        resolution=40,
        fixed_feature_medians={2: 0.0, 3: 0.0},
    )
    mesh2 = adapter.compute_3d_surface(clf, spec, n_features=4)
    theta, var = clf.theta_, np.maximum(clf.var_, 1e-12)
    xs = np.linspace(-3.0, 3.0, 40)
    ys = np.linspace(-3.0, 3.0, 40)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zx = ((xx - theta[1, 0]) ** 2) / (2.0 * var[1, 0])
    zy = ((yy - theta[1, 1]) ** 2) / (2.0 * var[1, 1])
    expected = clf.class_prior_[1] * np.exp(-(zx + zy))
    np.testing.assert_allclose(np.asarray(mesh2.z_values), expected.ravel(), rtol=1e-5)

    # Ten classes -> hills truncated to the Top-5 priors.
    rng = np.random.default_rng(3)
    X10 = rng.normal(size=(300, 4)).astype(np.float32)
    y10 = rng.integers(0, 10, size=300).astype(np.int32)
    many = GaussianNB().fit(X10, y10)
    mesh_many = adapter.compute_3d_surface(many, spec, n_features=4)
    assert mesh_many.extra_overlays["hills_truncated_to_top5"] is True
    assert len(mesh_many.extra_overlays["bell_hills_per_class"]) == 5

    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    traces = adapter.trace_internal_flow(restored, Xb[:2], [0, 1], FEATURE_NAMES)
    assert len(traces) == 2

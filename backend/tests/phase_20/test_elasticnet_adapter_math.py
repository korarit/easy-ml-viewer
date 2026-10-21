"""Phase 20 — ElasticNet / MultiTaskElasticNet adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.linear.elasticnet import (
    ZERO_TOL,
    ElasticNetAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURE_NAMES = ["feature_rm", "feature_lstat", "feature_tax", "feature_noise"]


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=120, n_features=4, n_informative=3, noise=5.0, random_state=42)
    return X.astype(np.float32), y.astype(np.float64)


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


def test_registry_resolves_all_four_classes(reg_data) -> None:
    X, y = reg_data
    Y = np.column_stack([y, y * 0.5 + 3.0])
    cases = [
        (ElasticNet(alpha=0.5, l1_ratio=0.7).fit(X, y)),
        (ElasticNetCV(l1_ratio=[0.2, 0.7], cv=2).fit(X, y)),
        (MultiTaskElasticNet(alpha=0.5, l1_ratio=0.6).fit(X, Y)),
        (MultiTaskElasticNetCV(l1_ratio=[0.3, 0.6], cv=2).fit(X, Y)),
    ]
    for model in cases:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "elastic_net"
        assert adapter.visual_3d_type == Visual3DType.FLAT_HYPERPLANE


def test_single_task_parity_and_lambda_identity(reg_data) -> None:
    X, y = reg_data
    model = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000).fit(X, y)
    adapter = ElasticNetAdapter()
    batch = adapter.predict_batch(model, X[:25])
    np.testing.assert_allclose(np.asarray(batch.predictions), model.predict(X[:25]), atol=1e-4)
    traces = adapter.trace_internal_flow(model, X[:25], list(range(25)), FEATURE_NAMES)
    assert len(traces) == 25
    preds = np.asarray(model.predict(X[:25])).ravel()
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds[i], atol=1e-4)
        mech = tr.extra_mechanics
        assert mech["l1_ratio"] == pytest.approx(0.7)
        assert mech["lambda_1"] == pytest.approx(0.5 * 0.7)
        assert mech["lambda_2"] == pytest.approx(0.5 * 0.5 * 0.3)
        assert mech["is_multitask"] is False
        assert mech["n_tasks"] == 1
        assert mech["convergence_warning"] is False
        assert len(mech["penalty_shares"]) == 4
        # Burden identity: lambda-weighted L1/L2 terms.
        w = np.asarray(model.coef_, dtype=np.float64)
        assert mech["l1_penalty_burden"] == pytest.approx(float(0.5 * 0.7 * np.sum(np.abs(w))), rel=1e-4)
        assert mech["l2_penalty_burden"] == pytest.approx(float(0.5 * 0.5 * 0.3 * np.sum(w**2)), rel=1e-4)


def test_cv_l1_ratio_list_resolves_winner(reg_data) -> None:
    X, y = reg_data
    model = ElasticNetCV(l1_ratio=[0.2, 0.7], cv=2, max_iter=5000).fit(X, y)
    adapter = ElasticNetAdapter()
    traces = adapter.trace_internal_flow(model, X[:3], [0, 1, 2], FEATURE_NAMES)
    winner = float(np.asarray(model.l1_ratio_).flat[0])
    assert traces[0].extra_mechanics["l1_ratio"] == pytest.approx(winner)
    assert np.isfinite(traces[0].extra_mechanics["alpha"])


def test_multitask_joint_selection_and_task_switch(reg_data) -> None:
    X, y = reg_data
    Y = np.column_stack([y, y * 0.5 + 3.0])
    model = MultiTaskElasticNet(alpha=2.0, l1_ratio=0.8, max_iter=5000).fit(X, Y)
    adapter = ElasticNetAdapter()
    assert np.asarray(model.coef_).shape == (2, 4)
    for task in (0, 1):
        traces = adapter.trace_internal_flow(model, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES, target_idx=task)
        preds = np.asarray(model.predict(X[:5]))[:, task]
        for i, tr in enumerate(traces):
            assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds[i], atol=1e-3)
            assert tr.extra_mechanics["active_task_idx"] == task
            assert tr.extra_mechanics["n_tasks"] == 2
            assert tr.extra_mechanics["is_multitask"] is True
    # target_class_idx alias selects the task as well.
    alt = adapter.trace_internal_flow(model, X[:2], [0, 1], FEATURE_NAMES, target_class_idx=1)
    assert alt[0].extra_mechanics["active_task_idx"] == 1


def test_multitask_cv_and_extreme_l1_ratio(reg_data) -> None:
    X, y = reg_data
    Y = np.column_stack([y, y * 0.5 + 3.0])
    model = MultiTaskElasticNetCV(l1_ratio=[0.3, 0.6], cv=2, max_iter=5000).fit(X, Y)
    adapter = ElasticNetAdapter()
    traces = adapter.trace_internal_flow(model, X[:2], [0, 1], FEATURE_NAMES)
    assert traces[0].extra_mechanics["n_tasks"] == 2

    # rho == 1.0 edge: lambda_2 collapses to 0, surface must stay finite.
    sharp = ElasticNet(alpha=0.5, l1_ratio=1.0, max_iter=5000).fit(X, y)
    traces_sharp = adapter.trace_internal_flow(sharp, X[:2], [0, 1], FEATURE_NAMES)
    assert traces_sharp[0].extra_mechanics["lambda_2"] == pytest.approx(0.0)
    mesh = adapter.compute_3d_surface(sharp, _spec(), n_features=4)
    assert np.all(np.isfinite(np.asarray(mesh.z_values)))
    assert mesh.extra_overlays["l1_ratio"] == pytest.approx(1.0)

    # rho == 0.0 edge: pure-Ridge mix, lambda_1 collapses to 0.
    smooth = ElasticNet(alpha=0.5, l1_ratio=0.0, max_iter=5000).fit(X, y)
    traces_smooth = adapter.trace_internal_flow(smooth, X[:2], [0, 1], FEATURE_NAMES)
    assert traces_smooth[0].extra_mechanics["lambda_1"] == pytest.approx(0.0)


def test_convergence_warning_and_positive_flag(reg_data) -> None:
    X, y = reg_data
    adapter = ElasticNetAdapter()
    # Force non-convergence with a single coordinate-descent sweep.
    stuck = ElasticNet(alpha=1e-6, l1_ratio=0.5, max_iter=1, tol=1e-12).fit(X, y)
    traces = adapter.trace_internal_flow(stuck, X[:2], [0, 1], FEATURE_NAMES)
    n_iter = int(np.asarray(stuck.n_iter_).max())
    assert traces[0].extra_mechanics["n_iter"] == n_iter
    assert traces[0].extra_mechanics["max_iter"] == 1
    if n_iter >= 1:
        assert traces[0].extra_mechanics["convergence_warning"] is True

    pos = ElasticNet(alpha=0.5, l1_ratio=0.5, positive=True, max_iter=5000).fit(
        np.abs(X), np.abs(y)
    )
    traces_pos = adapter.trace_internal_flow(pos, np.abs(X[:2]), [0, 1], FEATURE_NAMES)
    assert traces_pos[0].extra_mechanics["positive_only"] is True
    assert np.all(np.asarray(pos.coef_) >= 0.0)
    mesh = adapter.compute_3d_surface(pos, _spec(), n_features=4)
    assert mesh.extra_overlays["positive_only"] is True


def test_pipeline_sparse_and_all_pruned_banner(reg_data) -> None:
    X, y = reg_data
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet(alpha=0.5, l1_ratio=0.5))]).fit(X, y)
    adapter = ElasticNetAdapter()
    traces = adapter.trace_internal_flow(pipe, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    preds = np.asarray(pipe.predict(X[:5])).ravel()
    for i, tr in enumerate(traces):
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds[i], atol=1e-3)
    mesh = adapter.compute_3d_surface(pipe, _spec(), n_features=4)
    assert mesh.extra_overlays["gpu_formula_valid"] is False

    try:
        from scipy import sparse  # noqa: PLC0415
    except ImportError:
        pytest.skip("scipy not available")
        return
    X_sparse = sparse.csr_matrix(X[:10])
    plain = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(X, y)
    traces_sp = adapter.trace_internal_flow(plain, X_sparse, list(range(5)), FEATURE_NAMES)
    assert len(traces_sp) == 5
    preds_sp = np.asarray(plain.predict(X[:5])).ravel()
    for i, tr in enumerate(traces_sp):
        assert np.isclose(tr.base_value + sum(tr.feature_contributions.values()), preds_sp[i], atol=1e-4)

    over = ElasticNet(alpha=1e6, l1_ratio=0.9, max_iter=10000).fit(X, y)
    traces_all = adapter.trace_internal_flow(over, X[:2], [0, 1], FEATURE_NAMES)
    assert traces_all[0].extra_mechanics["active_weight_count"] == 0
    mesh_all = adapter.compute_3d_surface(over, _spec(), n_features=4)
    assert mesh_all.extra_overlays["zero_influence_axes"] is True


def test_surface_analytical_plane_and_resolution_clamp(reg_data) -> None:
    X, y = reg_data
    model = ElasticNet(alpha=0.5, l1_ratio=0.6, max_iter=5000).fit(X, y)
    adapter = ElasticNetAdapter()
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

    spec = _spec()
    mesh2 = adapter.compute_3d_surface(model, spec, n_features=4)
    w = np.asarray(model.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(model.intercept_).ravel()[0])
    off = b + float(w[2]) * 0.5 + float(w[3]) * -0.25
    xs = np.linspace(-3.0, 3.0, 40)
    ys = np.linspace(-3.0, 3.0, 40)
    assert np.isclose(mesh2.z_values[0], w[0] * xs[0] + w[1] * ys[0] + off, atol=1e-4)
    assert np.isclose(mesh2.extra_overlays["intercept_offset"], off, atol=1e-6)
    # Sparsity flags reuse the Phase-19 ZERO_TOL convention.
    assert isinstance(mesh2.extra_overlays["is_x_pruned"], bool)
    assert mesh2.extra_overlays["is_x_pruned"] == bool(abs(float(w[0])) <= ZERO_TOL)

    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    traces = adapter.trace_internal_flow(restored, X[:2], [0, 1], FEATURE_NAMES)
    assert len(traces) == 2

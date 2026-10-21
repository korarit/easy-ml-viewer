"""Phase 23 — SVM kernel-saddle adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons, make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import (
    SVC,
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
    SVR,
)

from app.domains.model_engine.adapters.distance.svm import (
    MAX_DISPLAY_SV,
    OneClassSvmAdapter,
    SvmClassifierAdapter,
    SvmRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURE_NAMES = ["feature_a", "feature_b", "feature_c", "feature_d"]


@pytest.fixture()
def clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=2, n_redundant=1, random_state=7
    )
    return StandardScaler().fit_transform(X).astype(np.float32), y


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=120, n_features=4, n_informative=3, noise=5.0, random_state=7)
    return StandardScaler().fit_transform(X).astype(np.float32), y.astype(np.float64)


def _spec(resolution: int = 40) -> GridSpec3D:
    return GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=1,
        x_min=-3.0,
        x_max=3.0,
        y_min=-3.0,
        y_max=3.0,
        resolution=resolution,
        fixed_feature_medians={2: 0.5, 3: -0.25},
    )


def test_registry_resolves_svm_family(clf_data, reg_data) -> None:
    Xc, yc = clf_data
    Xr, yr = reg_data
    for model in [
        SVC().fit(Xc, yc),
        NuSVC().fit(Xc, yc),
        LinearSVC().fit(Xc, yc),
    ]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "svm_classifier"
        assert adapter.visual_3d_type == Visual3DType.SVM_MARGIN_TUBE
    for model in [SVR().fit(Xr, yr), NuSVR().fit(Xr, yr), LinearSVR().fit(Xr, yr)]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "svm_regressor"
    oc = OneClassSVM().fit(Xc)
    adapter = ModelRegistry.resolve_adapter(oc)
    assert adapter.algorithm_key == "svm_oneclass"


@pytest.mark.parametrize("kernel", ["rbf", "poly", "sigmoid", "linear"])
def test_classifier_dual_expansion_identity(clf_data, kernel: str) -> None:
    X, y = clf_data
    kwargs: dict = {"kernel": kernel}
    if kernel == "poly":
        kwargs.update(degree=2)
    model = SVC(**kwargs).fit(X, y)
    adapter = SvmClassifierAdapter()
    batch = adapter.predict_batch(model, X[:25])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:25]))
    traces = adapter.trace_internal_flow(model, X[:25], list(range(25)), FEATURE_NAMES)
    scores = np.asarray(model.decision_function(X[:25])).ravel()
    assert len(traces) == 25
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        mech = tr.extra_mechanics
        assert mech["kernel"] == kernel
        assert mech["raw_score"] == pytest.approx(float(scores[i]), rel=1e-6)
        # Kernel expansion identity: sum(forces) + b == decision_function.
        np.testing.assert_allclose(
            mech["kernel_force_sum"] + tr.base_value, float(scores[i]), rtol=1e-6, atol=1e-6
        )
        assert len(mech["kernel_forces_top"]) == min(5, mech["total_support_vectors"])
        assert np.all(np.isfinite(list(tr.feature_contributions.values())))
        # Support display payload is capped.
        assert mech["support_vectors_display"]["displayed_count"] <= MAX_DISPLAY_SV


def test_linear_waterfall_and_no_support_vectors(clf_data) -> None:
    X, y = clf_data
    model = LinearSVC(max_iter=5000).fit(X, y)
    adapter = SvmClassifierAdapter()
    traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), FEATURE_NAMES)
    scores = np.asarray(model.decision_function(X[:10])).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), float(scores[i]), rtol=1e-4, atol=1e-2
        )
        assert tr.extra_mechanics["is_linear_model"] is True
        assert tr.extra_mechanics["support_vectors_display"]["total_support_vectors"] == 0
    # Linear models expose real feature weights (no None holes).
    assert all(v is not None for v in adapter.extract_feature_weights(model, 4))
    assert all(v is None for v in SvmClassifierAdapter().extract_feature_weights(SVC().fit(X, y), 4))


def test_probability_false_pseudo_and_true_native(clf_data) -> None:
    X, y = clf_data
    plain = SVC(probability=False).fit(X, y)
    traces = SvmClassifierAdapter().trace_internal_flow(plain, X[:3], [0, 1, 2], FEATURE_NAMES)
    for tr in traces:
        assert tr.extra_mechanics["has_native_proba"] is False
        assert tr.extra_mechanics["is_pseudo_proba"] is True
        proba = np.asarray(tr.extra_mechanics["probabilities"])
        assert proba.shape == (2,)
        assert np.isclose(proba.sum(), 1.0)

    cal = SVC(probability=True, random_state=0).fit(X, y)
    traces_c = SvmClassifierAdapter().trace_internal_flow(cal, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    native = np.asarray(cal.predict_proba(X[:5]))
    for i, tr in enumerate(traces_c):
        assert tr.extra_mechanics["has_native_proba"] is True
        np.testing.assert_allclose(np.asarray(tr.extra_mechanics["probabilities"]), native[i], rtol=1e-5)


def test_multiclass_ovr_scores_and_ovo_flag(clf_data) -> None:
    X, _ = clf_data
    rng = np.random.default_rng(5)
    y3 = rng.integers(0, 3, size=X.shape[0]).astype(np.int32)
    ovr = SVC(decision_function_shape="ovr", random_state=0).fit(X, y3)
    traces = SvmClassifierAdapter().trace_internal_flow(
        ovr, X[:4], [0, 1, 2, 3], FEATURE_NAMES, target_class_idx=2
    )
    assert traces[0].extra_mechanics["is_ovo"] is False
    assert traces[0].extra_mechanics["target_class_idx"] == 2
    assert len(traces[0].extra_mechanics["decision_scores"]) == 3

    ovo = SVC(decision_function_shape="ovo", random_state=0).fit(X, y3)
    traces_o = SvmClassifierAdapter().trace_internal_flow(ovo, X[:2], [0, 1], FEATURE_NAMES)
    assert traces_o[0].extra_mechanics["is_ovo"] is True


def test_nusvc_and_pipeline_sparse(clf_data) -> None:
    X, y = clf_data
    adapter = SvmClassifierAdapter()
    nu = NuSVC().fit(X, y)
    traces = adapter.trace_internal_flow(nu, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert traces[0].extra_mechanics["nu"] is not None
    scores = np.asarray(nu.decision_function(X[:3])).ravel()
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.extra_mechanics["kernel_force_sum"] + tr.base_value, float(scores[i]), rtol=1e-6, atol=1e-6
        )

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))]
    ).fit(X, y)
    traces_pipe = adapter.trace_internal_flow(pipe, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    preds = np.asarray(pipe.predict(X[:5])).ravel()
    for i, tr in enumerate(traces_pipe):
        assert tr.final_prediction == preds[i]

    try:
        from scipy import sparse  # noqa: PLC0415
    except ImportError:
        pytest.skip("scipy not available")
        return
    X_sparse = sparse.csr_matrix(X[:10])
    plain = SVC(kernel="linear").fit(X, y)
    traces_sp = adapter.trace_internal_flow(plain, X_sparse, list(range(5)), FEATURE_NAMES)
    assert len(traces_sp) == 5


def test_regressor_expansion_and_epsilon_tube(reg_data) -> None:
    X, y = reg_data
    for kernel in ("rbf", "linear"):
        model = SVR(kernel=kernel).fit(X, y)
        adapter = SvmRegressorAdapter()
        batch = adapter.predict_batch(model, X[:10])
        np.testing.assert_allclose(
            np.asarray(batch.predictions), np.asarray(model.predict(X[:10])), rtol=1e-6
        )
        traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), FEATURE_NAMES)
        preds = np.asarray(model.predict(X[:10])).ravel()
        for i, tr in enumerate(traces):
            assert tr.final_prediction == pytest.approx(float(preds[i]), rel=1e-6)
            np.testing.assert_allclose(
                tr.extra_mechanics["kernel_force_sum"] + tr.base_value,
                float(preds[i]),
                rtol=1e-6,
                atol=1e-6,
            )
            assert tr.extra_mechanics["margin_width"] == pytest.approx(
                tr.extra_mechanics["epsilon"]
            )
    lin = LinearSVR().fit(X, y)
    traces_l = SvmRegressorAdapter().trace_internal_flow(lin, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    preds_l = np.asarray(lin.predict(X[:5])).ravel()
    for i, tr in enumerate(traces_l):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()), float(preds_l[i]), rtol=1e-4, atol=1e-2
        )


def test_oneclass_outlier_envelope(clf_data) -> None:
    X, _ = clf_data
    model = OneClassSVM(kernel="rbf", nu=0.1).fit(X)
    adapter = OneClassSvmAdapter()
    traces = adapter.trace_internal_flow(model, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    scores = np.asarray(model.decision_function(X[:5])).ravel()
    preds = np.asarray(model.predict(X[:5])).ravel()
    for i, tr in enumerate(traces):
        assert tr.extra_mechanics["raw_score"] == pytest.approx(float(scores[i]), rel=1e-6)
        np.testing.assert_allclose(
            tr.extra_mechanics["kernel_force_sum"] + tr.base_value, float(scores[i]), rtol=1e-6, atol=1e-6
        )
        assert tr.final_prediction == int(preds[i])
        assert tr.extra_mechanics["predicted_label"] in ("INLIER", "OUTLIER")
    far = np.full((2, 4), 100.0, dtype=np.float32)
    far_traces = adapter.trace_internal_flow(model, far, [0, 1], FEATURE_NAMES)
    assert all(t.extra_mechanics["predicted_label"] == "OUTLIER" for t in far_traces)


def test_support_cap_and_surface_parity(clf_data, reg_data) -> None:
    Xc, yc = clf_data
    rng = np.random.default_rng(9)
    big_X = np.vstack([Xc, rng.normal(size=(2500, 4)).astype(np.float32)])
    big_y = np.concatenate([yc, rng.integers(0, 2, size=2500)])
    noisy = SVC(kernel="rbf", C=100.0).fit(big_X, big_y)
    traces = SvmClassifierAdapter().trace_internal_flow(noisy, big_X[:2], [0, 1], FEATURE_NAMES)
    for tr in traces:
        assert tr.extra_mechanics["support_vectors_display"]["displayed_count"] <= MAX_DISPLAY_SV

    clf = SVC(kernel="rbf").fit(Xc, yc)
    adapter = SvmClassifierAdapter()
    big = _spec(resolution=200)
    mesh = adapter.compute_3d_surface(clf, big, n_features=4)
    assert mesh.resolution == 100
    assert len(mesh.z_values) == 100 * 100
    assert mesh.extra_overlays["margin_levels"] == [-1.0, 0.0, 1.0]
    assert mesh.extra_overlays["boundary_threshold"] == 0.0
    assert np.all(np.isfinite(np.asarray(mesh.z_values)))

    spec = _spec()
    mesh2 = adapter.compute_3d_surface(clf, spec, n_features=4)
    grid = np.zeros((1600, 4), dtype=np.float32)
    xs = np.linspace(-3.0, 3.0, 40, dtype=np.float32)
    ys = np.linspace(-3.0, 3.0, 40, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()
    grid[:, 2] = 0.5
    grid[:, 3] = -0.25
    expected = np.asarray(clf.decision_function(grid)).ravel()
    np.testing.assert_allclose(np.asarray(mesh2.z_values), expected, rtol=1e-5)

    Xr, yr = reg_data
    reg = SVR(kernel="rbf").fit(Xr, yr)
    mesh_r = SvmRegressorAdapter().compute_3d_surface(reg, spec, n_features=4)
    assert mesh_r.extra_overlays["boundary_threshold"] is None
    expected_r = np.asarray(reg.predict(grid)).ravel()
    np.testing.assert_allclose(np.asarray(mesh_r.z_values), expected_r, rtol=1e-5)

    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    traces2 = adapter.trace_internal_flow(restored, Xc[:2], [0, 1], FEATURE_NAMES)
    assert len(traces2) == 2

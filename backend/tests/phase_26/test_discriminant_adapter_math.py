"""Phase 26 — LDA/QDA discriminant adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.probabilistic.discriminant import (
    DiscriminantAnalysisAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def blobs3() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def blobs2() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(26)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


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


def test_registry_resolves_both_discriminants(blobs3) -> None:
    X, y = blobs3
    for model in [LinearDiscriminantAnalysis().fit(X, y), QuadraticDiscriminantAnalysis().fit(X, y)]:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "discriminant"
        assert adapter.visual_3d_type == Visual3DType.DISCRIMINANT_ELLIPSOIDS


def test_lda_binary_coef_identity(blobs2) -> None:
    X, y = blobs2
    names = ["feat_x", "feat_y"]
    for kwargs in [{}, {"solver": "lsqr", "store_covariance": True}, {"solver": "eigen"}]:
        model = LinearDiscriminantAnalysis(**kwargs).fit(X, y)
        adapter = DiscriminantAnalysisAdapter()
        batch = adapter.predict_batch(model, X[:10])
        np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:10]))
        np.testing.assert_allclose(
            np.asarray(batch.probabilities), np.asarray(model.predict_proba(X[:10])), rtol=1e-5
        )
        dec = np.asarray(model.decision_function(X[:10]), dtype=np.float64).ravel()
        traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), names)
        for i, tr in enumerate(traces):
            assert set(tr.feature_contributions) == set(names)
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(dec[i]),
                rtol=1e-5,
                atol=1e-5,
            )
            mech = tr.extra_mechanics
            assert mech["discriminant"] == "lda"
            np.testing.assert_allclose(
                np.asarray(mech["probabilities"]),
                np.asarray(model.predict_proba(X[i : i + 1])).ravel(),
                rtol=1e-5,
                atol=1e-9,
            )
            assert np.all(np.isfinite(list(tr.feature_contributions.values())))
            assert len(mech["mahalanobis_distances"]) == 2
            assert len(mech["euclidean_distances"]) == 2


def test_lda_multiclass_target_switch(blobs3) -> None:
    X, y = blobs3
    model = LinearDiscriminantAnalysis().fit(X, y)
    adapter = DiscriminantAnalysisAdapter()
    dec = np.asarray(model.decision_function(X[:5]), dtype=np.float64)
    for k in range(3):
        traces = adapter.trace_internal_flow(model, X[:5], list(range(5)), FEATURES_4,
                                             target_class_idx=k)
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(dec[i, k]),
                rtol=1e-5,
                atol=1e-5,
            )
            assert tr.extra_mechanics["resolved_class_idx"] == k
    # Default store_covariance=False -> approximated path is flagged.
    assert traces[0].extra_mechanics["is_covariance_approximated"] is True
    assert traces[0].extra_mechanics["store_covariance"] is False


def test_lda_store_covariance_exact_path(blobs3) -> None:
    X, y = blobs3
    model = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True).fit(X, y)
    adapter = DiscriminantAnalysisAdapter()
    traces = adapter.trace_internal_flow(model, X[:3], [0, 1, 2], FEATURES_4)
    assert traces[0].extra_mechanics["is_covariance_approximated"] is False
    assert traces[0].extra_mechanics["store_covariance"] is True
    assert traces[0].extra_mechanics["solver"] == "lsqr"
    weights = adapter.extract_feature_weights(model, 4)
    assert all(v is not None and v >= 0.0 for v in weights)


def test_qda_waterfall_matches_scores(blobs3, blobs2) -> None:
    X, y = blobs3
    model = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)
    adapter = DiscriminantAnalysisAdapter()
    dec = np.asarray(model.decision_function(X[:5]), dtype=np.float64)
    for k in range(3):
        traces = adapter.trace_internal_flow(model, X[:5], list(range(5)), FEATURES_4,
                                             target_class_idx=k)
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(dec[i, k]),
                rtol=1e-5,
                atol=1e-5,
            )
            assert tr.extra_mechanics["discriminant"] == "qda"
    # Binary: log-odds-ratio identity.
    Xb, yb = blobs2
    qb = QuadraticDiscriminantAnalysis().fit(Xb, yb)
    dec_b = np.asarray(qb.decision_function(Xb[:5]), dtype=np.float64).ravel()
    traces_b = adapter.trace_internal_flow(qb, Xb[:5], list(range(5)), ["feat_x", "feat_y"])
    for i, tr in enumerate(traces_b):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(dec_b[i]),
            rtol=1e-5,
            atol=1e-5,
        )
    # QDA has no linear weights.
    assert all(v is None for v in adapter.extract_feature_weights(qb, 2))


def test_qda_covariance_rebuild_matches_store(blobs3) -> None:
    X, y = blobs3
    from app.domains.model_engine.adapters.probabilistic.discriminant import (
        _final_estimator,
        _resolve_covariances,
    )

    approx = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)
    exact = QuadraticDiscriminantAnalysis(reg_param=0.01, store_covariance=True).fit(X, y)
    covs_approx, flag_a = _resolve_covariances(_final_estimator(approx), True, 3, 4)
    covs_exact, flag_e = _resolve_covariances(_final_estimator(exact), True, 3, 4)
    assert flag_a is False and flag_e is False
    for ca, ce in zip(covs_approx, covs_exact):
        # rotations_/scalings_ are stored in float32 precision by sklearn,
        # so the rebuild matches up to float32 rounding (~1e-7).
        np.testing.assert_allclose(ca, ce, rtol=1e-5, atol=1e-7)
    # Trace parity holds on the rebuilt path too.
    adapter = DiscriminantAnalysisAdapter()
    traces = adapter.trace_internal_flow(approx, X[:3], [0, 1, 2], FEATURES_4)
    dec = np.asarray(approx.decision_function(X[:3]), dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(dec[i, 0]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_paradox_flag_and_singular_covariance(blobs2) -> None:
    X, y = blobs2
    model = LinearDiscriminantAnalysis().fit(X, y)
    adapter = DiscriminantAnalysisAdapter()
    traces = adapter.trace_internal_flow(model, X[:40], list(range(40)), ["feat_x", "feat_y"])
    flags = [t.extra_mechanics["paradox_detected"] for t in traces]
    assert any(isinstance(f, bool) for f in flags)
    for tr in traces:
        m, e = tr.extra_mechanics["mahalanobis_distances"], tr.extra_mechanics["euclidean_distances"]
        assert np.all(np.isfinite(m + e))
        assert all(v >= 0.0 for v in m + e)
        assert tr.extra_mechanics["nearest_mahalanobis_class"] in (0, 1)
    # Collinear features + tiny class: pinv path must not raise.
    rng = np.random.default_rng(3)
    Xc = np.stack([rng.normal(size=30), rng.normal(size=30)], axis=1).astype(np.float32)
    Xc = np.hstack([Xc, (2 * Xc[:, [0]] + 1).astype(np.float32)])  # exact collinearity
    yc = np.array([0] * 15 + [1] * 15, dtype=np.int32)
    qcol = QuadraticDiscriminantAnalysis(reg_param=0.05).fit(Xc, yc)
    tr_col = adapter.trace_internal_flow(qcol, Xc[:3], [0, 1, 2], ["a", "b", "c"])
    for tr in tr_col:
        assert np.all(np.isfinite(list(tr.feature_contributions.values()) + [tr.base_value]))


def test_pipeline_and_roundtrip(blobs2) -> None:
    X, y = blobs2
    pipe = Pipeline([("scaler", StandardScaler()), ("lda", LinearDiscriminantAnalysis())]).fit(X, y)
    adapter = DiscriminantAnalysisAdapter()
    traces = adapter.trace_internal_flow(pipe, X[:3], [0, 1, 2], ["feat_x", "feat_y"])
    assert [t.final_prediction for t in traces] == list(pipe.predict(X[:3]))
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    buf.seek(0)
    assert len(adapter.trace_internal_flow(joblib.load(buf), X[:2], [0, 1], ["feat_x", "feat_y"])) == 2


def test_surface_ellipsoids_parity_and_det(blobs3, blobs2) -> None:
    X, y = blobs3
    adapter = DiscriminantAnalysisAdapter()
    qda = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)
    mesh = adapter.compute_3d_surface(qda, _spec(), n_features=4)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    assert mesh.extra_overlays["discriminant"] == "qda"
    assert mesh.extra_overlays["boundary_threshold"] == 0.5
    assert np.all(np.isfinite(np.asarray(mesh.z_values)))
    ells = mesh.extra_overlays["ellipsoids_2d"]
    assert len(ells) == 3
    for spec in ells:
        assert spec["radii_1sigma"][0] > 0 and spec["radii_1sigma"][1] > 0
        det = float(np.linalg.det(np.asarray(spec["eigenvectors_2d"])))
        assert det == pytest.approx(1.0, abs=1e-9)
        assert np.isfinite(spec["rotation_angle_rad"])
    assert len(mesh.extra_overlays["means_2d"]) == 3
    # Per-class tilts differ for QDA on distinct covariances.
    angles = [e["rotation_angle_rad"] for e in ells]
    assert max(angles) - min(angles) > 1e-3

    # LDA shares one orientation across classes.
    lda = LinearDiscriminantAnalysis().fit(X, y)
    mesh_l = adapter.compute_3d_surface(lda, _spec(), n_features=4)
    angs_l = [e["rotation_angle_rad"] for e in mesh_l.extra_overlays["ellipsoids_2d"]]
    assert max(angs_l) - min(angs_l) < 1e-6

    big = _spec(resolution=200)
    assert adapter.compute_3d_surface(qda, big, n_features=4).resolution == 100

    # Binary surface renders the positive class.
    Xb, yb = blobs2
    lb = LinearDiscriminantAnalysis().fit(Xb, yb)
    mesh_b = adapter.compute_3d_surface(lb, _spec(x_min=-6.0, x_max=8.0), n_features=2)
    assert mesh_b.extra_overlays["is_binary"] is True
    assert np.all(np.isfinite(np.asarray(mesh_b.z_values)))

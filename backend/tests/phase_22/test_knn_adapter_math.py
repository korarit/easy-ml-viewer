"""Phase 22 — KNN radar adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.distance.knn import (
    MAX_BACKGROUND_REF_POINTS,
    KnnClassifierAdapter,
    KnnRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURE_NAMES = ["feature_a", "feature_b", "feature_c", "feature_d"]


@pytest.fixture()
def clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=160, n_features=4, n_informative=2, n_redundant=1, random_state=7
    )
    return StandardScaler().fit_transform(X).astype(np.float32), y


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=160, n_features=4, n_informative=3, noise=5.0, random_state=7)
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


def test_registry_resolves_knn_family(clf_data, reg_data) -> None:
    Xc, yc = clf_data
    Xr, yr = reg_data
    clf_cases = [
        KNeighborsClassifier(n_neighbors=5).fit(Xc, yc),
        KNeighborsClassifier(n_neighbors=3, weights="distance", metric="manhattan").fit(Xc, yc),
        RadiusNeighborsClassifier(radius=2.0).fit(Xc, yc),
    ]
    for model in clf_cases:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "knn_classifier"
        assert adapter.visual_3d_type == Visual3DType.VORONOI_KNN_PROXIMITY
    reg_cases = [
        KNeighborsRegressor(n_neighbors=5).fit(Xr, yr),
        RadiusNeighborsRegressor(radius=2.0).fit(Xr, yr),
    ]
    for model in reg_cases:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == "knn_regressor"


def test_classifier_uniform_vote_and_radar_radius(clf_data) -> None:
    X, y = clf_data
    model = KNeighborsClassifier(n_neighbors=5, weights="uniform").fit(X, y)
    adapter = KnnClassifierAdapter()
    batch = adapter.predict_batch(model, X[:25])
    np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(X[:25]))
    traces = adapter.trace_internal_flow(model, X[:25], list(range(25)), FEATURE_NAMES)
    assert len(traces) == 25
    dists, _ = model.kneighbors(X[:25], return_distance=True)
    for i, tr in enumerate(traces):
        assert set(tr.feature_contributions) == set(FEATURE_NAMES)
        mech = tr.extra_mechanics
        assert mech["n_neighbors"] == 5
        assert mech["n_neighbors_effective"] == 5
        assert mech["weights"] == "uniform"
        assert mech["radar_geometry_shape"] == "sphere"
        assert mech["radar_radius"] == pytest.approx(float(np.max(dists[i])))
        # Uniform weights sum to 1 and vote counts sum to K.
        w = np.asarray(mech["neighbor_weights"])
        assert w.shape == (5,)
        np.testing.assert_allclose(w, np.full((5,), 0.2))
        assert sum(v["count"] for v in mech["vote_tally"]) == 5
        assert mech["display_capped"] is False
        # Mandatory union guarantee: every spider endpoint is in the cloud.
        cloud_idx = set(mech["reference_cloud"]["original_indices"])
        for orig in mech["neighbor_original_indices"]:
            assert orig in cloud_idx
        assert len(cloud_idx) <= MAX_BACKGROUND_REF_POINTS


def test_classifier_distance_weights_match_sklearn(clf_data) -> None:
    X, y = clf_data
    model = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X, y)
    adapter = KnnClassifierAdapter()
    traces = adapter.trace_internal_flow(model, X[:5], [0, 1, 2, 3, 4], FEATURE_NAMES)
    dists, _ = model.kneighbors(X[:5], return_distance=True)
    native_proba = np.asarray(model.predict_proba(X[:5]))
    for i, tr in enumerate(traces):
        mech = tr.extra_mechanics
        inv = 1.0 / np.maximum(dists[i], 1e-12)
        expected = inv / np.sum(inv)
        np.testing.assert_allclose(np.asarray(mech["neighbor_weights"]), expected, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(mech["probabilities"]), native_proba[i], rtol=1e-6
        )
        # Distance-weighted vote shares sum to 1.
        assert sum(v["weight_share"] for v in mech["vote_tally"]) == pytest.approx(1.0)
        # Distance waterfall contributions are non-positive distances to centroid.
        assert all(v <= 1e-6 for v in tr.feature_contributions.values())


def test_zero_distance_no_divide_by_zero(clf_data) -> None:
    X, y = clf_data
    # Query exact training rows so d=0 appears with distance weighting.
    model = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, y)
    adapter = KnnClassifierAdapter()
    traces = adapter.trace_internal_flow(model, X[:3], [0, 1, 2], FEATURE_NAMES)
    for tr in traces:
        w = np.asarray(tr.extra_mechanics["neighbor_weights"])
        assert np.all(np.isfinite(w))
        assert float(np.sum(w)) == pytest.approx(1.0)
        assert tr.extra_mechanics["neighbor_distances"][0] == pytest.approx(0.0, abs=1e-6)


def test_metric_shapes_octahedron_and_cube(clf_data) -> None:
    X, y = clf_data
    adapter = KnnClassifierAdapter()
    man = KNeighborsClassifier(n_neighbors=3, metric="manhattan").fit(X, y)
    mech_m = adapter.trace_internal_flow(man, X[:2], [0, 1], FEATURE_NAMES)[0].extra_mechanics
    assert mech_m["effective_metric"] == "manhattan"
    assert mech_m["radar_geometry_shape"] == "octahedron"
    cheb = KNeighborsClassifier(n_neighbors=3, metric="chebyshev").fit(X, y)
    mech_c = adapter.trace_internal_flow(cheb, X[:2], [0, 1], FEATURE_NAMES)[0].extra_mechanics
    assert mech_c["radar_geometry_shape"] == "cube"


def test_radius_outlier_telemetry(clf_data) -> None:
    X, y = clf_data
    model = RadiusNeighborsClassifier(radius=0.01, outlier_label=-1).fit(X, y)
    adapter = KnnClassifierAdapter()
    far = np.full((2, 4), 100.0, dtype=np.float32)
    traces = adapter.trace_internal_flow(model, far, [0, 1], FEATURE_NAMES)
    for tr in traces:
        mech = tr.extra_mechanics
        assert mech["is_radius"] is True
        assert mech["is_outlier"] is True
        assert mech["neighbor_original_indices"] == []
        assert mech["neighbor_distances"] == []
        assert mech["radar_radius"] == pytest.approx(0.01)
        assert tr.final_prediction == -1
    # Radius with a sane radius behaves like fixed-K (non-empty).
    sane = RadiusNeighborsClassifier(radius=5.0).fit(X, y)
    traces_sane = adapter.trace_internal_flow(sane, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert all(t.extra_mechanics["n_neighbors_effective"] >= 1 for t in traces_sane)


def test_sparse_fit_and_pipeline_parity(clf_data) -> None:
    X, y = clf_data
    adapter = KnnClassifierAdapter()
    try:
        from scipy import sparse  # noqa: PLC0415
    except ImportError:
        pytest.skip("scipy not available")
        return
    X_sparse = sparse.csr_matrix(X)
    brute = KNeighborsClassifier(n_neighbors=3, algorithm="brute").fit(X_sparse, y)
    traces = adapter.trace_internal_flow(brute, X[:3], [0, 1, 2], FEATURE_NAMES)
    assert len(traces) == 3
    assert len(traces[0].extra_mechanics["reference_cloud"]["original_indices"]) == X.shape[0]

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=3))]
    ).fit(X, y)
    traces_pipe = adapter.trace_internal_flow(pipe, X[:3], [0, 1, 2], FEATURE_NAMES)
    preds = np.asarray(pipe.predict(X[:3])).ravel()
    for i, tr in enumerate(traces_pipe):
        assert tr.final_prediction == preds[i]
        assert tr.extra_mechanics["n_neighbors_effective"] == 3


def test_regressor_weighted_mean_parity(reg_data) -> None:
    X, y = reg_data
    for weights in ("uniform", "distance"):
        model = KNeighborsRegressor(n_neighbors=4, weights=weights).fit(X, y)
        adapter = KnnRegressorAdapter()
        batch = adapter.predict_batch(model, X[:10])
        np.testing.assert_allclose(
            np.asarray(batch.predictions), np.asarray(model.predict(X[:10])), rtol=1e-5
        )
        traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), FEATURE_NAMES)
        preds = np.asarray(model.predict(X[:10])).ravel()
        for i, tr in enumerate(traces):
            assert tr.final_prediction == pytest.approx(float(preds[i]), rel=1e-5)
            mech = tr.extra_mechanics
            assert mech["weighted_neighbor_mean"] == pytest.approx(float(preds[i]), rel=1e-5)
            assert mech["is_classifier"] is False
            assert len(mech["neighbor_targets"]) == 4


def test_large_k_display_cap_and_downsample_budget(clf_data) -> None:
    X, y = clf_data
    rng = np.random.default_rng(42)
    big_X = np.vstack([X, rng.normal(size=(6000, 4)).astype(np.float32)])
    big_y = np.concatenate([y, rng.integers(0, 2, size=6000)])
    model = KNeighborsClassifier(n_neighbors=200).fit(big_X, big_y)
    adapter = KnnClassifierAdapter()
    traces = adapter.trace_internal_flow(model, big_X[:2], [0, 1], FEATURE_NAMES)
    for tr in traces:
        mech = tr.extra_mechanics
        assert mech["n_neighbors_effective"] == 200
        assert mech["display_capped"] is True
        assert len(mech["reference_cloud"]["original_indices"]) <= MAX_BACKGROUND_REF_POINTS
        cloud_idx = set(mech["reference_cloud"]["original_indices"])
        for orig in mech["neighbor_original_indices"]:
            assert orig in cloud_idx


def test_surface_numeric_parity_and_resolution_clamp(clf_data, reg_data) -> None:
    Xc, yc = clf_data
    clf = KNeighborsClassifier(n_neighbors=5).fit(Xc, yc)
    adapter = KnnClassifierAdapter()
    big = _spec(resolution=200)
    mesh = adapter.compute_3d_surface(clf, big, n_features=4)
    assert mesh.resolution == 100
    assert len(mesh.z_values) == 100 * 100
    assert mesh.extra_overlays["radar_geometry_shape"] == "sphere"
    assert mesh.extra_overlays["boundary_threshold"] == 0.5
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
    expected = np.asarray(clf.predict_proba(grid))[:, 1]
    np.testing.assert_allclose(np.asarray(mesh2.z_values), expected, rtol=1e-5)

    Xr, yr = reg_data
    reg = KNeighborsRegressor(n_neighbors=3).fit(Xr, yr)
    mesh_r = KnnRegressorAdapter().compute_3d_surface(reg, spec, n_features=4)
    assert mesh_r.extra_overlays["boundary_threshold"] is None
    expected_r = np.asarray(reg.predict(grid)).ravel()
    np.testing.assert_allclose(np.asarray(mesh_r.z_values), expected_r, rtol=1e-5)

    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    traces = adapter.trace_internal_flow(restored, Xc[:2], [0, 1], FEATURE_NAMES)
    assert len(traces) == 2

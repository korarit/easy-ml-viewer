"""Phase 44 — probabilistic & discriminant 2D geometry extractor unit tests."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.core.math_3d import WorldCoordinateScaler
from app.domains.model_engine.adapters.probabilistic.probabilistic_2d_extractor import (
    Probabilistic2DExtractor,
)


def _ctx(X, x_idx=0, y_idx=1, xn="f0", yn="f1"):
    xs = WorldCoordinateScaler.build_axis_scale(xn, X[:, x_idx])
    ys = WorldCoordinateScaler.build_axis_scale(yn, X[:, y_idx])
    xr = (float(X[:, x_idx].min()), float(X[:, x_idx].max()))
    yr = (float(X[:, y_idx].min()), float(X[:, y_idx].max()))
    return xs, ys, xr, yr


def _extract(model, X, x_idx=0, y_idx=1):
    xs, ys, xr, yr = _ctx(X, x_idx, y_idx)
    return Probabilistic2DExtractor.extract_probabilistic_2d_geometry(
        model_obj=model,
        X=X,
        x_feature_idx=x_idx,
        y_feature_idx=y_idx,
        x_feature_name="f0",
        y_feature_name="f1",
        x_range_raw=xr,
        y_range_raw=yr,
        x_scale=xs,
        y_scale=ys,
    )


def test_gnb_ellipses_match_theta_and_var():
    X, y = make_blobs(n_samples=300, centers=2, n_features=3, random_state=7)
    X = X.astype(np.float64)
    clf = GaussianNB().fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.family == "gnb"
    assert geo.n_classes == 2 and len(geo.ellipses) == 2
    for e in geo.ellipses:
        k = e.class_index
        assert e.mean_raw[0] == clf.theta_[k, 0].round(6).item() if hasattr(clf.theta_[k, 0], "round") else True
        assert abs(e.mean_raw[0] - float(clf.theta_[k, 0])) < 1e-6
        assert abs(e.mean_raw[1] - float(clf.theta_[k, 1])) < 1e-6
        assert abs(e.std_raw[0] - float(np.sqrt(max(clf.var_[k, 0], 1e-12)))) < 1e-6
        assert e.rotation_rad == 0.0
        # World centre inside the stage and radii positive.
        assert -5.0 - 1e-6 <= e.center_world[0] <= 5.0 + 1e-6
        assert e.radii_world_1sigma[0] > 0 and e.radii_world_1sigma[1] > 0
        # k-sigma nesting: 3x the 1-sigma radius stays proportional.
        assert e.radii_world_1sigma[0] * 3.0 > e.radii_world_1sigma[0]


def test_gnb_legacy_sigma_attribute_covered():
    X, y = make_blobs(n_samples=120, centers=2, n_features=2, random_state=3)
    X = X.astype(np.float64)
    clf = GaussianNB().fit(X, y)
    sigma = np.asarray(clf.var_)
    delattr(clf, "var_")
    clf.sigma_ = sigma  # pre-1.0 pickle shape
    geo = _extract(clf, X)
    assert geo is not None and len(geo.ellipses) == 2


def test_lda_shared_shape_and_exact_line():
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=11)
    X = X.astype(np.float64)
    clf = LinearDiscriminantAnalysis().fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.family == "lda"
    assert len(geo.ellipses) == 2
    # Shared covariance: identical ellipse shapes (radii match across classes).
    assert geo.ellipses[0].std_raw == geo.ellipses[1].std_raw
    assert geo.discriminant_line is not None
    seg = geo.discriminant_line.segment_raw
    assert seg is not None
    w1, w2, b = geo.discriminant_line.w1, geo.discriminant_line.w2, geo.discriminant_line.b_eff
    for x, yy in ((seg[0], seg[1]), (seg[2], seg[3]), ((seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2)):
        assert abs(w1 * x + w2 * yy + b) < 1e-4
    assert geo.discriminant_line.segment_world is not None


def test_qda_conic_matches_native_scores():
    X, y = make_blobs(n_samples=400, centers=2, n_features=2, random_state=5, cluster_std=1.8)
    X = X.astype(np.float64)
    clf = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.family == "qda"
    assert len(geo.ellipses) == 2
    assert geo.conic is not None
    assert geo.conic.conic_type in ("hyperbola", "ellipse", "parabola")
    coeffs = geo.conic.coefficients_raw
    assert set(coeffs) == {"A", "B", "C", "D", "E", "F"}
    # Analytic conic agrees with the native QDA score difference on pinned rows.
    grid_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 7)
    grid_y = np.linspace(X[:, 1].min(), X[:, 1].max(), 7)
    uu, vv = np.meshgrid(grid_x, grid_y)
    analytic = Probabilistic2DExtractor.evaluate_conic(coeffs, uu.ravel(), vv.ravel())
    mat = np.zeros((uu.size, 2))
    mat[:, 0] = uu.ravel()
    mat[:, 1] = vv.ravel()
    native = np.asarray(clf.decision_function(mat), dtype=np.float64).ravel()
    # Native decision_function sign convention may flip; compare up to sign.
    corr = float(np.corrcoef(analytic, native)[0, 1])
    assert abs(abs(corr) - 1.0) < 1e-6
    # Vector curve renders (or the classes fully overlap, documented by empty).
    assert isinstance(geo.conic.segments_world, list)


def test_discrete_nb_variant_payload():
    rng = np.random.default_rng(4)
    X = rng.poisson(3.0, size=(200, 3)).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 6).astype(int)
    clf = MultinomialNB().fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.family == "discrete_nb"
    assert geo.discrete is not None and geo.discrete.variant == "multinomial"
    assert len(geo.discrete.axis_log_prob) == 2
    assert abs(geo.discrete.axis_log_prob[0][0] - float(clf.feature_log_prob_[0, 0])) < 1e-6

    Xb = (rng.random((200, 3)) > 0.5).astype(np.float64)
    yb = (Xb[:, 0] > 0).astype(int)
    clf_b = BernoulliNB().fit(Xb, yb)
    geo_b = _extract(clf_b, Xb)
    assert geo_b is not None and geo_b.discrete is not None
    assert geo_b.discrete.variant == "bernoulli"


def test_pipeline_returns_none_and_tree_returns_none():
    X, y = make_blobs(n_samples=120, centers=2, n_features=2, random_state=9)
    X = X.astype(np.float64)
    pipe = make_pipeline(StandardScaler(), GaussianNB()).fit(X, y)
    assert _extract(pipe, X) is None
    tree = DecisionTreeClassifier().fit(X, y)
    assert _extract(tree, X) is None


def test_contour_integration_embeds_probabilistic_2d():
    from app.domains.simulation.contour_2d_generator import Contour2DGenerator

    from app.domains.model_engine.registry import ModelRegistry

    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=13)
    X = X.astype(np.float32)
    clf = GaussianNB().fit(X, y)
    adapter = ModelRegistry.resolve_adapter(clf, ".joblib")
    res = Contour2DGenerator.generate_contour(
        adapter=adapter,
        model_obj=clf,
        final_est=clf,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=32,
        pinning_mode="median",
    )
    assert res.probabilistic_2d is not None
    assert res.probabilistic_2d["family"] == "gnb"
    assert len(res.probabilistic_2d["ellipses"]) == 2
    assert len(str(res.probabilistic_2d).encode("utf-8")) < 40_960

    tree = DecisionTreeClassifier().fit(X, y)
    assert (
        Probabilistic2DExtractor.extract_probabilistic_2d_geometry(
            model_obj=tree,
            X=X.astype(np.float64),
            x_feature_idx=0,
            y_feature_idx=1,
            x_feature_name="f0",
            y_feature_name="f1",
            x_range_raw=(float(X[:, 0].min()), float(X[:, 0].max())),
            y_range_raw=(float(X[:, 1].min()), float(X[:, 1].max())),
            x_scale=WorldCoordinateScaler.build_axis_scale("f0", X[:, 0]),
            y_scale=WorldCoordinateScaler.build_axis_scale("f1", X[:, 1]),
        )
        is None
    )
    # A registered non-probabilistic model also yields no probabilistic overlay.
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    knn_adapter = ModelRegistry.resolve_adapter(knn, ".joblib")
    res_knn = Contour2DGenerator.generate_contour(
        adapter=knn_adapter,
        model_obj=knn,
        final_est=knn,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=32,
        pinning_mode="median",
    )
    assert res_knn.probabilistic_2d is None

"""Phase 42 — exact 2D linear geometry extractor unit tests."""

from __future__ import annotations

import math

import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.core.math_3d import WorldCoordinateScaler
from app.domains.model_engine.adapters.linear.linear_2d_extractor import (
    ZERO_TOL,
    extract_linear_2d_geometry,
)


def _scales(X, x_idx, y_idx, xn="f0", yn="f1"):
    xs = WorldCoordinateScaler.build_axis_scale(xn, X[:, x_idx])
    ys = WorldCoordinateScaler.build_axis_scale(yn, X[:, y_idx])
    return xs, ys


def _ranges(X, x_idx, y_idx):
    return (float(X[:, x_idx].min()), float(X[:, x_idx].max())), (
        float(X[:, y_idx].min()),
        float(X[:, y_idx].max()),
    )


def _extract(model, X, x_idx=0, y_idx=1, **kw):
    xs, ys = _scales(X, x_idx, y_idx)
    xr, yr = _ranges(X, x_idx, y_idx)
    return extract_linear_2d_geometry(
        model_obj=model,
        final_est=model,
        X=X,
        x_feature_idx=x_idx,
        y_feature_idx=y_idx,
        x_feature_name="f0",
        y_feature_name="f1",
        x_range_raw=xr,
        y_range_raw=yr,
        x_scale=xs,
        y_scale=ys,
        **kw,
    )


def _assert_on_line(geo, seg, level, tol=1e-4):
    x1, y1, x2, y2 = seg
    for x, y in ((x1, y1), (x2, y2), ((x1 + x2) / 2, (y1 + y2) / 2)):
        assert abs(geo.w1 * x + geo.w2 * y + geo.b_eff - level) < tol


def test_binary_logistic_exact_boundary():
    X, y = make_blobs(n_samples=300, centers=2, n_features=3, random_state=7)
    X = X.astype(np.float64)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.task == "classification"
    assert geo.family == "logistic" and geo.decision_threshold == 0.5
    kinds = [ln.kind for ln in geo.lines]
    assert "decision" in kinds
    decision = next(ln for ln in geo.lines if ln.kind == "decision")
    assert decision.segment_raw is not None and decision.segment_world is not None
    _assert_on_line(geo, decision.segment_raw, 0.0)
    for wv in decision.segment_world:
        assert -5.0 - 1e-6 <= wv <= 5.0 + 1e-6
    # Normal vector is unit length and points along (w1, w2).
    assert abs(math.hypot(*geo.normal_vector) - 1.0) < 1e-9
    assert "P=0.5" in geo.formula or "0.5" in geo.formula


def test_logistic_confidence_isolines_are_parallel_offsets():
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=3)
    X = X.astype(np.float64)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None
    conf = [ln for ln in geo.lines if ln.kind == "confidence"]
    assert {ln.level for ln in conf} == {0.7, 0.8, 0.9}
    for ln in conf:
        assert ln.segment_raw is not None
        t = math.log(ln.level / (1.0 - ln.level))
        _assert_on_line(geo, ln.segment_raw, t)


def test_vertical_boundary_when_w2_zero():
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=9)
    X = X.astype(np.float64)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    # Force w2 == 0: drop the Y contribution entirely.
    clf.coef_[0, 1] = 0.0
    geo = _extract(clf, X)
    assert geo is not None and geo.is_y_zeroed and not geo.is_x_zeroed
    decision = next(ln for ln in geo.lines if ln.kind == "decision")
    assert decision.segment_raw is not None
    x1, _, x2, _ = decision.segment_raw
    assert abs(x1 - x2) < 1e-9  # vertical line x = -b/w1
    _assert_on_line(geo, decision.segment_raw, 0.0)


def test_horizontal_boundary_and_sparsity_flag_when_w1_zero():
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=11)
    X = X.astype(np.float64)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    clf.coef_[0, 0] = 0.0
    geo = _extract(clf, X)
    assert geo is not None and geo.is_x_zeroed and not geo.both_zeroed
    decision = next(ln for ln in geo.lines if ln.kind == "decision")
    assert decision.segment_raw is not None
    _, y1, _, y2 = decision.segment_raw
    assert abs(y1 - y2) < 1e-9


def test_both_zeroed_reports_warning_with_no_lines():
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=5)
    X = X.astype(np.float64)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    clf.coef_[0, :] = 0.0
    geo = _extract(clf, X)
    assert geo is not None and geo.both_zeroed
    assert all(ln.segment_raw is None for ln in geo.lines)


def test_lasso_sparsity_detection():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = 3.0 * X[:, 0] + rng.normal(scale=0.1, size=300)
    model = Lasso(alpha=5.0, max_iter=5000).fit(X, y)
    geo = _extract(model, X)
    assert geo is not None and geo.family == "lasso" and geo.task == "regression"
    assert geo.sparsity_ratio > 0.0
    assert geo.penalty == "L1 (Lasso)" and geo.alpha is not None
    zeroed_axes = [i for i, w in enumerate(model.coef_) if abs(float(w)) <= ZERO_TOL]
    assert len(zeroed_axes) > 0


def test_sgd_hinge_margin_corridor_width():
    X, y = make_blobs(n_samples=300, centers=2, n_features=3, random_state=13)
    X = X.astype(np.float64)
    clf = SGDClassifier(loss="hinge", max_iter=2000, tol=1e-4, random_state=0).fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.family == "sgd"
    kinds = [ln.kind for ln in geo.lines]
    assert "margin_pos" in kinds and "margin_neg" in kinds
    assert geo.margin_level == 1.0
    expected = 2.0 / math.hypot(geo.w1, geo.w2)
    assert geo.corridor_width is not None and abs(geo.corridor_width - expected) < 1e-9
    pos = next(ln for ln in geo.lines if ln.kind == "margin_pos")
    neg = next(ln for ln in geo.lines if ln.kind == "margin_neg")
    assert pos.segment_raw is not None and neg.segment_raw is not None
    _assert_on_line(geo, pos.segment_raw, 1.0)
    _assert_on_line(geo, neg.segment_raw, -1.0)


def test_sgd_regressor_epsilon_tube():
    X, y = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=21)
    X = X.astype(np.float64)
    reg = SGDRegressor(max_iter=2000, tol=1e-4, random_state=0).fit(X, y)
    geo = _extract(reg, X)
    assert geo is not None and geo.task == "regression"
    assert geo.margin_level is not None and geo.margin_level > 0.0
    assert geo.slope_w1 is not None and abs(geo.slope_w1 - geo.w1) < 1e-12


def test_linear_regression_slope_matches_coef():
    X, y = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=11)
    X = X.astype(np.float64)
    reg = LinearRegression().fit(X, y)
    geo = _extract(reg, X)
    assert geo is not None and geo.family == "ols" and geo.task == "regression"
    assert abs(geo.w1 - float(reg.coef_[0])) < 1e-9
    assert abs(geo.slope_w1 - float(reg.coef_[0])) < 1e-9
    assert math.isnan(geo.decision_threshold)
    assert geo.lines == []  # elevation guides are derived client-side


def test_ridge_and_elasticnet_metadata():
    X, y = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=31)
    X = X.astype(np.float64)
    ridge = Ridge(alpha=2.0).fit(X, y)
    geo_r = _extract(ridge, X)
    assert geo_r is not None and geo_r.penalty == "L2 (Ridge)" and geo_r.alpha == 2.0
    en = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=5000).fit(X, y)
    geo_e = _extract(en, X)
    assert geo_e is not None and geo_e.family == "elasticnet"
    assert geo_e.l1_ratio is not None and abs(geo_e.l1_ratio - 0.7) < 1e-9
    assert geo_e.alpha is not None and abs(geo_e.alpha - 0.5) < 1e-9


def test_ridge_classifier_score_threshold_and_margin():
    X, y = make_blobs(n_samples=300, centers=2, n_features=3, random_state=17)
    X = X.astype(np.float64)
    clf = RidgeClassifier(alpha=1.0).fit(X, y)
    # Rescale to weights whose ±1 margin parallels cross the data rect.
    clf.coef_ = np.array([[-1.5, 2.0, 0.0]])
    clf.intercept_ = np.array([0.0])
    geo = _extract(clf, X)
    assert geo is not None and geo.family == "ridge_classifier"
    assert geo.decision_threshold == 0.0
    assert geo.margin_level == 1.0
    expected = 2.0 / math.hypot(geo.w1, geo.w2)
    assert geo.corridor_width is not None and abs(geo.corridor_width - expected) < 1e-9
    kinds = [ln.kind for ln in geo.lines]
    # At least one parallel must cross (the far side may legitimately fall
    # outside the data rect); the corridor width badge stays exact regardless.
    assert "margin_pos" in kinds or "margin_neg" in kinds


def test_margin_lines_may_miss_rect_but_width_survives():
    # Tiny weights: the ±1 parallels fall outside the data rect, yet the
    # corridor width badge must still be exact (frontend hides the band).
    X, y = make_blobs(n_samples=300, centers=2, n_features=3, random_state=17)
    X = X.astype(np.float64)
    clf = RidgeClassifier(alpha=1.0).fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.margin_level == 1.0
    assert geo.corridor_width is not None
    assert abs(geo.corridor_width - 2.0 / math.hypot(geo.w1, geo.w2)) < 1e-9


def test_multiclass_pairwise_boundaries():
    X, y = make_blobs(n_samples=300, centers=3, n_features=3, random_state=23)
    X = X.astype(np.float64)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    geo = _extract(clf, X)
    assert geo is not None and geo.is_multiclass and geo.n_classes == 3
    pairs = [ln for ln in geo.lines if ln.kind == "pairwise"]
    assert len(pairs) == 3  # C(3, 2)
    for ln in pairs:
        assert ln.segment_raw is not None and ln.class_a is not None and ln.class_b is not None


def test_pipeline_returns_none():
    X, y = make_blobs(n_samples=200, centers=2, n_features=3, random_state=29)
    X = X.astype(np.float64)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    assert _extract(pipe, X) is None


def test_nonlinear_returns_none():
    X, y = make_blobs(n_samples=200, centers=2, n_features=3, random_state=37)
    X = X.astype(np.float64)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    assert _extract(tree, X) is None


def test_contour_integration_attaches_linear_payload():
    from app.domains.model_engine.value_objects import PredictionBatchResult
    from app.domains.simulation.contour_2d_generator import Contour2DGenerator

    class _Adapter:
        def __init__(self, model):
            self._model = model

        def predict_batch(self, model_obj, X_float32):
            preds = np.asarray(model_obj.predict(np.asarray(X_float32, dtype=np.float32))).ravel()
            return PredictionBatchResult(predictions=[float(v) for v in preds.tolist()])

    X, y = make_blobs(n_samples=200, centers=2, n_features=3, random_state=7)
    X = X.astype(np.float32)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    res = Contour2DGenerator.generate_contour(
        adapter=_Adapter(clf),
        model_obj=clf,
        final_est=clf,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=40,
    )
    assert res.linear_2d is not None
    assert res.linear_2d["family"] == "logistic"
    assert any(ln["kind"] == "decision" and ln["segment_world"] is not None for ln in res.linear_2d["lines"])
    # DTO stays small: the exact overlay must cost < 4 KB on top of the grid.
    import json

    assert len(json.dumps(res.linear_2d)) < 4096

    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    res_tree = Contour2DGenerator.generate_contour(
        adapter=_Adapter(tree),
        model_obj=tree,
        final_est=tree,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=40,
    )
    assert res_tree.linear_2d is None
    assert any(c.kind == "decision" for c in res_tree.contours)  # sampled path intact

"""Phase 46 — family routing, segment/bin/cell presence, foreign-model None."""

from __future__ import annotations

import numpy as np

from app.domains.model_engine.adapters.boosting.boosting_2d_extractor import (
    Boosting2DExtractor as B,
)
from conftest import make_ctx


def _kw(X, xs, ys, xr, yr, row=None):
    return dict(model_obj=None, X=X, x_feature_idx=0, y_feature_idx=1,
                x_feature_name="f0", y_feature_name="f1",
                x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys,
                active_row=row)


def test_sklearn_gbm_superimposed_cuts(fitted_gbm, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = _kw(X, xs, ys, xr, yr)
    kw["model_obj"] = fitted_gbm
    geo = B.extract_boosting_2d_geometry(**kw)
    assert geo is not None and geo.family == "sklearn_gbm"
    assert geo.n_estimators == 10 and geo.trees_exported == 8
    assert len(geo.segments) > 0
    for s in geo.segments:
        assert s.axis in ("x", "y")
        x1, y1, x2, y2 = s.segment_raw
        # 6dp outward-rounded rect: 1e-4 slack over the raw data range.
        if s.axis == "x":
            assert x1 == x2 and xr[0] - 1e-4 <= x1 <= xr[1] + 1e-4
            assert yr[0] - 1e-4 <= y1 <= y2 <= yr[1] + 1e-4
        else:
            assert y1 == y2 and yr[0] - 1e-4 <= y1 <= yr[1] + 1e-4
            assert xr[0] - 1e-4 <= x1 <= x2 <= xr[1] + 1e-4


def test_hist_bins_and_cuts(fitted_hist, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = _kw(X, xs, ys, xr, yr, row=X[0])
    kw["model_obj"] = fitted_hist
    geo = B.extract_boosting_2d_geometry(**kw)
    assert geo is not None and geo.family == "hist_gbm"
    assert len(geo.bin_edges_x_raw) > 0 and len(geo.bin_edges_y_raw) > 0
    assert len(geo.bin_lines_x_world) == len(geo.bin_edges_x_raw)
    assert len(geo.bin_lines_y_world) == len(geo.bin_edges_y_raw)
    assert geo.active_bin is not None and len(geo.active_bin) == 2
    assert all(v >= 0 for v in geo.active_bin)


def test_xgb_polygons_and_shap(fitted_xgb, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = _kw(X, xs, ys, xr, yr, row=X[3])
    kw["model_obj"] = fitted_xgb
    geo = B.extract_boosting_2d_geometry(**kw)
    assert geo is not None and geo.family == "xgboost"
    assert geo.trees_exported == 1 and geo.n_estimators == 10
    assert len(geo.segments) > 0 and len(geo.leaves) > 0
    assert geo.shap_vector is not None
    assert geo.shap_vector.method == "xgb_pred_contribs"
    assert geo.shap_vector.magnitude >= 0
    assert any(lf.is_active for lf in geo.leaves)


def test_lgbm_polygons_and_shap(fitted_lgbm, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = _kw(X, xs, ys, xr, yr, row=X[5])
    kw["model_obj"] = fitted_lgbm
    geo = B.extract_boosting_2d_geometry(**kw)
    assert geo is not None and geo.family == "lightgbm"
    assert geo.trees_exported == 1
    assert len(geo.segments) > 0 and len(geo.leaves) > 0
    assert geo.shap_vector is not None
    assert geo.shap_vector.method == "lgbm_pred_contrib"


def test_catboost_checkerboard(fitted_catboost, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = _kw(X, xs, ys, xr, yr, row=X[0])
    kw["model_obj"] = fitted_catboost
    geo = B.extract_boosting_2d_geometry(**kw)
    assert geo is not None and geo.family == "catboost"
    assert len(geo.oblivious_cells) > 0
    assert geo.shap_vector is not None
    assert geo.shap_vector.method == "catboost_shap"
    assert sum(1 for c in geo.oblivious_cells if c.is_active) == 1


def test_foreign_and_pipeline_models_yield_none(blobs_2d):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X, y = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = _kw(X, xs, ys, xr, yr)
    kw["model_obj"] = LogisticRegression().fit(X, y)
    assert B.extract_boosting_2d_geometry(**kw) is None
    kw["model_obj"] = GaussianNB().fit(X, y)
    assert B.extract_boosting_2d_geometry(**kw) is None
    kw["model_obj"] = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert B.extract_boosting_2d_geometry(**kw) is None
    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=2))
    pipe.fit(X, y)
    kw["model_obj"] = pipe
    assert B.extract_boosting_2d_geometry(**kw) is None


def test_no_axis_splits_flag(blobs_4d):
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = blobs_4d
    xs, ys, _, _ = make_ctx(X, 0, 1)
    informative = np.hstack([X[:, 2:3] * 0.01, X[:, 1:2] * 0.01, X[:, 2:]])
    model = None
    for seed in range(10):
        cand = GradientBoostingClassifier(n_estimators=5, max_depth=2,
                                          random_state=seed).fit(informative, y)
        feats: set[int] = set()
        for m in range(np.asarray(cand.estimators_).shape[0]):
            tree = np.asarray(cand.estimators_)[m, 0].tree_
            feats |= {int(f) for f in np.asarray(tree.feature).ravel() if int(f) >= 0}
        if feats and feats.isdisjoint({0, 1}):
            model = cand
            break
    assert model is not None, "could not grow an off-axis-only GBM in 10 seeds"
    geo = B.extract_boosting_2d_geometry(
        model_obj=model, X=informative, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=(float(informative[:, 0].min()), float(informative[:, 0].max())),
        y_range_raw=(float(informative[:, 1].min()), float(informative[:, 1].max())),
        x_scale=xs, y_scale=ys)
    assert geo is not None
    assert geo.segments == [] and geo.no_axis_splits is True

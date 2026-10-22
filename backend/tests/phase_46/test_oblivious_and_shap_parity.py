"""Phase 46 — CatBoost checkerboard bitwise parity + native SHAP parity (DoD)."""

from __future__ import annotations

import numpy as np

from app.domains.model_engine.adapters.boosting.boosting_2d_extractor import (
    Boosting2DExtractor as B,
)
from conftest import make_ctx


def test_checkerboard_cells_form_exact_tiling(fitted_catboost, blobs_2d):
    """Cells tile the rect with no gaps/overlaps: areas sum to rect area."""
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = B.extract_boosting_2d_geometry(
        model_obj=fitted_catboost, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys, active_row=X[0])
    assert geo is not None and geo.family == "catboost"
    assert len(geo.oblivious_cells) == (len(geo.oblivious_x_cuts_raw) + 1) * (
        len(geo.oblivious_y_cuts_raw) + 1)
    assert len(geo.oblivious_x_lines_world) == len(geo.oblivious_x_cuts_raw)
    assert len(geo.oblivious_y_lines_world) == len(geo.oblivious_y_cuts_raw)
    assert all(-5.5 <= v <= 5.5 for v in geo.oblivious_x_lines_world)
    assert all(-5.5 <= v <= 5.5 for v in geo.oblivious_y_lines_world)
    total = sum((c.box_raw[1] - c.box_raw[0]) * (c.box_raw[3] - c.box_raw[2])
                for c in geo.oblivious_cells)
    rect = (xr[1] - xr[0]) * (yr[1] - yr[0])
    assert abs(total - rect) < 1e-3 * max(rect, 1e-12)
    depth = len(fitted_catboost._get_tree_splits(0, __import__("catboost").Pool(X[:50])))
    for c in geo.oblivious_cells:
        assert len(c.bitmask) == depth
        assert 0 <= c.leaf_index < 2 ** depth


def test_bitwise_leaf_matches_calc_leaf_indexes(fitted_catboost, blobs_2d):
    """Cell leaf indexes agree with ``calc_leaf_indexes`` at cell centers."""
    from catboost import Pool

    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = B.extract_boosting_2d_geometry(
        model_obj=fitted_catboost, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and len(geo.oblivious_cells) > 0
    centers = np.array([[ (c.box_raw[0] + c.box_raw[1]) / 2.0,
                           (c.box_raw[2] + c.box_raw[3]) / 2.0]
                        for c in geo.oblivious_cells], dtype=np.float32)
    med = np.nanmedian(X.astype(np.float64), axis=0)
    full = np.tile(med, (len(centers), 1))
    full[:, 0] = centers[:, 0]
    full[:, 1] = centers[:, 1]
    leaves = np.asarray(
        fitted_catboost.calc_leaf_indexes(Pool(full), thread_count=2))
    assert leaves.shape == (len(centers), int(fitted_catboost.tree_count_))
    for c, lv in zip(geo.oblivious_cells, leaves[:, 0].tolist()):
        assert c.leaf_index == int(lv)


def test_xgb_shap_matches_pred_contribs(fitted_xgb, blobs_2d):
    """2D push vector equals the native C++ TreeSHAP slice (DoD)."""
    import warnings

    import xgboost as xgb

    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    row = X[3]
    geo = B.extract_boosting_2d_geometry(
        model_obj=fitted_xgb, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys, active_row=row)
    assert geo is not None and geo.shap_vector is not None
    booster = fitted_xgb.get_booster()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        dm = xgb.DMatrix(np.ascontiguousarray(row.reshape(1, -1).astype(np.float32)))
        raw = np.asarray(booster.predict(dm, pred_contribs=True, strict_shape=True))
    if raw.ndim == 2:
        phi = raw[0]
    else:
        phi = raw[0, 0]
    # Extractor rounds to 5dp for the DTO; parity holds to 1e-4.
    assert abs(geo.shap_vector.phi_x - float(np.asarray(phi[0], dtype=np.float64))) < 1e-4
    assert abs(geo.shap_vector.phi_y - float(np.asarray(phi[1], dtype=np.float64))) < 1e-4


def test_lgbm_shap_matches_pred_contrib(fitted_lgbm, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    row = X[5]
    geo = B.extract_boosting_2d_geometry(
        model_obj=fitted_lgbm, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys, active_row=row)
    assert geo is not None and geo.shap_vector is not None
    booster = fitted_lgbm.booster_
    raw = np.asarray(booster.predict(row.reshape(1, -1).astype(np.float32),
                                     pred_contrib=True, num_thread=2)).ravel()
    assert abs(geo.shap_vector.phi_x - float(raw[0])) < 1e-4
    assert abs(geo.shap_vector.phi_y - float(raw[1])) < 1e-4

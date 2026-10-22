"""Phase 46 — HistGBM bin edges match the fitted ``_bin_mapper`` 100% (DoD)."""

from __future__ import annotations

import numpy as np

from app.domains.model_engine.adapters.boosting.boosting_2d_extractor import (
    Boosting2DExtractor as B,
)
from conftest import make_ctx


def test_bin_edges_match_mapper_exactly(fitted_hist, blobs_2d):
    """E46.3 — exported edges equal the mapper thresholds with no downsampling."""
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = B.extract_boosting_2d_geometry(
        model_obj=fitted_hist, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and geo.family == "hist_gbm"
    mapper_edges = list(fitted_hist._bin_mapper.bin_thresholds_)
    for axis, idx in (("x", 0), ("y", 1)):
        expected = [round(float(v), 5) for v in np.asarray(mapper_edges[idx]).ravel().tolist()]
        exported = geo.bin_edges_x_raw if axis == "x" else geo.bin_edges_y_raw
        assert exported == expected
        assert all(expected[i] < expected[i + 1] for i in range(len(expected) - 1))


def test_active_bin_matches_searchsorted(fitted_hist, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    row = X[7]
    geo = B.extract_boosting_2d_geometry(
        model_obj=fitted_hist, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys, active_row=row)
    assert geo is not None and geo.active_bin is not None
    ex = int(np.searchsorted(np.asarray(geo.bin_edges_x_raw), float(row[0]), side="left"))
    ey = int(np.searchsorted(np.asarray(geo.bin_edges_y_raw), float(row[1]), side="left"))
    assert geo.active_bin == [ex, ey]


def test_no_active_bin_without_row(fitted_hist, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = B.extract_boosting_2d_geometry(
        model_obj=fitted_hist, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and geo.active_bin is None

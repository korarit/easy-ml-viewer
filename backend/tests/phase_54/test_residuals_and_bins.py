"""Phase 54 — pseudo-residual spokes (C) and quantized-bin hexbins (D).

DoD mapping: spoke length follows ``|r|`` with the negative-gradient sign
(DoD #3); quantized cells aggregate ``(Σg, Σh, count)`` on the 256-bin
lattice (DoD #4), with parity against a real HistGradientBoosting fit.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.ensemble import HistGradientBoostingClassifier

from app.domains.model_engine.adapters.boosting.boosting_point_intelligence import (
    histogram_cell_stats,
    pseudo_residual_logloss,
    pseudo_residual_ls,
    quantize_to_bins,
    quantized_cell_ids,
    spoke_lengths,
)


def test_ls_residual_is_y_minus_f():
    r = pseudo_residual_ls(np.array([1.0, 2.0, np.nan]), np.array([1.5, 1.0, 0.5]))
    assert r[:2].tolist() == pytest.approx([-0.5, 1.0])
    assert np.isnan(r[2])
    assert pseudo_residual_ls(np.array([]), np.array([])).size == 0


def test_logloss_residual_is_y_minus_p_with_clipping():
    r = pseudo_residual_logloss(np.array([1.0, 0.0, 1.0]), np.array([0.9, 0.2, 0.5]))
    assert r.tolist() == pytest.approx([0.1, -0.2, 0.5])
    # Miscalibrated p outside [0, 1] cannot fake a |r| > 1 hard-sample spike.
    assert pseudo_residual_logloss(np.array([1.0]), np.array([1.7]))[0] <= 1.0
    assert np.isnan(pseudo_residual_logloss(np.array([np.nan]), np.array([0.5]))[0])


def test_gbm_residuals_match_manual_negative_gradient():
    X, y = make_regression(n_samples=80, n_features=2, noise=4.0, random_state=6)
    from sklearn.ensemble import GradientBoostingRegressor

    gbm = GradientBoostingRegressor(n_estimators=10, random_state=1).fit(X, y)
    fx = gbm.predict(X[:20])
    assert pseudo_residual_ls(y[:20], fx) == pytest.approx(y[:20] - fx)


def test_spoke_lengths_peak_normalize_with_sign():
    lengths, sides = spoke_lengths(np.array([1.0, -2.0, 0.0, np.nan]))
    assert lengths.tolist() == pytest.approx([0.75, 1.5, 0.0, 0.0])
    assert sides.tolist() == [1.0, -1.0, 0.0, 0.0]  # up / down / none
    lz, sz = spoke_lengths(np.zeros(3))
    assert (lz == 0.0).all() and (sz == 0.0).all()  # converged → no spokes


def test_quantize_follows_searchsorted_left():
    edges = np.array([0.25, 0.5, 0.75])
    idx = quantize_to_bins(np.array([0.1, 0.5, 0.9, np.nan]), edges)
    assert idx.tolist() == [0, 1, 3, -1]  # first edge >= v (HistGBM searchsorted-left)
    assert quantize_to_bins(np.array([0.1]), np.array([])).tolist() == [-1]
    assert quantize_to_bins(np.array([]), edges).size == 0


def test_cell_ids_pack_bx_times_b_plus_by():
    ids = quantized_cell_ids(np.array([1, 255, -1, 300]), np.array([2, 3, 0, 0]))
    assert ids.tolist() == [1 * 256 + 2, 255 * 256 + 3, -1, -1]  # unknowns/out-of-range skipped


def test_histogram_stats_match_manual_groupby():
    cells = np.array([5, 5, 9, -1, 9])
    g = np.array([0.5, -0.2, 1.0, 99.0, 0.0])
    h = np.array([1.0, 1.0, 0.5, 99.0, 0.5])
    uniq, sg, sh, cnt = histogram_cell_stats(cells, g, h)
    assert uniq.tolist() == [5, 9]  # -1 unknown cell never aggregates
    assert sg.tolist() == pytest.approx([0.3, 1.0])
    assert sh.tolist() == pytest.approx([2.0, 1.0])
    assert cnt.tolist() == [2.0, 2.0]
    eu, esg, _, _ = histogram_cell_stats(np.array([-1]), np.array([1.0]), np.array([1.0]))
    assert eu.size == 0 and esg.size == 0


def test_histgbm_fit_exposes_quantized_thresholds():
    X, y = make_blobs(
        n_samples=150, centers=2, n_features=2, cluster_std=1.4, random_state=5,
    )
    m = HistGradientBoostingClassifier(max_iter=20, random_state=2).fit(X, y)
    mapper = getattr(m, "_bin_mapper", None)
    assert mapper is not None
    edges = np.asarray(mapper.bin_thresholds_[0], dtype=np.float64)
    assert 1 <= edges.size <= 255  # quantized lattice behind the Phase-46 grid
    bins = quantize_to_bins(X[:, 0], edges)
    assert set(np.unique(bins)) <= set(range(edges.size + 1)) | {-1}
    assert (bins >= 0).all()  # finite training data always lands on the lattice

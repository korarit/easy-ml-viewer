"""Phase 51 — SVM halo dual-ring (B), SVR slack spokes (C) and KNN density (D).

DoD mapping: SV halos follow ``|α|`` vs ``C`` (DoD #1), SVR spokes appear
only outside the ``ε``-tube (DoD #2), vote/density math feeds DoD #3/#4.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.svm import SVC, SVR

from app.domains.model_engine.adapters.distance.distance_point_intelligence import (
    BOUND_TOL,
    classify_support_vector,
    knn_density,
    support_halo_scales,
    svr_slack,
    svr_spoke_lengths,
)


def _svc():
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=4)
    m = SVC(C=1.0, kernel="rbf").fit(X, y)
    return m


def test_bound_vs_free_status_follows_alpha_over_c():
    m = _svc()
    dual = np.asarray(m.dual_coef_, dtype=np.float64).ravel()
    c = float(m.C)
    statuses = [classify_support_vector(a, c) for a in dual]
    for a, s in zip(dual.tolist(), statuses):
        if abs(a) >= c * (1.0 - BOUND_TOL):
            assert s == "violates_margin"
        elif abs(a) > 1e-12:
            assert s == "on_margin"
        else:
            assert s == "support"
    assert "violates_margin" in statuses or "on_margin" in statuses


def test_spot_check_against_spec_thresholds():
    assert classify_support_vector(1.0, 1.0) == "violates_margin"
    assert classify_support_vector(0.3, 1.0) == "on_margin"
    assert classify_support_vector(0.0, 1.0) == "support"
    assert classify_support_vector(0.9, None) == "support"


def test_halo_scales_clamped_and_ranked():
    m = _svc()
    dual = np.asarray(m.dual_coef_, dtype=np.float64).ravel()
    scales = support_halo_scales(dual, float(m.C))
    assert scales.shape == dual.shape
    assert bool((scales >= 0.0).all() and (scales <= 1.0).all())
    # Ranking parity: strongest |α| maps to the largest halo.
    assert int(np.argmax(scales)) == int(np.argmax(np.abs(dual)))


def test_svr_slack_only_outside_epsilon_tube():
    yt = np.array([1.0, 0.05, -0.05, -1.0])
    yp = np.zeros(4)
    xi, xi_star, outside = svr_slack(yt, yp, 0.1)
    assert outside.tolist() == [True, False, False, True]
    assert xi[0] == pytest.approx(0.9) and xi_star[0] == 0.0
    assert xi_star[3] == pytest.approx(0.9) and xi[3] == 0.0
    assert (xi[1:3] == 0.0).all() and (xi_star[1:3] == 0.0).all()


def test_svr_spoke_lengths_peak_normalized():
    xi = np.array([0.9, 0.0, 0.0])
    xs = np.array([0.0, 0.0, 0.45])
    lengths, sides = svr_spoke_lengths(xi, xs)
    assert max(lengths) == pytest.approx(1.5)  # peak slack → max_world
    assert sides.tolist() == [-1.0, 0.0, 1.0]
    l0, s0 = svr_spoke_lengths(np.zeros(2), np.zeros(2))
    assert (l0 == 0.0).all() and (s0 == 0.0).all()


def test_svr_end_to_end_slack_matches_residuals():
    X, y = make_regression(n_samples=60, n_features=2, noise=5.0, random_state=9)
    m = SVR(epsilon=0.5).fit(X, y)
    pred = m.predict(X)
    xi, xi_star, outside = svr_slack(y, pred, float(m.epsilon))
    resid = np.abs(y - pred)
    assert outside.tolist() == (resid > 0.5 + 1e-12).tolist()
    assert np.all(xi[outside] + xi_star[outside] > 0.0)


def test_knn_density_matches_closed_form():
    rho = knn_density(5, 1.0)
    assert rho[0] == pytest.approx(5.0 / np.pi, rel=1e-12)
    rhos = knn_density(np.array([5, 10]), np.array([1.0, 1.0]))
    assert rhos[1] == pytest.approx(2.0 * rhos[0], rel=1e-12)
    bad = knn_density(np.array([5, 5]), np.array([0.0, np.nan]))
    assert bool(np.isnan(bad).all())  # degenerate bins → NaN, never a spike

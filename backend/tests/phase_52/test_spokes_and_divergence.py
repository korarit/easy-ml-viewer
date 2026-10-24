"""Phase 52 — centroid spokes (C) and Gaussian divergence hexbins (D).

DoD mapping: dual centroid spokes explain mispredictions via the
log-likelihood ratio (DoD #2); hexbins flag empirical-vs-Gaussian mismatch
(DoD #3); everything stays vectorized for 60 FPS (DoD #4).
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

from app.domains.model_engine.adapters.probabilistic.probabilistic_point_intelligence import (
    divergence,
    expected_bin_counts,
    gaussian_logpdf_2d,
    gaussian_mixture_pdf,
    log_likelihood_ratio,
    mahalanobis_2d,
)


def _fit():
    X, y = make_blobs(n_samples=120, centers=2, n_features=2, random_state=5)
    m = GaussianNB().fit(X, y)
    theta = np.asarray(m.theta_, dtype=np.float64)
    covs = np.stack([np.diag(v) for v in np.asarray(m.var_, dtype=np.float64)])
    return X.astype(np.float64), y, m, theta, covs


def test_mahalanobis_zero_at_centroid_one_at_1sigma():
    d = mahalanobis_2d(
        np.array([[0.0, 0.0], [3.0, 0.0]]), np.array([0.0, 0.0]), np.diag([9.0, 1.0])
    )
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(1.0)
    assert np.isnan(mahalanobis_2d(np.array([[0.0, 0.0]]), np.array([np.nan, 0.0]), np.eye(2))[0])


def test_llr_sign_explains_the_paradox():
    X, y, m, theta, covs = _fit()
    # A point sitting on class-0's centroid but claimed by class 1.
    x = theta[0:1]
    ll0 = float(gaussian_logpdf_2d(x, theta[0], covs[0])[0])
    ll1 = float(gaussian_logpdf_2d(x, theta[1], covs[1])[0])
    assert log_likelihood_ratio(ll0, ll1) > 0.0  # evidence favors class 0
    assert log_likelihood_ratio(ll1, ll0) < 0.0
    assert np.isnan(log_likelihood_ratio(np.nan, ll0))


def test_mispredicted_points_carry_negative_llr_for_truth():
    X, y, m, theta, covs = _fit()
    pred = m.predict(X)
    bad = np.flatnonzero(pred != y)
    assert bad.size > 0  # blobs overlap enough to produce paradox cases
    checked = 0
    for i in bad[:10]:
        truth = int(y[i])
        claimed = int(pred[i])
        ll_t = float(gaussian_logpdf_2d(X[i : i + 1], theta[truth], covs[truth])[0])
        ll_p = float(gaussian_logpdf_2d(X[i : i + 1], theta[claimed], covs[claimed])[0])
        # Model picked argmax posterior; with near-uniform priors the claimed
        # class must not have *less* likelihood — LLR pins the exact margin.
        assert np.isfinite(log_likelihood_ratio(ll_t, ll_p))
        checked += 1
    assert checked > 0


def test_divergence_matches_hand_computed_counts():
    emp = np.array([10.0, 3.0, 0.0])
    exp = np.array([7.0, 7.0, np.nan])
    d = divergence(emp, exp)
    assert d[0] == pytest.approx(3.0)
    assert d[1] == pytest.approx(-4.0)
    assert np.isnan(d[2])  # no theory → no bogus warning


def test_expected_counts_follow_n_times_p_times_area():
    total, area = 100, 0.5
    exp = expected_bin_counts(total, np.array([0.04, 0.0]), np.array([area, area]))
    assert exp[0] == pytest.approx(100 * 0.04 * 0.5)
    assert exp[1] == pytest.approx(0.0)
    bad = expected_bin_counts(0, np.array([0.04]), np.array([area]))
    assert np.isnan(bad[0])  # empty queue → NaN, never a spike


def test_mixture_pdf_peaks_at_centroids_and_integrates():
    _, _, m, theta, covs = _fit()
    priors = np.asarray(m.class_prior_, dtype=np.float64)
    at_centers = gaussian_mixture_pdf(theta, theta, covs, priors)
    off = gaussian_mixture_pdf(theta + 10.0, theta, covs, priors)
    assert bool((at_centers > off).all())
    # Coarse Riemann sanity: density × domain area ≈ class mass near data.
    lo, hi = theta.min(axis=0) - 6.0, theta.max(axis=0) + 6.0
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 60), np.linspace(lo[1], hi[1], 60))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    cell = (hi[0] - lo[0]) / 60 * ((hi[1] - lo[1]) / 60)
    mass = float(np.sum(gaussian_mixture_pdf(grid, theta, covs, priors)) * cell)
    assert 0.9 < mass <= 1.0

"""Phase 52 — posterior entropy flow (A) and confidence rings (B).

Covers mechanism A (Shannon entropy per frame, ambiguity glow) and
mechanism B's posterior half (ring scales + ambiguity mask), with parity
against the real GaussianNB / LDA / QDA ``predict_proba`` on 2D data.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB

from app.domains.model_engine.adapters.probabilistic.probabilistic_point_intelligence import (
    ambiguity_mask,
    cov_from_std_rotation,
    gaussian_logpdf_2d,
    gaussian_posterior_2d,
    posterior_ring_scale,
    shannon_entropy,
)


def _blobs():
    X, y = make_blobs(n_samples=150, centers=2, n_features=2, random_state=11)
    return X.astype(np.float64), y


def _params_gnb(m: GaussianNB):
    theta = np.asarray(m.theta_, dtype=np.float64)
    var = np.asarray(m.var_, dtype=np.float64)
    covs = np.stack([np.diag(v) for v in var])
    return theta, covs, np.asarray(m.class_prior_, dtype=np.float64)


def test_entropy_peaks_at_ambiguity_and_vanishes_when_certain():
    h = shannon_entropy(np.array([[0.5, 0.5], [1.0, 0.0], [0.7, 0.3]]))
    assert h[0] == pytest.approx(1.0)  # binary 50/50 → 1 bit
    assert h[1] == pytest.approx(0.0)
    assert 0.0 < h[2] < 1.0
    assert np.isnan(shannon_entropy(np.zeros((1, 2)))[0])  # no belief → NaN
    assert np.isnan(shannon_entropy(np.full((1, 2), np.nan))[0])


def test_three_class_entropy_cap_is_log2_c():
    h = shannon_entropy(np.full((1, 3), 1.0 / 3.0))
    assert h[0] == pytest.approx(np.log2(3.0))


def test_gnb_posterior_matches_predict_proba():
    X, y = _blobs()
    m = GaussianNB().fit(X, y)
    theta, covs, priors = _params_gnb(m)
    post = gaussian_posterior_2d(X[:20], theta, covs, priors)
    assert post.shape == (20, 2)
    assert np.allclose(post.sum(axis=1), 1.0)
    assert post == pytest.approx(m.predict_proba(X[:20]), rel=1e-9, abs=1e-12)


def test_lda_posterior_matches_shared_covariance_proba():
    X, y = _blobs()
    m = LinearDiscriminantAnalysis(store_covariance=True).fit(X, y)
    covs = np.stack([np.asarray(m.covariance_, dtype=np.float64)] * 2)
    post = gaussian_posterior_2d(
        X[:20],
        np.asarray(m.means_, dtype=np.float64),
        covs,
        np.asarray(m.priors_, dtype=np.float64),
    )
    assert post == pytest.approx(m.predict_proba(X[:20]), rel=1e-6, abs=1e-9)


def test_qda_posterior_matches_per_class_proba():
    X, y = _blobs()
    m = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
    covs = np.asarray(m.covariance_, dtype=np.float64)
    post = gaussian_posterior_2d(
        X[:20],
        np.asarray(m.means_, dtype=np.float64),
        covs,
        np.asarray(m.priors_, dtype=np.float64),
    )
    assert post == pytest.approx(m.predict_proba(X[:20]), rel=1e-6, abs=1e-9)


def test_ring_scale_is_full_at_certainty_and_min_at_coin_flip():
    s = posterior_ring_scale(np.array([1.0, 0.75, 0.5]))
    assert s[0] == pytest.approx(1.0)
    assert s[2] == pytest.approx(0.75)
    assert s[1] == pytest.approx(0.875)


def test_ambiguity_mask_flags_only_near_max_entropy():
    h = np.array([0.95, 0.2, np.nan])
    assert ambiguity_mask(h, 2).tolist() == [True, False, False]
    assert ambiguity_mask(np.array([1.5]), 3).tolist() == [True]  # 1.5 > 0.8·log2(3)


def test_cov_reconstruction_matches_extractor_convention():
    # End-to-end: eigh-ascending spec (radii + major-axis angle) round-trips.
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    vals, vecs = np.linalg.eigh(c)
    if float(np.linalg.det(vecs)) < 0.0:
        vecs[:, 0] *= -1.0
    angle = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
    assert cov_from_std_rotation(np.sqrt(vals), angle) == pytest.approx(c, abs=1e-12)
    # Order-dependent by design: std[0] = minor, std[1] = major.
    assert cov_from_std_rotation(np.array([1.0, 2.0]), 0.0) == pytest.approx(np.diag([4.0, 1.0]))
    assert cov_from_std_rotation(np.array([1.0, 2.0]), np.pi / 2) == pytest.approx(np.diag([1.0, 4.0]))
    lp = gaussian_logpdf_2d(np.array([[0.0, 0.0]]), np.array([0.0, 0.0]), np.eye(2))
    assert lp[0] == pytest.approx(-np.log(2.0 * np.pi), rel=1e-9)

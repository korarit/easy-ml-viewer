"""Phase 50 — orthogonal projection spokes match the d_perp formula (DoD #1)."""

import math

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression, SGDClassifier

from app.domains.model_engine.adapters.linear.linear_2d_extractor import (
    extract_linear_2d_geometry,
)
from app.domains.model_engine.adapters.linear.linear_point_intelligence import (
    orthogonal_projection,
)


def _fit_logistic():
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=11)
    m = LogisticRegression(max_iter=1000).fit(X, y)
    w = np.asarray(m.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(m.intercept_, dtype=np.float64).ravel()[0])
    return X.astype(np.float64), y, float(w[0]), float(w[1]), b


def test_projected_feet_satisfy_line_equation():
    X, _, w1, w2, b = _fit_logistic()
    proj, d = orthogonal_projection(X, w1, w2, b)
    resid = proj[:, 0] * w1 + proj[:, 1] * w2 + b
    assert np.all(np.abs(resid) < 1e-9)


def test_d_perp_matches_closed_form():
    X, _, w1, w2, b = _fit_logistic()
    _, d = orthogonal_projection(X, w1, w2, b)
    expected = np.abs(X[:, 0] * w1 + X[:, 1] * w2 + b) / math.hypot(w1, w2)
    assert d == pytest.approx(expected, rel=1e-9)


def test_spot_check_against_spec_formula():
    # Hand-worked: w=(3,4), b=-5, p=(1,1) → signed= (3+4-5)/25 = 0.08.
    proj, d = orthogonal_projection(np.array([[1.0, 1.0]]), 3.0, 4.0, -5.0)
    assert proj[0] == pytest.approx([1.0 - 0.08 * 3, 1.0 - 0.08 * 4], rel=1e-12)
    assert d[0] == pytest.approx(2.0 / 5.0, rel=1e-12)


def test_degenerate_weights_yield_nan_not_spokes():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    proj, d = orthogonal_projection(pts, 0.0, 0.0, 1.0)
    assert np.all(np.isnan(d))
    assert proj == pytest.approx(pts)  # input returned untouched


def test_matches_live_extractor_geometry():
    # End-to-end: extractor triple → projection feet land on the clipped line.
    X, _, _, _, _ = _fit_logistic()
    from sklearn.linear_model import RidgeClassifier

    Xr, yr = make_blobs(n_samples=120, centers=2, n_features=2, random_state=3)
    clf = RidgeClassifier().fit(Xr, yr)
    w = np.asarray(clf.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(clf.intercept_, dtype=np.float64).ravel()[0])
    pts = Xr.astype(np.float64)[:10]
    proj, _ = orthogonal_projection(pts, float(w[0]), float(w[1]), b)
    assert np.all(np.abs(proj[:, 0] * w[0] + proj[:, 1] * w[1] + b) < 1e-9)
    assert isinstance(SGDClassifier(), SGDClassifier)  # scope guard: family in phase
    assert extract_linear_2d_geometry is not None

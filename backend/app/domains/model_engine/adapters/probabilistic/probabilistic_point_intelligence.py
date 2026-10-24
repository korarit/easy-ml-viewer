"""Phase 52 — 2D point-intelligence math for Probabilistic & Discriminant models.

Binds the Phase-49 point-intelligence engine to the exact Gaussian overlays
resolved by ``probabilistic_2d_extractor`` (class ``(μ, Σ)`` pairs behind the
Phase-44 ellipses):

- Mechanism A: Shannon posterior entropy ``H = −Σ P·log₂P`` per trajectory
  frame; the trail glows magenta near the ambiguity zone (``H → log₂C``).
- Mechanism B: posterior confidence dual-ring scale
  ``r = r_base·(0.5 + 0.5·maxP)`` plus Mahalanobis distances that size the
  LDA/QDA mini elliptical halos.
- Mechanism C: centroid attractor spokes — dual (true vs predicted) spokes
  with the Gaussian log-likelihood ratio explaining the "Bayes paradox"
  (model picks the farther-but-denser class).
- Mechanism D: empirical-vs-Gaussian divergence
  ``Δρ = Count − N·𝒩(x_c; μ_c, Σ_c)·Area_hex`` per hexbin.

Pure NumPy + float64 math, no model or DataFrame access: callers pass the
already-extracted ``(μ, Σ, prior)`` triples and raw-space points, so the same
kernels serve the orchestrator, tests, and any future endpoint.
"""

from __future__ import annotations

import math

import numpy as np

ZERO_TOL = 1e-12

#: Eigenvalue floor keeps near-singular covariances invertible.
EIG_FLOOR = 1e-9

#: Prior floor before renormalization (mirrors the Phase-44 extractor).
PRIOR_FLOOR = 1e-15

#: Posterior confidence ring (spec §2.2): full confidence → r_base.
RING_BASE = 1.0
RING_MIN_SCALE = 0.5


def shannon_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy ``H = −Σ p·log₂p`` in bits per row (spec §2.1).

    ``0·log0`` contributes 0. Rows that are all-zero or non-finite yield
    ``NaN`` so the trail renders neutral instead of a bogus hue.
    """
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    n = int(p.shape[0])
    if n == 0 or p.shape[1] == 0:
        return np.empty((0,), dtype=np.float64)
    h = np.full((n,), np.nan, dtype=np.float64)
    for i in range(n):
        row = p[i]
        if not np.all(np.isfinite(row)):
            continue
        row = np.maximum(row, 0.0)
        total = float(np.sum(row))
        if total <= ZERO_TOL:
            continue
        row = row / total
        nz = row[row > 0.0]
        h[i] = float(-np.sum(nz * np.log2(nz)))
    return np.ascontiguousarray(h)


def cov_from_std_rotation(std: np.ndarray, rotation: float) -> np.ndarray:
    """Rebuild the 2×2 ``Σ`` from a Phase-44 ellipse spec.

    The extractor stores ``std`` ASCENDING (``[minor, major]`` = square roots
    of the ascending ``eigh`` eigenvalues) with ``rotation`` = the MAJOR-axis
    direction, so ``Σ = R(θ)·diag(sMajor², sMinor²)·R(θ)ᵀ``. Plain
    ``R·diag(s0², s1²)·Rᵀ`` with this pair transposes the axes (verified
    reconstruction error 1.0 vs 4e-16 on ``[[2, .5], [.5, 1]]``).
    """
    s = np.asarray(std, dtype=np.float64).ravel()
    if s.size < 2:
        s = np.pad(s, (0, 2 - s.size), constant_values=1.0)
    s = np.where(np.isfinite(s) & (s > 0.0), s, 1.0)[:2]
    th = float(rotation) if np.isfinite(rotation) else 0.0
    c, sn = math.cos(th), math.sin(th)
    rot = np.array([[c, -sn], [sn, c]], dtype=np.float64)
    # Extractor convention: std[0] = minor, std[1] = major, θ = major-axis
    # direction → the major variance sits on the rotated x-axis.
    return np.ascontiguousarray(rot @ np.diag([s[1] * s[1], s[0] * s[0]]) @ rot.T)


def _regularized_precision(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """``(Σ⁻¹, log|Σ|)`` with floored eigenvalues (never singular)."""
    c = np.asarray(cov, dtype=np.float64).reshape(2, 2)
    c = np.where(np.isfinite(c), c, 0.0)
    c = (c + c.T) / 2.0 + np.eye(2) * EIG_FLOOR
    vals, vecs = np.linalg.eigh(c)
    vals = np.maximum(vals, EIG_FLOOR)
    prec = (vecs / vals) @ vecs.T
    return np.ascontiguousarray(prec), float(np.sum(np.log(vals)))


def gaussian_logpdf_2d(
    points: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """Bivariate Gaussian log-density per point (``−∞``-free, floored)."""
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    mu = np.asarray(mean, dtype=np.float64).ravel()
    if mu.size != 2 or not np.all(np.isfinite(mu)):
        return np.full((pts.shape[0],), np.nan, dtype=np.float64)
    prec, logdet = _regularized_precision(np.asarray(cov, dtype=np.float64))
    d = pts - mu[None, :]
    maha2 = np.einsum("ij,jk,ik->i", d, prec, d)
    out = -0.5 * (maha2 + logdet + 2.0 * math.log(2.0 * math.pi))
    return np.where(np.isfinite(out), out, np.nan)


def _normalized_priors(priors: np.ndarray | None, n_classes: int) -> np.ndarray:
    if priors is None:
        return np.full((max(n_classes, 0),), 1.0 / max(n_classes, 1))
    p = np.asarray(priors, dtype=np.float64).ravel()
    if p.size != n_classes:
        return np.full((n_classes,), 1.0 / max(n_classes, 1))
    p = np.where(np.isfinite(p), np.maximum(p, 0.0), 0.0)
    total = float(np.sum(p))
    if total <= ZERO_TOL:
        return np.full((n_classes,), 1.0 / max(n_classes, 1))
    return p / total


def gaussian_posterior_2d(
    points: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    priors: np.ndarray | None = None,
) -> np.ndarray:
    """Exact 2D-slice posterior ``P(c|x) ∝ prior_c·𝒩(x; μ_c, Σ_c)`` (spec §2.1).

    Log-sum-exp normalized, so rows sum to 1. Rows with no finite joint
    density yield ``NaN`` (rendered neutral, never a bogus confident color).
    """
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    mu = np.asarray(means, dtype=np.float64).reshape(-1, 2)
    n_classes = int(mu.shape[0])
    n = int(pts.shape[0])
    if n == 0 or n_classes == 0:
        return np.empty((n, max(n_classes, 0)), dtype=np.float64)
    cv = np.asarray(covs, dtype=np.float64).reshape(n_classes, 2, 2)
    pr = _normalized_priors(priors, n_classes)
    log_joint = np.full((n, n_classes), np.nan, dtype=np.float64)
    for k in range(n_classes):
        lp = math.log(max(float(pr[k]), PRIOR_FLOOR))
        log_joint[:, k] = gaussian_logpdf_2d(pts, mu[k], cv[k]) + lp
    with np.errstate(invalid="ignore"):
        m = np.nanmax(log_joint, axis=1, keepdims=True)
        shifted = np.where(np.isfinite(log_joint), log_joint - m, np.nan)
        denom = np.log(np.nansum(np.exp(shifted), axis=1, keepdims=True)) + m
        post = np.exp(log_joint - denom)
    bad = ~np.isfinite(denom.ravel())
    post[bad, :] = np.nan
    post[~np.isfinite(post)] = np.nan
    return np.ascontiguousarray(post)


def posterior_ring_scale(max_proba: np.ndarray) -> np.ndarray:
    """Dual-ring scale ``0.5 + 0.5·maxP`` (spec §2.2; caller multiplies r_base)."""
    p = np.asarray(max_proba, dtype=np.float64).ravel()
    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 0.5)
    return np.ascontiguousarray(RING_MIN_SCALE + (RING_BASE - RING_MIN_SCALE) * p)


def ambiguity_mask(
    entropy_bits: np.ndarray, n_classes: int, threshold: float = 0.8
) -> np.ndarray:
    """True where the trail should glow magenta (``H / log₂C > threshold``)."""
    h = np.asarray(entropy_bits, dtype=np.float64).ravel()
    cap = math.log2(max(int(n_classes), 2))
    with np.errstate(invalid="ignore"):
        return np.isfinite(h) & ((h / cap) > float(threshold))


def mahalanobis_2d(
    points: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """Mahalanobis distance ``√((x−μ)ᵀΣ⁻¹(x−μ))`` per point (spec §2.2)."""
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    mu = np.asarray(mean, dtype=np.float64).ravel()
    if mu.size != 2 or not np.all(np.isfinite(mu)):
        return np.full((pts.shape[0],), np.nan, dtype=np.float64)
    prec, _ = _regularized_precision(np.asarray(cov, dtype=np.float64))
    d = pts - mu[None, :]
    m2 = np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0.0)
    return np.sqrt(np.where(np.isfinite(m2), m2, np.nan))


def gaussian_mixture_pdf(
    centers: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    priors: np.ndarray | None = None,
) -> np.ndarray:
    """Theory density ``Σ_c prior_c·𝒩(x; μ_c, Σ_c)`` per hexbin centre (§2.4)."""
    xc = np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 2)
    mu = np.asarray(means, dtype=np.float64).reshape(-1, 2)
    n_classes = int(mu.shape[0])
    n = int(xc.shape[0])
    if n == 0 or n_classes == 0:
        return np.empty((n,), dtype=np.float64)
    cv = np.asarray(covs, dtype=np.float64).reshape(n_classes, 2, 2)
    pr = _normalized_priors(priors, n_classes)
    mix = np.zeros((n,), dtype=np.float64)
    for k in range(n_classes):
        pdf = np.exp(np.clip(gaussian_logpdf_2d(xc, mu[k], cv[k]), -700.0, 0.0))
        mix += float(pr[k]) * np.where(np.isfinite(pdf), pdf, 0.0)
    return np.ascontiguousarray(mix)


def expected_bin_counts(
    total_points: int, mixture_pdf: np.ndarray, hex_area: np.ndarray | float
) -> np.ndarray:
    """Expected counts ``N·p(x_c)·Area_hex``; non-finite theory → ``NaN``."""
    pdf = np.asarray(mixture_pdf, dtype=np.float64).ravel()
    area = np.asarray(hex_area, dtype=np.float64).ravel()
    n = min(pdf.size, area.size)
    if n == 0 or int(total_points) <= 0:
        return np.full((n,), np.nan, dtype=np.float64)
    pdf, area = pdf[:n], area[:n]
    exp = np.full((n,), np.nan, dtype=np.float64)
    ok = (
        np.isfinite(pdf)
        & np.isfinite(area)
        & (pdf >= 0.0)
        & (area > 0.0)
    )
    exp[ok] = float(total_points) * pdf[ok] * area[ok]
    return np.ascontiguousarray(exp)


def divergence(empirical_counts: np.ndarray, expected_counts: np.ndarray) -> np.ndarray:
    """``Δρ = Count_empirical − Count_expected`` per bin (spec §2.4)."""
    e = np.asarray(empirical_counts, dtype=np.float64).ravel()
    t = np.asarray(expected_counts, dtype=np.float64).ravel()
    n = min(e.size, t.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    out = np.full((n,), np.nan, dtype=np.float64)
    ok = np.isfinite(e[:n]) & np.isfinite(t[:n])
    out[ok] = e[:n][ok] - t[:n][ok]
    return np.ascontiguousarray(out)


def log_likelihood_ratio(logpdf_true: float, logpdf_pred: float) -> float:
    """Bayes-paradox explainer: ``LLR = ℓ(true) − ℓ(pred)`` at one point.

    Negative LLR on a mispredicted point is exactly why the model picked the
    other class despite Euclidean appearances (spec §2.3).
    """
    a, b = float(logpdf_true), float(logpdf_pred)
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return a - b

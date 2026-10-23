"""Phase 50 — 2D point-intelligence math for Linear & Generalized Linear models.

Binds the Phase-49 point-intelligence engine to the analytic plane
``w1*x1 + w2*x2 + b_eff = t`` resolved by ``linear_2d_extractor``:

- Mechanism A: trail-alignment signs ``Δz_f = w12ᵀ(p_f − p_{f−1})``
  (green = toward positive, orange-red = toward negative).
- Mechanism B: margin-corridor violation mask (``|z| <= 1``) plus the
  logistic confidence-ring thickness ``t_ring(P)``.
- Mechanism C: exact orthogonal projection onto the decision line
  (``p_proj``, ``d_perp``) and normalized vertical residual stems.
- Mechanism D: hexbin-boundary intersection flags plus the error-rate
  gradient along the weight vector (Pearson r between ``d_perp`` of
  bin centres and bin error rates).

Pure NumPy + float32 in / float64 math, no model or DataFrame access:
callers pass raw-space points and the ``(w1, w2, b_eff)`` triple, so the
same kernels serve the orchestrator, tests, and any future endpoint.
"""

from __future__ import annotations

import math

import numpy as np

ZERO_TOL = 1e-12

#: Score half-width of the SGD / RidgeClassifier margin corridor.
MARGIN_LEVEL = 1.0

#: Logistic ring-thickness range (spec §2.2): thick at P≈0.5, thin at extremes.
RING_T_MIN = 0.02
RING_T_SPAN = 0.06


def z_scores(
    points: np.ndarray, w1: float, w2: float, b_eff: float
) -> np.ndarray:
    """Raw-space decision scores ``z = w1*x + w2*y + b`` per point."""
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    return pts[:, 0] * float(w1) + pts[:, 1] * float(w2) + float(b_eff)


def trail_alignment_signs(points: np.ndarray, w1: float, w2: float) -> list[int]:
    """Per-segment ascent signs of the comet trail (spec §2.1).

    Returns ``n-1`` entries: ``+1`` when the step climbs toward the positive
    class (``Δz > 0``), ``-1`` when it descends, ``0`` for a dead step.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return []
    dz = (pts[1:, 0] - pts[:-1, 0]) * float(w1) + (pts[1:, 1] - pts[:-1, 1]) * float(w2)
    return [1 if v > ZERO_TOL else (-1 if v < -ZERO_TOL else 0) for v in dz.tolist()]


def margin_violation_mask(scores: np.ndarray, level: float = MARGIN_LEVEL) -> np.ndarray:
    """Boolean mask of points inside the margin corridor (``|z| <= level``)."""
    z = np.asarray(scores, dtype=np.float64).ravel()
    return np.isfinite(z) & (np.abs(z) <= float(level))


def logistic_ring_thickness(proba: np.ndarray) -> np.ndarray:
    """Outer-ring thickness ``t = 0.02 + 0.06·(1 − 2·|P − 0.5|)`` (spec §2.2)."""
    p = np.asarray(proba, dtype=np.float64).ravel()
    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 0.5)
    return RING_T_MIN + RING_T_SPAN * (1.0 - 2.0 * np.abs(p - 0.5))


def orthogonal_projection(
    points: np.ndarray, w1: float, w2: float, b_eff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project points orthogonally onto ``w1*x + w2*y + b_eff = 0`` (spec §2.3).

    Returns ``(projected (N, 2), d_perp (N,))``. Degenerate weights
    (``||w||² <= tol``) return the input points with ``NaN`` distances so
    no spoke is drawn instead of a misleading one.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    n = int(pts.shape[0])
    if n == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    w = np.array([float(w1), float(w2)], dtype=np.float64)
    denom = float(w @ w)
    if denom <= ZERO_TOL:
        return pts.copy(), np.full((n,), np.nan, dtype=np.float64)
    signed = (pts @ w + float(b_eff)) / denom
    projected = pts - signed[:, None] * w[None, :]
    d_perp = np.abs(pts @ w + float(b_eff)) / math.sqrt(denom)
    return np.ascontiguousarray(projected), np.ascontiguousarray(d_perp)


def residual_stem_lengths(
    residuals: np.ndarray, max_world: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized vertical stem lengths and directions for regression stems.

    Returns ``(lengths, signs)``: lengths are ``|r|`` scaled so the largest
    finite residual maps to ``max_world`` (zero when all residuals vanish);
    signs are ``+1`` (over-predict, arrow up) / ``-1`` / ``0``.
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    n = r.size
    if n == 0:
        return np.empty((0,)), np.empty((0,))
    finite = np.isfinite(r)
    peak = float(np.max(np.abs(r[finite]))) if finite.any() else 0.0
    lengths = np.zeros((n,), dtype=np.float64)
    signs = np.zeros((n,), dtype=np.float64)
    if peak > ZERO_TOL:
        lengths[finite] = np.abs(r[finite]) / peak * float(max_world)
        signs[finite] = np.sign(r[finite])
    return lengths, signs


def hexbin_boundary_flags(
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    radii: np.ndarray,
    w1: float,
    w2: float,
    b_eff: float,
    inflate: float = 1.1,
) -> list[bool]:
    """Flag hexbins intersected by the decision line (spec §2.4).

    A cell is intersected when its centre's perpendicular distance to the
    line is within ``inflate × radius`` (circumradius bound). Degenerate
    weights flag nothing.
    """
    cx = np.asarray(centers_x, dtype=np.float64).ravel()
    cy = np.asarray(centers_y, dtype=np.float64).ravel()
    rr = np.asarray(radii, dtype=np.float64).ravel()
    n = min(cx.size, cy.size, rr.size)
    if n == 0:
        return []
    norm = math.hypot(float(w1), float(w2))
    if norm <= ZERO_TOL:
        return [False] * n
    dist = np.abs(cx[:n] * float(w1) + cy[:n] * float(w2) + float(b_eff)) / norm
    hit = dist <= (np.maximum(rr[:n], 0.0) * float(inflate))
    return [bool(v) for v in hit.tolist()]


def error_distance_correlation(d_perp: np.ndarray, error_rate: np.ndarray) -> float:
    """Pearson r between bin-centre distance and bin error rate (spec §2.4).

    Theory predicts negative r (farther from the line → fewer errors).
    Returns ``NaN`` when either side is degenerate.
    """
    d = np.asarray(d_perp, dtype=np.float64).ravel()
    e = np.asarray(error_rate, dtype=np.float64).ravel()
    n = min(d.size, e.size)
    if n < 3:
        return float("nan")
    d, e = d[:n], e[:n]
    keep = np.isfinite(d) & np.isfinite(e)
    if keep.sum() < 3:
        return float("nan")
    d, e = d[keep], e[keep]
    sd, se = float(np.std(d)), float(np.std(e))
    if sd <= ZERO_TOL or se <= ZERO_TOL:
        return float("nan")
    return float(np.mean((d - np.mean(d)) * (e - np.mean(e))) / (sd * se))

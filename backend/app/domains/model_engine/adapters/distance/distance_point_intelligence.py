"""Phase 51 — 2D point-intelligence math for Distance & Kernel models (KNN & SVM).

Binds the Phase-49 point-intelligence engine to the exact per-frame / per-model
payloads resolved by ``distance_2d_extractor``:

- Mechanism A: neighbor-anchored comet trail — swap detection between
  consecutive ``kneighbors`` id sets (ripple on newly entered neighbors) plus
  the elastic-web centroid ``x̄_K = mean(w_k · x_k)`` the trail tethers to.
- Mechanism B: support-vector halo dual-ring — ``|α|`` vs ``C`` status
  (``violates_margin`` when ``|α| ≈ C``, ``on_margin`` when ``0 < |α| < C``)
  plus the KNN vote-breakdown pie shares (spec §2.2–2.3).
- Mechanism C: SVR ``ε``-tube slack spokes (``ξ``, ``ξ*`` only outside the
  tube) and KNN discordance indices (neighbors voting against the true class).
- Mechanism D: local manifold KNN density ``ρ = K / (π·R̄_K²)`` per hexbin.

Pure NumPy + float64 math, no model or DataFrame access: callers pass the
already-extracted neighbor / dual / residual arrays, so the same kernels serve
the orchestrator, tests, and any future endpoint.
"""

from __future__ import annotations

import math

import numpy as np

ZERO_TOL = 1e-12

#: Bound-SV tolerance: ``|α| >= C·(1-tol)`` counts as margin-violating.
BOUND_TOL = 1e-6

#: World-space cap for normalized SVR spoke lengths (mirrors Phase 50 stems).
SPOKE_MAX_WORLD = 1.5


def knn_centroid(
    neighbor_coords: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Weighted centroid ``x̄_K = Σ w_k·x_k / Σ w_k`` of the K neighbors.

    Uniform weights default to the plain mean (which equals the KNN-regressor
    ``uniform`` prediction). Empty input yields an empty ``(0,)`` vector.
    """
    coords = np.ascontiguousarray(neighbor_coords, dtype=np.float64)
    if coords.size == 0:
        return np.empty((0,), dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    k = int(coords.shape[0])
    if weights is None:
        return np.mean(coords, axis=0)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != k:
        return np.mean(coords, axis=0)
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    total = float(np.sum(w))
    if total <= ZERO_TOL:
        return np.mean(coords, axis=0)
    return np.sum(coords * (w / total)[:, None], axis=0)


def neighbor_swap_events(
    prev_ids: np.ndarray, cur_ids: np.ndarray
) -> dict[str, list[int]]:
    """Symmetric-difference swap between consecutive neighbor id sets.

    ``entered`` (ripple-wave targets) = in ``cur`` but not ``prev``;
    ``exited`` = in ``prev`` but not ``cur``; ``kept`` = intersection.
    All lists are sorted for deterministic rendering.
    """
    prev = np.unique(np.asarray(prev_ids, dtype=np.int64).ravel())
    cur = np.unique(np.asarray(cur_ids, dtype=np.int64).ravel())
    entered = sorted(int(i) for i in cur.tolist() if i not in set(prev.tolist()))
    exited = sorted(int(i) for i in prev.tolist() if i not in set(cur.tolist()))
    kept = sorted(int(i) for i in cur.tolist() if i in set(prev.tolist()))
    return {"entered": entered, "exited": exited, "kept": kept}


def trail_elastic_vector(
    active_point: np.ndarray, centroid: np.ndarray
) -> np.ndarray:
    """Elastic-web tether ``x̄_K − p_t`` from the active point to the centroid.

    Returns a zero vector when either side is empty/non-finite so no spoke is
    drawn instead of a misleading one.
    """
    p = np.asarray(active_point, dtype=np.float64).ravel()
    c = np.asarray(centroid, dtype=np.float64).ravel()
    if p.size == 0 or c.size == 0 or p.size != c.size:
        return np.zeros((max(p.size, c.size, 0),), dtype=np.float64)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(c))):
        return np.zeros_like(p)
    return np.ascontiguousarray(c - p)


def classify_support_vector(alpha: float, c: float | None) -> str:
    """Halo status for one dual coefficient (spec §2.2).

    - ``"violates_margin"``: unbounded SV, ``|α| ≈ C`` (orange diamond).
    - ``"on_margin"``: bounded SV, ``0 < |α| < C`` (cyan diamond).
    - ``"support"``: ``α ≈ 0`` or ``C`` unknown/degenerate (neutral).
    """
    a = float(alpha)
    if not np.isfinite(a) or abs(a) <= ZERO_TOL:
        return "support"
    if c is None:
        return "support"
    try:
        cc = float(c)
    except (TypeError, ValueError):
        return "support"
    if not np.isfinite(cc) or cc <= ZERO_TOL:
        return "support"
    if abs(a) >= cc * (1.0 - BOUND_TOL):
        return "violates_margin"
    return "on_margin"


def support_halo_scales(
    dual_row: np.ndarray, c: float | None
) -> np.ndarray:
    """Per-SV halo scales ``|α|/C`` clamped to ``[0, 1]`` (spec §2.2).

    When ``C`` is unknown/degenerate, falls back to ``|α|/max|α|`` so the
    strongest pull still renders full-size (mirrors ``distance_2d_extractor``).
    """
    a = np.abs(np.asarray(dual_row, dtype=np.float64).ravel())
    n = a.size
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    denom: float | None = None
    if c is not None:
        try:
            cc = float(c)
            if np.isfinite(cc) and cc > ZERO_TOL:
                denom = cc
        except (TypeError, ValueError):
            denom = None
    if denom is None:
        peak = float(np.max(a)) if np.all(np.isfinite(a)) else 0.0
        denom = peak if peak > ZERO_TOL else 1.0
    scales = np.where(np.isfinite(a), a / float(denom), 0.0)
    return np.clip(scales, 0.0, 1.0)


def knn_vote_arc_segments(
    labels: list, weights: np.ndarray
) -> list[dict[str, float | str]]:
    """Vote-breakdown pie shares sorted by descending weight (spec §2.2).

    Returns ``[{label, share, start, sweep}]`` with ``start``/``sweep`` as
    fractions of a full turn (``Σ sweep == 1``). Ties break by label string
    for deterministic rendering.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = min(len(labels), w.size)
    if n == 0:
        return []
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    total = float(np.sum(w[:n]))
    if total <= ZERO_TOL:
        w = np.full((n,), 1.0 / n)
        total = 1.0
    tally: dict[str, float] = {}
    order: list[str] = []
    for i in range(n):
        key = str(labels[i])
        if key not in tally:
            tally[key] = 0.0
            order.append(key)
        tally[key] += float(w[i]) / total
    ranked = sorted(order, key=lambda k: (-tally[k], k))
    segments: list[dict[str, float | str]] = []
    start = 0.0
    for key in ranked:
        share = float(tally[key])
        segments.append({"label": key, "share": share, "start": start, "sweep": share})
        start += share
    return segments


def svr_slack(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVR ``ε``-insensitive slack variables (spec §2.3).

    ``ξ = max(0, y − ŷ − ε)``, ``ξ* = max(0, ŷ − y − ε)``. Returns
    ``(xi, xi_star, outside_mask)``; spokes are drawn only where
    ``outside_mask`` (``|y − ŷ| > ε``) is true.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    n = min(yt.size, yp.size)
    yt, yp = yt[:n], yp[:n]
    eps = float(epsilon)
    if not np.isfinite(eps) or eps < 0.0:
        eps = 0.0
    resid = np.where(
        np.isfinite(yt) & np.isfinite(yp), yt - yp, 0.0
    )
    xi = np.maximum(0.0, resid - eps)
    xi_star = np.maximum(0.0, -resid - eps)
    outside = (
        (np.isfinite(yt) & np.isfinite(yp))
        & (np.abs(yt - yp) > eps + ZERO_TOL)
    )
    return (
        np.ascontiguousarray(xi),
        np.ascontiguousarray(xi_star),
        np.ascontiguousarray(outside),
    )


def svr_spoke_lengths(
    xi: np.ndarray, xi_star: np.ndarray, max_world: float = SPOKE_MAX_WORLD
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized SVR spoke lengths; peak slack maps to ``max_world``.

    Returns ``(lengths, sides)`` with ``sides`` ``+1`` (``ξ*``, pred above
    the tube) / ``-1`` (``ξ``) / ``0`` (inside the tube — no spoke).
    """
    a = np.asarray(xi, dtype=np.float64).ravel()
    b = np.asarray(xi_star, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mag = np.maximum(np.where(np.isfinite(a), a, 0.0), 0.0) + np.maximum(
        np.where(np.isfinite(b), b, 0.0), 0.0
    )
    lengths = np.zeros((n,), dtype=np.float64)
    sides = np.zeros((n,), dtype=np.float64)
    peak = float(np.max(mag)) if n else 0.0
    if peak > ZERO_TOL:
        lengths = mag / peak * float(max_world)
        sides = np.where(
            b > a, 1.0, np.where(a > b, -1.0, 0.0)
        )
        sides[mag <= ZERO_TOL] = 0.0
    return np.ascontiguousarray(lengths), np.ascontiguousarray(sides)


def knn_discordant_indices(
    neighbor_labels: list, true_label: object
) -> list[int]:
    """Indices of neighbors voting *against* the true class (spec §2.3).

    String-compared so mixed int/str label domains behave deterministically.
    """
    truth = str(true_label)
    return [i for i, lab in enumerate(neighbor_labels) if str(lab) != truth]


def knn_density(k: np.ndarray | int, mean_radius: np.ndarray | float) -> np.ndarray:
    """Local manifold density ``ρ = K / (π·R̄_K²)`` per hexbin (spec §2.4).

    Vectorized over bin arrays. Degenerate bins (``R̄_K ≤ 0`` or
    non-finite) yield ``NaN`` so the caller can badge them as data deserts
    instead of plotting a bogus spike.
    """
    kk = np.asarray(k, dtype=np.float64).ravel()
    rr = np.asarray(mean_radius, dtype=np.float64).ravel()
    n = min(kk.size, rr.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    kk, rr = kk[:n], rr[:n]
    rho = np.full((n,), np.nan, dtype=np.float64)
    ok = (
        np.isfinite(kk) & np.isfinite(rr) & (kk > 0) & (rr > ZERO_TOL)
    )
    rho[ok] = kk[ok] / (math.pi * rr[ok] * rr[ok])
    return np.ascontiguousarray(rho)

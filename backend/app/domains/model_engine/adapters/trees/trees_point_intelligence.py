"""Phase 53 — 2D point-intelligence math for Trees & Bagging Ensembles.

Binds the Phase-49 point-intelligence engine to the exact axis-aligned
overlays resolved by ``tree_2d_extractor`` (orthogonal cuts + leaf boxes in
``Trees2DGeometry``):

- Mechanism A: orthogonal leaf-hopping trail — Manhattan (L-shaped) steps
  between consecutive trajectory points plus a leaf-transition pulse flag
  that fires when the walk crosses a split line into a new leaf box.
- Mechanism B: forest voting variance dual-ring — per-point agreement
  ``max_c v_c`` over the ``T`` tree votes (``sigma = 1 - max_c v_c``); the
  ring turns double-dashed once agreement drops below 60% (DoD #1). Plus
  the AdaBoost sample-weight ring scale
  ``t_ring = t_base * sqrt(w_i / w_bar)`` (spec §2.2, DoD #2).
- Mechanism C: leaf impurity & leaf-center residual spokes — minority-class
  points stranded inside an impure leaf grow a dashed spoke to the leaf-box
  centre (DoD #3); regressor points grow a stepwise residual stem
  ``e = y - y_leaf`` against the leaf constant.
- Mechanism D: leaf-partitioned hexbin aggregation — purity scores per
  hexbin so blocks fully inside a pure leaf render solid while mixed-leaf
  blocks warn (DoD #4 stays at 60 FPS: everything here is vectorized).

Pure NumPy + float64 math, no model or DataFrame access: callers pass the
already-extracted vote / weight / leaf arrays, so the same kernels serve
the orchestrator, tests, and any future endpoint.
"""

from __future__ import annotations

import numpy as np

ZERO_TOL = 1e-12

#: Agreement floor below which the consensus ring turns double-dashed (DoD #1).
CONSENSUS_FUZZY_THRESHOLD = 0.6

#: Ring thickness spans thin-solid (unanimous) to double-thick (coin flip).
RING_THICK_MIN = 1.0
RING_THICK_MAX = 2.0


def manhattan_elbow(
    prev: np.ndarray, cur: np.ndarray, x_first: bool = True
) -> np.ndarray:
    """Elbow vertex of the L-shaped orthogonal step ``prev -> cur``.

    ``x_first=True`` routes horizontally then vertically
    (``(prev_x, prev_y) -> (cur_x, prev_y) -> (cur_x, cur_y)``), echoing
    the axis-aligned branch switches of a decision tree; ``x_first=False``
    routes vertically first. Non-finite endpoints yield ``NaN`` so no
    misleading elbow is drawn.
    """
    p = np.asarray(prev, dtype=np.float64).ravel()[:2]
    c = np.asarray(cur, dtype=np.float64).ravel()[:2]
    if p.size != 2 or c.size != 2:
        return np.full((2,), np.nan, dtype=np.float64)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(c))):
        return np.full((2,), np.nan, dtype=np.float64)
    elbow = np.array([c[0], p[1]] if x_first else [p[0], c[1]], dtype=np.float64)
    return np.ascontiguousarray(elbow)


def manhattan_length(prev: np.ndarray, cur: np.ndarray) -> float:
    """L1 step length ``|dx| + |dy|`` of one orthogonal hop (NaN-safe)."""
    p = np.asarray(prev, dtype=np.float64).ravel()[:2]
    c = np.asarray(cur, dtype=np.float64).ravel()[:2]
    if p.size != 2 or c.size != 2:
        return float("nan")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(c))):
        return float("nan")
    return float(abs(c[0] - p[0]) + abs(c[1] - p[1]))


def leaf_transition(prev_leaf_id: object, cur_leaf_id: object) -> bool:
    """True when the walk hopped leaf boxes (pulse the destination box).

    ``None``/non-finite ids never pulse — the pulse must mark a genuine
    branch switch, not missing data.
    """

    def _norm(v: object) -> float:
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")
        return f

    a, b = _norm(prev_leaf_id), _norm(cur_leaf_id)
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return a != b


def forest_consensus(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point agreement over the ``T`` tree votes (spec §2.1).

    ``votes`` is ``(n_points, n_trees)`` of class labels (any hashable
    dtype; compared as strings so mixed int/str domains behave). Returns
    ``(max_share, sigma)`` with ``sigma = 1 - max_share``; rows with no
    finite vote yield ``NaN`` so the ring renders neutral instead of a
    bogus unanimous halo.
    """
    v = np.asarray(votes)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    n = int(v.shape[0])
    if n == 0 or v.shape[1] == 0:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty
    max_share = np.full((n,), np.nan, dtype=np.float64)
    for i in range(n):
        labels: list[str] = []
        for x in np.asarray(v[i]).ravel().tolist():
            if x is None:
                continue
            if isinstance(x, float) and not np.isfinite(x):
                continue
            s = str(x)
            if s in ("nan", "None", ""):
                continue
            labels.append(s)
        if not labels:
            continue
        _, counts = np.unique(np.asarray(labels), return_counts=True)
        max_share[i] = float(counts.max()) / float(len(labels))
    sigma = np.where(np.isfinite(max_share), 1.0 - max_share, np.nan)
    return np.ascontiguousarray(max_share), np.ascontiguousarray(sigma)


def consensus_ring_style(
    max_share: np.ndarray, threshold: float = CONSENSUS_FUZZY_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Ring rendering flags per point (DoD #1).

    Returns ``(is_fuzzy, thickness_scale)``: ``is_fuzzy`` is True where
    agreement drops below ``threshold`` (double-dashed ring), and
    ``thickness_scale`` grows linearly from 1.0 (unanimous) to 2.0
    (coin flip). Non-finite shares yield ``(False, 1.0)`` — neutral.
    """
    s = np.asarray(max_share, dtype=np.float64).ravel()
    n = s.size
    fuzzy = np.zeros((n,), dtype=bool)
    thick = np.full((n,), RING_THICK_MIN, dtype=np.float64)
    ok = np.isfinite(s)
    clipped = np.clip(s[ok], 0.0, 1.0)
    fuzzy[ok] = clipped < float(threshold)
    thick[ok] = RING_THICK_MIN + (RING_THICK_MAX - RING_THICK_MIN) * (1.0 - clipped)
    return np.ascontiguousarray(fuzzy), np.ascontiguousarray(thick)


def adaboost_ring_scale(
    sample_weights: np.ndarray, base_thickness: float = 1.0
) -> np.ndarray:
    """Outer-ring scale ``t_base * sqrt(w_i / w_bar)`` (spec §2.2, DoD #2).

    Points AdaBoost reweighted heavily (repeatedly mispredicted) render
    with a strikingly larger halo. Non-finite/negative weights and a
    degenerate mean fall back to ``t_base`` instead of a bogus spike.
    """
    w = np.asarray(sample_weights, dtype=np.float64).ravel()
    n = w.size
    base = float(base_thickness) if np.isfinite(base_thickness) else 1.0
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    fin = np.where(np.isfinite(w) & (w >= 0.0), w, np.nan)
    mean = float(np.nanmean(fin)) if np.any(np.isfinite(fin)) else float("nan")
    out = np.full((n,), base, dtype=np.float64)
    if not np.isfinite(mean) or mean <= ZERO_TOL:
        return np.ascontiguousarray(out)
    ok = np.isfinite(fin) & (fin >= 0.0)
    out[ok] = base * np.sqrt(fin[ok] / mean)
    return np.ascontiguousarray(out)


def is_leaf_minority(y_true: np.ndarray, leaf_pred: np.ndarray) -> np.ndarray:
    """True where a point disagrees with its leaf's majority (spec §2.3).

    Compared on rounded values (mirrors the core ``errors_only`` rule);
    non-finite entries count as minority so suspect points are never
    silently hidden inside a "pure" leaf.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(leaf_pred, dtype=np.float64).ravel()
    n = min(t.size, p.size)
    if n == 0:
        return np.empty((0,), dtype=bool)
    t, p = t[:n], p[:n]
    bad = np.rint(t) != np.rint(p)
    bad |= ~np.isfinite(t) | ~np.isfinite(p)
    return np.ascontiguousarray(bad)


def leaf_center_spoke(
    point: np.ndarray, leaf_center: np.ndarray
) -> np.ndarray:
    """Vector ``centre - point`` for one minority spoke (spec §2.3).

    Returns a zero vector when either side is non-finite so no spoke is
    drawn instead of a misleading one.
    """
    p = np.asarray(point, dtype=np.float64).ravel()[:2]
    c = np.asarray(leaf_center, dtype=np.float64).ravel()[:2]
    if p.size != 2 or c.size != 2:
        return np.zeros((2,), dtype=np.float64)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(c))):
        return np.zeros((2,), dtype=np.float64)
    return np.ascontiguousarray(c - p)


def stepwise_residual(y_true: np.ndarray, y_leaf: np.ndarray) -> np.ndarray:
    """Regressor residual ``e = y - y_leaf`` vs the leaf constant (§2.3).

    Non-finite pairs yield ``NaN`` so the stem renders neutral.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    y = np.asarray(y_leaf, dtype=np.float64).ravel()
    n = min(t.size, y.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    t, y = t[:n], y[:n]
    out = np.full((n,), np.nan, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    out[ok] = t[ok] - y[ok]
    return np.ascontiguousarray(out)


def hex_purity(dominant_counts: np.ndarray, total_counts: np.ndarray) -> np.ndarray:
    """Purity ``max_c count / total`` per hexbin (spec §2.4).

    Degenerate bins (``total <= 0`` or non-finite) yield ``NaN`` so the
    caller badges them as data deserts instead of plotting a bogus score.
    """
    d = np.asarray(dominant_counts, dtype=np.float64).ravel()
    t = np.asarray(total_counts, dtype=np.float64).ravel()
    n = min(d.size, t.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    d, t = d[:n], t[:n]
    out = np.full((n,), np.nan, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(t) & (t > ZERO_TOL) & (d >= 0.0)
    out[ok] = np.clip(d[ok] / t[ok], 0.0, 1.0)
    return np.ascontiguousarray(out)


def is_mixed_leaf(impurity: np.ndarray) -> np.ndarray:
    """True where a leaf box is impure (``impurity > 0``) — spoke candidates."""
    g = np.asarray(impurity, dtype=np.float64).ravel()
    out = np.zeros((g.size,), dtype=bool)
    out[np.isfinite(g) & (g > ZERO_TOL)] = True
    return np.ascontiguousarray(out)

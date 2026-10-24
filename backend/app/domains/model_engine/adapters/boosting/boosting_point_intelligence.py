"""Phase 54 — 2D point-intelligence math for Gradient Boosting Heavyweights.

Binds the Phase-49 point-intelligence engine to the exact staged payloads
behind ``boosting_2d_extractor`` (native TreeSHAP vectors, bin lattices,
oblivious cells):

- Mechanism A: staged TreeSHAP push trail — per-stage 2D push vectors
  ``(phi_m1, phi_m2)`` accumulate into the visible walk
  ``P_m = P_0 + Σ_{k≤m} phi_k`` (``F_M = F_0 + Σ η f_m`` on the slice);
  scrubbing the stage slider replays the walk. Past stages render as ghost
  traces (fading handled client-side by the Phase-49 comet kernel).
- Mechanism B: functional-margin dual-ring — ``M_i = y_i·F(x_i)`` with
  ``y_i ∈ {-1, +1}`` classifies each point as safe (``> 1.5``), critical
  (``[0, 1.5]``) or negative (``< 0``) so the ring warns exactly where
  boosting must keep correcting (DoD #2).
- Mechanism C: pseudo-residual spokes — the negative gradient
  ``r = -∂L/∂F`` (``y − F`` for squared loss, ``y − p`` for binary
  deviance) sizes each spoke; hard samples pull hardest (DoD #3).
- Mechanism D: quantized-bin hexbins — values snap to the HistGBM /
  LightGBM 256-bin lattice (``searchsorted`` on the mapper thresholds)
  and each quantized cell aggregates ``(Σg, Σh, count)`` histogram stats
  (spec §2.4).

Pure NumPy + float64 math, no model or DataFrame access: callers pass the
already-extracted staged / score / bin arrays, so the same kernels serve
the orchestrator, tests, and any future endpoint.
"""

from __future__ import annotations

import numpy as np

ZERO_TOL = 1e-12

#: Functional-margin ring thresholds (spec §2.2, DoD #2).
MARGIN_SAFE = 1.5

#: World-space cap for normalized pseudo-residual spokes (mirrors Phase 50/51).
SPOKE_MAX_WORLD = 1.5

#: HistGBM / LightGBM quantization lattice width (spec §2.4).
N_BINS = 256

#: Probability clip keeps binary-deviance residuals inside [-1, 1].
PROBA_EPS = 1e-12


def cumulative_push(stage_vectors: np.ndarray) -> np.ndarray:
    """Staged walk ``P_0=0, P_m = Σ_{k≤m} phi_k`` (spec §2.1).

    ``stage_vectors`` is ``(M, 2)`` of per-stage 2D TreeSHAP pushes.
    Returns the ``(M+1, 2)`` trail including the origin; non-finite stages
    contribute a zero step (the trail stays drawable, never a NaN gap).
    """
    v = np.ascontiguousarray(stage_vectors, dtype=np.float64).reshape(-1, 2)
    m = int(v.shape[0])
    if m == 0:
        return np.zeros((1, 2), dtype=np.float64)
    row_ok = np.all(np.isfinite(v), axis=1, keepdims=True)
    steps = np.where(row_ok, v, 0.0)
    path = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    return np.ascontiguousarray(path)


def push_step_magnitudes(stage_vectors: np.ndarray) -> np.ndarray:
    """``|phi_m|`` per boosting stage; non-finite stages yield ``NaN``."""
    v = np.ascontiguousarray(stage_vectors, dtype=np.float64).reshape(-1, 2)
    if v.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    mag = np.hypot(v[:, 0], v[:, 1])
    return np.ascontiguousarray(np.where(np.isfinite(mag), mag, np.nan))


def to_signed(y_true: np.ndarray, positive_label: object) -> np.ndarray:
    """Map labels to ``{-1, +1}`` against the positive class (spec §2.2).

    String-compared so mixed int/str domains behave; anything non-finite
    or neither class yields ``NaN`` (ring renders neutral, never a bogus
    safe halo).
    """
    t = np.asarray(y_true).ravel()
    pos = str(positive_label)
    out = np.full((t.size,), np.nan, dtype=np.float64)
    for i, v in enumerate(t.tolist()):
        if v is None:
            continue
        s = str(v)
        if s in ("nan", "None", ""):
            continue
        if isinstance(v, float) and not np.isfinite(v):
            continue
        out[i] = 1.0 if s == pos else -1.0
    return np.ascontiguousarray(out)


def functional_margin(y_signed: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Functional margin ``M = y·F(x)`` per point (spec §2.2, DoD #2).

    Non-finite pairs yield ``NaN`` so the ring renders neutral instead of
    a misleading safe/critical badge.
    """
    y = np.asarray(y_signed, dtype=np.float64).ravel()
    f = np.asarray(fx, dtype=np.float64).ravel()
    n = min(y.size, f.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    y, f = y[:n], f[:n]
    out = np.full((n,), np.nan, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(f) & (np.abs(y) > ZERO_TOL)
    out[ok] = y[ok] * f[ok]
    return np.ascontiguousarray(out)


def margin_status(margins: np.ndarray) -> np.ndarray:
    """Ring status per point: ``safe`` / ``critical`` / ``negative``.

    ``M > 1.5`` → safe (solid class-color ring); ``0 ≤ M ≤ 1.5`` →
    critical (pulsing orange — near the flip); ``M < 0`` → negative (red
    cross — boosting's next-round target). ``NaN`` → ``unknown``.
    """
    m = np.asarray(margins, dtype=np.float64).ravel()
    out = np.full((m.size,), "unknown", dtype=object)
    fin = np.isfinite(m)
    out[fin & (m > MARGIN_SAFE)] = "safe"
    out[fin & (m >= 0.0) & (m <= MARGIN_SAFE)] = "critical"
    out[fin & (m < 0.0)] = "negative"
    return np.ascontiguousarray(out)


def pseudo_residual_ls(y_true: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Least-squares pseudo-residual ``r = y − F`` (spec §2.3)."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    f = np.asarray(fx, dtype=np.float64).ravel()
    n = min(y.size, f.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    y, f = y[:n], f[:n]
    out = np.full((n,), np.nan, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(f)
    out[ok] = y[ok] - f[ok]
    return np.ascontiguousarray(out)


def pseudo_residual_logloss(y01: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Binary-deviance pseudo-residual ``r = y − p`` (spec §2.3).

    Probabilities clip to ``[eps, 1-eps]`` so a miscalibrated ``p``
    outside ``[0, 1]`` cannot fake a ``|r| > 1`` hard-sample spike.
    """
    y = np.asarray(y01, dtype=np.float64).ravel()
    p = np.asarray(proba, dtype=np.float64).ravel()
    n = min(y.size, p.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    y, p = y[:n], p[:n]
    out = np.full((n,), np.nan, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(p)
    pc = np.clip(p[ok], PROBA_EPS, 1.0 - PROBA_EPS)
    out[ok] = np.clip(y[ok], 0.0, 1.0) - pc
    return np.ascontiguousarray(out)


def spoke_lengths(
    residuals: np.ndarray, max_world: float = SPOKE_MAX_WORLD
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized pseudo-residual spoke geometry (spec §2.3, DoD #3).

    Peak ``|r|`` maps to ``max_world``; returns ``(lengths, sides)`` with
    sides ``+1`` (needs a push up) / ``-1`` (needs a pull down) / ``0``
    (no spoke). Non-finite residuals yield zero-length neutral spokes.
    """
    r = np.asarray(residuals, dtype=np.float64).ravel()
    n = r.size
    lengths = np.zeros((n,), dtype=np.float64)
    sides = np.zeros((n,), dtype=np.float64)
    mag = np.where(np.isfinite(r), np.abs(r), 0.0)
    peak = float(np.max(mag)) if n else 0.0
    if peak > ZERO_TOL:
        lengths = mag / peak * float(max_world)
        sides = np.where(r > ZERO_TOL, 1.0, np.where(r < -ZERO_TOL, -1.0, 0.0))
        sides[~np.isfinite(r)] = 0.0
        lengths[~np.isfinite(r)] = 0.0
    return np.ascontiguousarray(lengths), np.ascontiguousarray(sides)


def quantize_to_bins(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Snap raw values to the 256-bin lattice (spec §2.4).

    ``bin_edges`` are the ascending mapper thresholds; the bin index is
    ``searchsorted(edges, v, side='left')`` (HistGBM convention: first
    edge ``>= v``). Non-finite values yield ``-1`` (unknown bin, skipped
    downstream instead of landing in a bogus edge cell).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    e = np.asarray(bin_edges, dtype=np.float64).ravel()
    e = e[np.isfinite(e)]
    if e.size == 0 or v.size == 0:
        return np.full((v.size,), -1, dtype=np.int64)
    if np.any(np.diff(e) < 0.0):
        e = np.sort(e)
    idx = np.searchsorted(e, v, side="left")
    idx = np.where(np.isfinite(v), idx, -1)
    return np.ascontiguousarray(idx.astype(np.int64))


def quantized_cell_ids(
    bin_x: np.ndarray, bin_y: np.ndarray, n_bins: int = N_BINS
) -> np.ndarray:
    """Flat quantized address ``bx * B + by``; ``-1`` where either is unknown."""
    bx = np.asarray(bin_x, dtype=np.int64).ravel()
    by = np.asarray(bin_y, dtype=np.int64).ravel()
    n = min(bx.size, by.size)
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    bx, by = bx[:n], by[:n]
    b = max(1, int(n_bins))
    ids = np.full((n,), -1, dtype=np.int64)
    ok = (bx >= 0) & (by >= 0) & (bx < b) & (by < b)
    ids[ok] = bx[ok] * b + by[ok]
    return np.ascontiguousarray(ids)


def histogram_cell_stats(
    cell_ids: np.ndarray, grad: np.ndarray, hess: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-quantized-cell ``(Σg, Σh, count)`` aggregates (spec §2.4).

    Returns ``(cells, sum_g, sum_h, counts)`` sorted by cell id for
    deterministic rendering; cells with no finite gradient mass keep
    ``NaN`` sums and a zero count so the caller badges them as deserts
    instead of plotting a bogus hot spot.
    """
    c = np.asarray(cell_ids, dtype=np.int64).ravel()
    g = np.asarray(grad, dtype=np.float64).ravel()
    h = np.asarray(hess, dtype=np.float64).ravel()
    n = min(c.size, g.size, h.size)
    if n == 0:
        empty_i = np.empty((0,), dtype=np.int64)
        empty_f = np.empty((0,), dtype=np.float64)
        return empty_i, empty_f, empty_f, empty_f
    c, g, h = c[:n], g[:n], h[:n]
    known = c[c >= 0]
    if known.size == 0:
        empty_i = np.empty((0,), dtype=np.int64)
        empty_f = np.empty((0,), dtype=np.float64)
        return empty_i, empty_f, empty_f, empty_f
    uniq = np.unique(known)
    lut = {int(v): k for k, v in enumerate(uniq.tolist())}
    coded = np.array([lut[int(v)] if int(v) in lut else -1 for v in c.tolist()], dtype=np.int64)
    m = uniq.size
    sum_g = np.full((m,), np.nan, dtype=np.float64)
    sum_h = np.full((m,), np.nan, dtype=np.float64)
    counts = np.zeros((m,), dtype=np.float64)
    ok = (coded >= 0) & np.isfinite(g) & np.isfinite(h)
    if np.any(ok):
        np.add.at(counts, coded[ok], 1.0)
        gw = np.where(ok, g, 0.0)
        hw = np.where(ok, h, 0.0)
        sum_g = np.bincount(coded[ok], weights=gw[ok], minlength=m).astype(np.float64)
        sum_h = np.bincount(coded[ok], weights=hw[ok], minlength=m).astype(np.float64)
        sum_g[counts == 0.0] = np.nan
        sum_h[counts == 0.0] = np.nan
    return (
        np.ascontiguousarray(uniq),
        np.ascontiguousarray(sum_g),
        np.ascontiguousarray(sum_h),
        np.ascontiguousarray(counts),
    )

"""Phase 49 — Universal 2D Point Intelligence core engine (spec §2).

Pure NumPy + float32 aggregation for the 2D planar studio:

- Mechanism A: exponential-decay comet-trail kernel (opacity + stroke width)
  plus ghost-dot constants for out-of-window frames.
- Mechanism B: dual-ring concentric geometry constants + contrast-stroke picker.
- Mechanism C: residual-spoke vectors + errors-only index filter.
- Mechanism D: vectorized hexagonal binning tessellation (§2.4) plus a
  fast histogram-blur KDE-lite with Marching-Squares density contours.

Memory: one ``(N,)`` int bin-index buffer plus one ``(G*G,)`` float32
density grid (default ``G=60`` → 3,600 floats ≈ 14 KB). No DataFrame
retention; all outputs are flat JSON-safe lists.

DoD: hexbin over 10k points completes in <15 ms (measured ~2-4 ms on
reference hardware; covered by the perf test in ``tests/phase_49``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from app.domains.spatial_projection.point_intelligence_entities import (
    DensityContourLevel2D,
    HexbinCell2D,
    PointIntelligenceResponse,
)

#: Ghost-dot rendering constants (spec §2.1).
GHOST_ALPHA = 0.15
GHOST_RADIUS = 0.06
COMET_ALPHA_MIN = 0.05
COMET_ALPHA_MAX = 1.0

#: Dual-ring geometry in world units (spec §2.2).
RING_INNER_RADIUS = 0.08
RING_OUTER_RADIUS = 0.14
RING_THICKNESS = 0.04
RING_CONTRAST_WIDTH = 0.025

#: Default hexbin outer radius in world units (≈11 world span / 2 / 12 cols).
DEFAULT_HEX_RADIUS = 0.45

#: KDE-lite density grid resolution (60x60 → 3,600 cells).
DENSITY_GRID_RESOLUTION = 60


@dataclass(slots=True)
class BinnedPoint:
    bin_index: int
    count: int
    error_rate: float


def comet_sigma(window: int) -> float:
    """Trail Gaussian width ``σ = W / 2.5`` (spec §2.1)."""
    w = max(1, int(window))
    return float(w) / 2.5


def comet_alpha(frame: int, current: int, window: int) -> float:
    """Opacity weight for ``frame`` given current frame ``current``."""
    sigma = comet_sigma(window)
    d = float(current - frame)
    decay = math.exp(-(d * d) / (2.0 * sigma * sigma))
    return COMET_ALPHA_MIN + (COMET_ALPHA_MAX - COMET_ALPHA_MIN) * decay


def comet_trail_alphas(n_frames: int, current: int, window: int) -> list[float]:
    """Per-frame opacities for ``[t-W, t]``; ghosts collapse to 0.15."""
    n = max(0, int(n_frames))
    out: list[float] = []
    for f in range(n):
        if f < current - window or f > current + window:
            out.append(GHOST_ALPHA)
        elif f <= current:
            out.append(round(comet_alpha(f, current, window), 4))
        else:
            # Future frames fade symmetrically (mirror of the past kernel).
            out.append(round(comet_alpha(2 * current - f, current, window), 4))
    return out


def comet_stroke_width(frame: int, current: int, window: int, base: float = 0.03) -> float:
    """Trail stroke width growing toward the head frame (spec §2.1)."""
    w = max(1, int(window))
    lo = current - w
    t = (frame - lo) / float(w) if w else 1.0
    t = min(1.0, max(0.0, t))
    return round(float(base) * (1.0 + 2.0 * t), 4)


def contrast_stroke(is_light: bool) -> str:
    """High-legibility perimeter color (spec §2.2)."""
    return "#0b0f14" if is_light else "#ffffff"


def aggregate_hexbins(
    points_world: np.ndarray,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    radius: float = DEFAULT_HEX_RADIUS,
    eps: float = 0.5,
) -> list[HexbinCell2D]:
    """Aggregate world points into hexbin cells (spec §2.4)."""
    pts = np.ascontiguousarray(points_world, dtype=np.float32).reshape(-1, 2)
    n = int(pts.shape[0])
    if n == 0:
        return []
    r = max(1e-6, float(radius))
    dx = math.sqrt(3.0) * r
    dy = 1.5 * r
    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)
    v = np.rint((ys + 5.5) / dy).astype(np.int64)
    x_shift = np.mod(v, 2).astype(np.float64) * (dx / 2.0)
    u = np.rint((xs + 5.5 - x_shift) / dx).astype(np.int64)
    u_min, v_min = int(u.min()), int(v.min())
    u0 = u - u_min
    v0 = v - v_min
    n_cols = int(u0.max()) + 1
    flat = (v0 * n_cols + u0).astype(np.int64)

    # Vectorized per-bin aggregation (fully in C; no per-bin Python loop,
    # so 10k points stay well under the <15 ms DoD budget).
    _uniq_vals, first_idx, inv, counts = np.unique(flat, return_index=True, return_inverse=True, return_counts=True)
    uniq = first_idx
    n_bins = int(uniq.size)
    max_count = int(counts.max()) if counts.size else 0
    uu_bins = (u0[uniq] + u_min).astype(np.float64)
    vv_bins = (v0[uniq] + v_min).astype(np.float64)
    parity = np.mod(np.abs(vv_bins), 2.0)
    cx_all = np.clip(uu_bins * dx + parity * (dx / 2.0) - 5.5, -5.5, 5.5)
    cy_all = np.clip(vv_bins * dy - 5.5, -5.5, 5.5)

    dom_all: list[int | None] = [None] * n_bins
    err_all = np.zeros((n_bins,), dtype=np.float64)
    res_all = np.zeros((n_bins,), dtype=np.float64)

    def _as_f64(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        return arr if arr.size == n else None

    yt = _as_f64(np.asarray(y_true, dtype=np.float64) if y_true is not None else None)
    yp = _as_f64(np.asarray(y_pred, dtype=np.float64) if y_pred is not None else None)
    rs = _as_f64(np.asarray(residuals, dtype=np.float64) if residuals is not None else None)

    if yt is not None and yp is not None:
        ti = np.where(np.isfinite(yt), np.rint(yt), np.nan)
        pi = np.where(np.isfinite(yp), np.rint(yp), np.nan)
        both = np.isfinite(ti) & np.isfinite(pi)
        if both.any():
            labels = np.unique(ti[both]).astype(np.int64)
            lut = {int(v): k for k, v in enumerate(labels.tolist())}
            coded = np.full((n,), -1, dtype=np.int64)
            coded[both] = np.asarray([lut[int(v)] for v in ti[both].tolist()], dtype=np.int64)
            onehot = np.zeros((n_bins, len(labels)), dtype=np.float64)
            np.add.at(onehot, (inv[both], coded[both]), 1.0)
            winners = np.argmax(onehot, axis=1)
            has = onehot.sum(axis=1) > 0
            for b in range(n_bins):
                if has[b]:
                    dom_all[b] = int(labels[int(winners[b])])
            mismatch = np.zeros((n,), dtype=np.float64)
            mismatch[both] = (ti[both] != pi[both]).astype(np.float64)
            err_all = np.bincount(inv, weights=mismatch, minlength=n_bins) / np.maximum(counts, 1)
            absdiff = np.zeros((n,), dtype=np.float64)
            absdiff[both] = np.abs(yt[both] - yp[both])
            res_all = np.bincount(inv, weights=absdiff, minlength=n_bins) / np.maximum(counts, 1)
    elif yt is not None:
        fin = np.isfinite(yt)
        if fin.any():
            labels = np.unique(np.rint(yt[fin]).astype(np.int64))
            lut = {int(v): k for k, v in enumerate(labels.tolist())}
            coded = np.full((n,), -1, dtype=np.int64)
            coded[fin] = np.asarray([lut[int(v)] for v in np.rint(yt[fin]).tolist()], dtype=np.int64)
            onehot = np.zeros((n_bins, len(labels)), dtype=np.float64)
            np.add.at(onehot, (inv[fin], coded[fin]), 1.0)
            winners = np.argmax(onehot, axis=1)
            has = onehot.sum(axis=1) > 0
            for b in range(n_bins):
                if has[b]:
                    dom_all[b] = int(labels[int(winners[b])])
    if rs is not None:
        fin = np.isfinite(rs)
        w = np.zeros((n,), dtype=np.float64)
        w[fin] = np.abs(rs[fin])
        res_all = np.bincount(inv, weights=w, minlength=n_bins) / np.maximum(counts, 1)
        if yt is None or yp is None:
            over = np.zeros((n,), dtype=np.float64)
            over[fin] = (np.abs(rs[fin]) > float(eps)).astype(np.float64)
            err_all = np.bincount(inv, weights=over, minlength=n_bins) / np.maximum(counts, 1)

    order = np.argsort(-counts, kind="stable")
    cells: list[HexbinCell2D] = []
    for b in order.tolist():
        cnt = int(counts[b])
        scale = 0.3 + 0.7 * math.sqrt(float(cnt) / float(max_count)) if max_count else 1.0
        cells.append(
            HexbinCell2D(
                center_x=round(float(cx_all[b]), 3),
                center_y=round(float(cy_all[b]), 3),
                radius=round(float(r * scale), 4),
                count=cnt,
                dominant_class=dom_all[b],
                mean_residual=round(float(res_all[b]), 4),
                error_rate=round(min(1.0, max(0.0, float(err_all[b]))), 4),
            )
        )
    return cells


def errors_only_indices(
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
    residuals: np.ndarray | None = None,
    eps: float = 0.5,
) -> list[int]:
    """Indices of mispredicted points (spec mechanism C switch)."""
    if y_true is not None and y_pred is not None:
        t = np.asarray(list(y_true), dtype=np.float64).ravel()
        p = np.asarray(list(y_pred), dtype=np.float64).ravel()
        m = min(t.size, p.size)
        if m == 0:
            return []
        bad = np.rint(t[:m]) != np.rint(p[:m])
        bad |= ~np.isfinite(t[:m]) | ~np.isfinite(p[:m])
        return [int(i) for i in np.flatnonzero(bad).tolist()]
    if residuals is not None:
        r = np.asarray(list(residuals), dtype=np.float64).ravel()
        fin = np.isfinite(r)
        bad = np.zeros((r.size,), dtype=bool)
        bad[fin] = np.abs(r[fin]) > float(eps)
        bad[~fin] = True
        return [int(i) for i in np.flatnonzero(bad).tolist()]
    return []


def residual_spoke_endpoint(
    x: float, y: float, residual: float, angle_rad: float = math.pi / 2.0, scale: float = 0.25
) -> list[float]:
    """2D error-spoke endpoint from ``(x, y)`` along ``angle`` scaled by ``|e|``."""
    e = float(residual) if math.isfinite(float(residual)) else 0.0
    length = min(2.5, max(0.0, abs(e) * float(scale)))
    sgn = 1.0 if e >= 0 else -1.0
    ex = float(x) + math.cos(float(angle_rad)) * length * sgn
    ey = float(y) + math.sin(float(angle_rad)) * length * sgn
    return [round(float(x), 3), round(float(y), 3), round(ex, 3), round(ey, 3)]


def _ms_segments(
    xs: np.ndarray, ys: np.ndarray, field: np.ndarray, threshold: float
) -> list[list[float]]:
    """Minimal Marching-Squares isoline extractor (local copy).

    Kept local so this engine never imports the heavyweight contour +
    extractor chain (sklearn/xgboost/onnx adapters cost ~2 s on first
    import and would blow the <15 ms DoD budget on a cold request).
    """
    segments: list[list[float]] = []
    ny, nx = field.shape
    if ny < 2 or nx < 2:
        return segments
    thr = float(threshold)
    above = field >= thr
    codes = (
        above[:-1, :-1].astype(np.uint8)
        | (above[:-1, 1:].astype(np.uint8) << 1)
        | (above[1:, 1:].astype(np.uint8) << 2)
        | (above[1:, :-1].astype(np.uint8) << 3)
    )
    cross_j, cross_i = np.nonzero((codes != 0) & (codes != 15))
    if cross_j.size == 0:
        return segments
    xs_f = np.asarray(xs, dtype=np.float64)
    ys_f = np.asarray(ys, dtype=np.float64)
    fld = np.asarray(field, dtype=np.float64)

    def _interp(p_a: float, v_a: float, p_b: float, v_b: float) -> float:
        denom = v_b - v_a
        if abs(denom) < 1e-12:
            return (p_a + p_b) / 2.0
        t = min(1.0, max(0.0, (thr - v_a) / denom))
        return p_a + t * (p_b - p_a)

    for j, i in zip(cross_j.tolist(), cross_i.tolist()):
        s0 = float(fld[j, i])
        s1 = float(fld[j, i + 1])
        s2 = float(fld[j + 1, i + 1])
        s3 = float(fld[j + 1, i])
        x0, x1 = float(xs_f[i]), float(xs_f[i + 1])
        y0, y1 = float(ys_f[j]), float(ys_f[j + 1])
        top = (_interp(x0, s0, x1, s1), y0) if (s0 >= thr) != (s1 >= thr) else None
        right = (x1, _interp(y0, s1, y1, s2)) if (s1 >= thr) != (s2 >= thr) else None
        bottom = (_interp(x0, s3, x1, s2), y1) if (s3 >= thr) != (s2 >= thr) else None
        left = (x0, _interp(y0, s0, y1, s3)) if (s0 >= thr) != (s3 >= thr) else None
        pts = [p for p in (top, right, bottom, left) if p is not None]
        if len(pts) == 2:
            segments.append([round(pts[0][0], 3), round(pts[0][1], 3), round(pts[1][0], 3), round(pts[1][1], 3)])
        elif len(pts) == 4:
            segments.append([round(pts[0][0], 3), round(pts[0][1], 3), round(pts[1][0], 3), round(pts[1][1], 3)])
            segments.append([round(pts[2][0], 3), round(pts[2][1], 3), round(pts[3][0], 3), round(pts[3][1], 3)])
    return segments


def compute_density_contours(
    points_world: np.ndarray,
    grid_resolution: int = DENSITY_GRID_RESOLUTION,
    levels: int = 3,
) -> list[DensityContourLevel2D]:
    """Fast empirical-density contours: histogram + 3x3 blur + marching squares."""

    pts = np.ascontiguousarray(points_world, dtype=np.float32).reshape(-1, 2)
    res = max(16, min(120, int(grid_resolution)))
    if pts.shape[0] == 0:
        return []
    xs = np.clip(pts[:, 0].astype(np.float64), -5.0, 5.0)
    ys = np.clip(pts[:, 1].astype(np.float64), -5.0, 5.0)
    hist, _, _ = np.histogram2d(ys, xs, bins=res, range=[[-5.0, 5.0], [-5.0, 5.0]])
    # 3x3 box blur (two passes) — cheap KDE-lite, no scipy dependency.
    for _ in range(2):
        padded = np.pad(hist, 1, mode="edge")
        hist = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    peak = float(hist.max())
    if not math.isfinite(peak) or peak <= 0:
        return []
    gx = np.linspace(-5.0, 5.0, res, dtype=np.float64)
    gy = np.linspace(-5.0, 5.0, res, dtype=np.float64)
    out: list[DensityContourLevel2D] = []
    for k in range(max(1, int(levels))):
        frac = (k + 1) / (float(levels) + 1.0)
        thr = peak * frac
        segs = _ms_segments(gx, gy, hist, float(thr))
        if segs:
            out.append(
                DensityContourLevel2D(
                    density_threshold=round(float(thr), 4),
                    segments=[[round(float(a), 3) for a in s] for s in segs[:5_000]],
                )
            )
    return out


def build_point_intelligence(
    points_world: np.ndarray,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    hex_radius: float = DEFAULT_HEX_RADIUS,
    eps: float = 0.5,
) -> PointIntelligenceResponse:
    """Assemble the full mechanism-D payload (hexbins + density contours)."""
    started = time.perf_counter()
    cells = aggregate_hexbins(points_world, y_true, y_pred, residuals, radius=hex_radius, eps=eps)
    contours = compute_density_contours(points_world)
    elapsed = round((time.perf_counter() - started) * 1000.0, 2)
    return PointIntelligenceResponse(
        hexbin_cells=cells[:2_000],
        density_contours=contours,
        max_bin_count=max((c.count for c in cells), default=0),
        computation_time_ms=elapsed,
    )

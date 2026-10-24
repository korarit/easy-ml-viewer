"""Phase 55 — 2D point-intelligence math for Neural Networks, Pipelines & ONNX.

Binds the Phase-49 point-intelligence engine to the exact payloads behind
``neural_2d_extractor`` (MLP hyperplanes, pipeline scaler affines, ONNX
latent SVD plane):

- Mechanism A: dual-space comet trail — the exact affine
  ``z = (x - mu) / sigma`` maps raw units to scaled (Z-scored) space 1-to-1
  (``StandardScaler`` / ``RobustScaler`` / ``MinMaxScaler`` prefixes); the
  trail length visibly stretches / compresses (metric distortion). For deep
  ONNX the trail replays the latent geodesic on the SVD plane.
- Mechanism B: softmax logit-gap dual-ring — ``Δz = z_(1) - z_(2)`` from the
  pre-softmax / pre-sigmoid logits: ``Δz > 3.0`` saturated confident (solid
  ring), ``Δz < 0.5`` tempting tie (neon diffuse halo). Binary heads map
  ``Δz = |logit|``. The last-hidden activation optionally tints the halo.
- Mechanism C: input loss-gradient spokes — the negative input gradient
  ``g = ∇_x L`` direction sizes each spoke; hard / adversarially fragile
  samples pull hardest (DoD #4 companion).
- Mechanism D: data-desert & overconfident-extrapolation hexbins —
  ``Hazard(u,v) = 1[count==0] · max_c P``; cells with ``Hazard > 0.85``
  earn the desert badge (spec §2.4).

Pure NumPy + float64 math, no model or DataFrame access: callers pass the
already-extracted scaler / logit / gradient / hexbin arrays, so the same
kernels serve the orchestrator, tests, and any future endpoint.
"""

from __future__ import annotations

import numpy as np

ZERO_TOL = 1e-12

#: Logit-gap ring thresholds (spec §2.2, DoD #2).
LOGIT_CONFIDENT = 3.0
LOGIT_DIFFUSE = 0.5

#: Desert-hazard probability floor (spec §2.4, DoD #3).
HAZARD_THRESHOLD = 0.85

#: World-space cap for normalized loss-gradient spokes (mirrors Phase 50/54).
SPOKE_MAX_WORLD = 1.5


def dual_space_transform(
    values: np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """Affine map ``z = (x - mu) / sigma`` per axis (spec §2.1).

    Non-finite inputs stay ``NaN``; a degenerate ``sigma`` (``<= tol``)
    yields all-``NaN`` instead of a bogus scaled frame (DoD #1).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return np.empty((0,), dtype=np.float64)
    if not np.isfinite(mu) or not np.isfinite(sigma) or abs(float(sigma)) <= ZERO_TOL:
        return np.full((v.size,), np.nan, dtype=np.float64)
    out = (v - float(mu)) / float(sigma)
    out[~np.isfinite(v)] = np.nan
    return np.ascontiguousarray(np.where(np.isfinite(out), out, np.nan))


def dual_space_points(
    points_xy: np.ndarray, mu_x: float, sigma_x: float, mu_y: float, sigma_y: float
) -> np.ndarray:
    """Map ``(N, 2)`` raw points into scaled space (spec §2.1, 1-to-1).

    Returns ``(N, 2)`` float64; rows with any non-finite coordinate map to
    ``(NaN, NaN)`` so the trail gaps honestly instead of teleporting.
    """
    pts = np.ascontiguousarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    zx = dual_space_transform(pts[:, 0], mu_x, sigma_x)
    zy = dual_space_transform(pts[:, 1], mu_y, sigma_y)
    return np.ascontiguousarray(np.column_stack([zx, zy]))


def logit_gap(logits: np.ndarray) -> np.ndarray:
    """Logit margin ``Δz = z_(1) - z_(2)`` per row (spec §2.2, DoD #2).

    ``logits`` is ``(N, C)`` pre-softmax (binary heads pass ``(N, 1)`` or
    ``(N, 2)`` and map to ``|logit|``). Non-finite rows yield ``NaN``
    (ring renders neutral, never a bogus confident badge).
    """
    z = np.ascontiguousarray(logits, dtype=np.float64)
    if z.size == 0:
        return np.empty((0,), dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {z.shape}")
    n = int(z.shape[0])
    c = int(z.shape[1])
    if c == 0:
        return np.full((n,), np.nan, dtype=np.float64)
    if c == 1:
        col = z[:, 0]
        return np.ascontiguousarray(np.where(np.isfinite(col), np.abs(col), np.nan))
    out = np.full((n,), np.nan, dtype=np.float64)
    for i in range(n):
        row = z[i]
        if not np.all(np.isfinite(row)):
            continue
        part = np.partition(row, -2)
        out[i] = float(part[-1] - part[-2])
    return np.ascontiguousarray(out)


def logit_gap_status(gaps: np.ndarray) -> np.ndarray:
    """Ring status per point: ``confident`` / ``diffuse`` / ``decisive``.

    ``Δz > 3.0`` → confident (saturated solid ring); ``Δz < 0.5`` →
    diffuse (neon tie halo); otherwise decisive. ``NaN`` → ``unknown``.
    """
    g = np.asarray(gaps, dtype=np.float64).ravel()
    out = np.full((g.size,), "unknown", dtype=object)
    fin = np.isfinite(g)
    out[fin & (g > LOGIT_CONFIDENT)] = "confident"
    out[fin & (g < LOGIT_DIFFUSE)] = "diffuse"
    mid = fin & (g >= LOGIT_DIFFUSE) & (g <= LOGIT_CONFIDENT)
    out[mid] = "decisive"
    return np.ascontiguousarray(out)


def binary_logit_from_proba(proba: np.ndarray) -> np.ndarray:
    """Recover the binary logit ``log(p / (1 - p))`` from ``P(class_1)``.

    Probabilities clip to ``[eps, 1 - eps]`` so a miscalibrated ``p``
    outside ``[0, 1]`` cannot fake an infinite gap (DoD #2).
    """
    p = np.asarray(proba, dtype=np.float64).ravel()
    out = np.full((p.size,), np.nan, dtype=np.float64)
    ok = np.isfinite(p)
    pc = np.clip(p[ok], 1e-12, 1.0 - 1e-12)
    out[ok] = np.log(pc / (1.0 - pc))
    return np.ascontiguousarray(out)


def gradient_spoke_lengths(
    grad_norms: np.ndarray, max_world: float = SPOKE_MAX_WORLD
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized input-loss-gradient spoke geometry (spec §2.3).

    Peak ``||g||`` maps to ``max_world``; returns ``(lengths, angles_ok)``
    with lengths ``0`` for non-finite norms. Callers pair lengths with the
    per-row unit direction ``(gx, gy) / ||g||`` for the SVG endpoint.
    """
    r = np.asarray(grad_norms, dtype=np.float64).ravel()
    n = r.size
    lengths = np.zeros((n,), dtype=np.float64)
    ok = np.zeros((n,), dtype=bool)
    mag = np.where(np.isfinite(r), np.abs(r), 0.0)
    peak = float(np.max(mag)) if n else 0.0
    if peak > ZERO_TOL:
        lengths = mag / peak * float(max_world)
        ok = np.isfinite(r) & (np.abs(r) > ZERO_TOL)
        lengths[~np.isfinite(r)] = 0.0
    return np.ascontiguousarray(lengths), np.ascontiguousarray(ok)


def gradient_unit_vectors(grad_xy: np.ndarray) -> np.ndarray:
    """Unit directions ``g / ||g||`` for ``(N, 2)`` input gradients.

    Zero / non-finite rows map to ``(0, 0)`` so no spoke is drawn instead
    of a misleading one.
    """
    g = np.ascontiguousarray(grad_xy, dtype=np.float64).reshape(-1, 2)
    if g.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    norms = np.hypot(g[:, 0], g[:, 1])
    out = np.zeros_like(g)
    ok = np.isfinite(norms) & (norms > ZERO_TOL) & np.all(np.isfinite(g), axis=1)
    out[ok] = g[ok] / norms[ok].reshape(-1, 1)
    return np.ascontiguousarray(out)


def desert_hazard(
    counts: np.ndarray, max_proba: np.ndarray, threshold: float = HAZARD_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Data-desert hazard flags (spec §2.4, DoD #3).

    ``Hazard = 1[count == 0] · max_c P``; returns ``(hazard (N,),
    is_desert (N,))`` with ``is_desert = hazard > threshold``. Non-finite
    probabilities yield ``NaN`` hazard and no badge.
    """
    c = np.asarray(counts, dtype=np.float64).ravel()
    p = np.asarray(max_proba, dtype=np.float64).ravel()
    n = min(c.size, p.size)
    if n == 0:
        empty_f = np.empty((0,), dtype=np.float64)
        return empty_f, np.zeros((0,), dtype=bool)
    c, p = c[:n], p[:n]
    hazard = np.full((n,), np.nan, dtype=np.float64)
    ok = np.isfinite(p)
    empty = (c == 0) | ((np.isfinite(c)) & (c <= 0))
    hazard[ok] = np.where(empty[ok], np.clip(p[ok], 0.0, 1.0), 0.0)
    is_desert = np.isfinite(hazard) & (hazard > float(threshold))
    return np.ascontiguousarray(hazard), np.ascontiguousarray(is_desert)


def hidden_halo_tone(activations: np.ndarray) -> np.ndarray:
    """Last-hidden activation tone per row for the diffuse halo (spec §2.2).

    Mean absolute activation clipped to ``[0, 1]``-ish tone; non-finite
    rows yield ``NaN`` so the halo falls back to the plain neon style.
    """
    a = np.ascontiguousarray(activations, dtype=np.float64)
    if a.size == 0:
        return np.empty((0,), dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D activations, got shape {a.shape}")
    with np.errstate(invalid="ignore"):
        masked = np.abs(np.where(np.isfinite(a), a, np.nan))
        count = np.sum(~np.isnan(masked), axis=1)
        total = np.nansum(masked, axis=1)
    tone = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return np.ascontiguousarray(tone)

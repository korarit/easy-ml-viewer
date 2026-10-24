"""Phase 56 — Point Intelligence 2D multi-track baker for the ``.mlv`` container.

Spec §2 (``backend/app/core/binary_codec.py`` / ``MLV v2``)::

    point_intelligence_2d_track = {
        "flags": {"trails": bool, "dual_ring": bool, "error_lens": bool, "hexbin": bool},
        "hexbin_radius": float,
        "hexbin_grid": {"cells": [{cx, cy, r, count, error_rate, dom_class}]},
        "density_contours": [{"threshold": float, "segments": [[x1,y1,x2,y2]]}],
        "frame_point_states": columnar float32 track (N frames),
    }

Design notes (memory §5.1 + spec §2.2 compression target):
- Frame states are stored **columnar** (Struct-of-Arrays float32) so the
  shared ``BinaryCodec`` (MsgPack ``__nd__`` + Zstd) compresses them ~10x.
  A ``list[dict]`` with repeated keys would blow the <150 KB budget for
  1,000 rows × 100 frames.
- Hexbin cells capped at 2,000 (renderer cap) with an explicit
  ``truncated`` flag; density segments capped at 5,000 per level.
- ``restore_point_intelligence_track`` returns ``None`` for legacy files
  without the block so callers fall back to on-the-fly recompute.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_HEXBIN_CELLS = 2_000
MAX_DENSITY_LEVELS = 8
MAX_SEGMENTS_PER_LEVEL = 5_000
MAX_FRAME_POINTS = 100_000
TRAIL_LENGTHS = (5, 10, 20)

#: Phase 56 — fixed-point quantization for frame states (world ±32 → int16).
#: Storing quantized int16/uint8 (instead of float32) lets Zstd exploit the
#: slow-varying high bytes, so 1,000 rows × 100 frames stays < 150 KB.
QUANT_SCALE = 1000.0
QUANT_CLAMP = 32.0

_DEFAULT_FLAGS = {"trails": True, "dual_ring": True, "error_lens": False, "hexbin": False}


def _clean_flags(flags: dict[str, Any] | None) -> dict[str, bool]:
    out = dict(_DEFAULT_FLAGS)
    if isinstance(flags, dict):
        for key in out:
            if key in flags:
                out[key] = bool(flags[key])
    return out


def _clean_hexbin_cells(cells: Any, radius_fallback: float) -> tuple[list[dict[str, Any]], bool, float]:
    radius = float(radius_fallback) if np.isfinite(float(radius_fallback or 0)) else 0.45
    radius = min(2.0, max(0.05, radius))
    clean: list[dict[str, Any]] = []
    if isinstance(cells, list):
        for cell in cells[: MAX_HEXBIN_CELLS + 1]:
            if not isinstance(cell, dict):
                continue
            try:
                cx = float(cell.get("center_x", cell.get("cx", 0.0)))
                cy = float(cell.get("center_y", cell.get("cy", 0.0)))
                r = float(cell.get("radius", cell.get("r", radius)))
                cnt = int(cell.get("count", 0) or 0)
                err = float(cell.get("error_rate", 0.0) or 0.0)
                dom = cell.get("dominant_class", cell.get("dom_class", None))
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite([cx, cy, r, err])) or cnt < 0:
                continue
            clean.append({
                "cx": round(cx, 3),
                "cy": round(cy, 3),
                "r": round(min(2.0, max(1e-3, r)), 4),
                "count": cnt,
                "error_rate": round(min(1.0, max(0.0, err)), 4),
                "dom_class": int(dom) if dom is not None else None,
            })
    truncated = len(clean) > MAX_HEXBIN_CELLS
    return clean[:MAX_HEXBIN_CELLS], truncated, round(radius, 4)


def _clean_density_contours(levels: Any) -> tuple[list[dict[str, Any]], bool]:
    clean: list[dict[str, Any]] = []
    truncated = False
    if isinstance(levels, list):
        for level in levels[: MAX_DENSITY_LEVELS + 1]:
            if not isinstance(level, dict):
                continue
            try:
                thr = float(level.get("density_threshold", level.get("threshold", 0.0)))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(thr):
                continue
            segs_in = level.get("segments", []) or []
            segs: list[list[float]] = []
            for seg in segs_in:
                if len(segs) >= MAX_SEGMENTS_PER_LEVEL:
                    truncated = True
                    break
                try:
                    vals = [float(v) for v in list(seg)[:4]]
                except (TypeError, ValueError):
                    continue
                if len(vals) != 4 or not all(np.isfinite(vals)):
                    continue
                segs.append([round(v, 3) for v in vals])
            clean.append({"threshold": round(thr, 4), "segments": segs})
    if len(clean) > MAX_DENSITY_LEVELS:
        clean = clean[:MAX_DENSITY_LEVELS]
        truncated = True
    return clean, truncated


def _as_f32(arr: Any, n: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32).ravel()
    if a.size != n:
        # Honest truncation/pad: never silently misalign frame indices.
        out = np.zeros((n,), dtype=np.float32)
        m = min(n, a.size) if a.size else 0
        if m:
            out[:m] = a[:m]
        return np.ascontiguousarray(out, dtype=np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


def _quantize_i16(values: Any, n: int) -> np.ndarray:
    """Fixed-point quantize world/residual floats to int16 (1e-3 resolution)."""
    a = np.asarray(values, dtype=np.float64).ravel() if values is not None else np.zeros((0,))
    out = np.zeros((n,), dtype=np.int16)
    m = min(n, a.size)
    if m:
        clipped = np.clip(a[:m], -QUANT_CLAMP, QUANT_CLAMP)
        finite = np.isfinite(clipped)
        q = np.zeros((m,), dtype=np.float64)
        q[finite] = np.rint(clipped[finite] * QUANT_SCALE)
        out[:m] = np.clip(q, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(out, dtype=np.int16)


def _as_class_i16(values: Any, n: int) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).ravel() if values is not None else np.zeros((0,))
    out = np.zeros((n,), dtype=np.int16)
    m = min(n, a.size)
    if m:
        finite = np.isfinite(a[:m])
        q = np.zeros((m,), dtype=np.float64)
        q[finite] = np.rint(a[:m][finite])
        out[:m] = np.clip(q, -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(out, dtype=np.int16)


def _dequantize_i16(q: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(q, dtype=np.float32) / np.float32(QUANT_SCALE), dtype=np.float32)


def bake_point_intelligence_track(
    flags: dict[str, Any] | None = None,
    hexbin_cells: list[dict[str, Any]] | Any = None,
    density_contours: list[dict[str, Any]] | Any = None,
    hex_radius: float = 0.45,
    frame_x: Any = None,
    frame_y: Any = None,
    frame_actual: Any = None,
    frame_pred: Any = None,
    frame_residual: Any = None,
    frame_is_error: Any = None,
) -> dict[str, Any]:
    """Encode a point-intelligence track block (fail-safe: never raises on junk)."""
    cells_in: list[dict[str, Any]] = []
    if hexbin_cells is not None and isinstance(hexbin_cells, list):
        cells_in = hexbin_cells
    elif isinstance(hexbin_cells, dict) and isinstance(hexbin_cells.get("cells"), list):
        cells_in = hexbin_cells["cells"]
    contours_in = density_contours if isinstance(density_contours, list) else []

    cells, hex_truncated, radius = _clean_hexbin_cells(cells_in, hex_radius)
    contours, contour_truncated = _clean_density_contours(contours_in)

    n = 0
    for candidate in (frame_x, frame_y, frame_actual, frame_pred, frame_residual, frame_is_error):
        if candidate is not None:
            try:
                n = max(n, int(np.asarray(candidate).ravel().size))
            except (TypeError, ValueError):
                continue
    n = min(max(0, n), MAX_FRAME_POINTS)

    if n == 0:
        return {
            "flags": _clean_flags(flags if isinstance(flags, dict) else None),
            "hexbin_radius": radius,
            "hexbin_grid": {"cells": cells, "truncated": hex_truncated},
            "density_contours": contours,
            "contours_truncated": contour_truncated,
            "n_frames": 0,
            "quant_scale": QUANT_SCALE,
            "point_x": np.zeros((0,), dtype=np.int16),
            "point_y": np.zeros((0,), dtype=np.int16),
            "point_actual": np.zeros((0,), dtype=np.int16),
            "point_pred": np.zeros((0,), dtype=np.int16),
            "point_residual": np.zeros((0,), dtype=np.int16),
            "point_is_error": np.zeros((0,), dtype=np.uint8),
        }

    errors: np.ndarray | None = None
    if frame_is_error is not None:
        try:
            errors = np.asarray(frame_is_error).ravel()[:n].astype(np.int32)
            if errors.size != n:
                pad = np.zeros((n,), dtype=np.int32)
                pad[: errors.size] = errors
                errors = pad
        except (TypeError, ValueError):
            errors = None
    if errors is None and frame_actual is not None and frame_pred is not None:
        try:
            ta = np.asarray(list(frame_actual), dtype=np.float64).ravel()[:n]
            pa = np.asarray(list(frame_pred), dtype=np.float64).ravel()[:n]
            m = min(ta.size, pa.size)
            bad = np.zeros((n,), dtype=np.int32)
            if m:
                bad[:m] = (np.rint(ta[:m]) != np.rint(pa[:m])).astype(np.int32)
            errors = np.ascontiguousarray(bad, dtype=np.int32)
        except (TypeError, ValueError):
            errors = np.zeros((n,), dtype=np.int32)
    if errors is None:
        errors = np.zeros((n,), dtype=np.int32)

    return {
        "flags": _clean_flags(flags if isinstance(flags, dict) else None),
        "hexbin_radius": radius,
        "hexbin_grid": {"cells": cells, "truncated": hex_truncated},
        "density_contours": contours,
        "contours_truncated": contour_truncated,
        "n_frames": int(n),
        "quant_scale": QUANT_SCALE,
        "point_x": _quantize_i16(frame_x if frame_x is not None else np.zeros((n,)), n),
        "point_y": _quantize_i16(frame_y if frame_y is not None else np.zeros((n,)), n),
        "point_actual": _as_class_i16(frame_actual if frame_actual is not None else np.zeros((n,)), n),
        "point_pred": _as_class_i16(frame_pred if frame_pred is not None else np.zeros((n,)), n),
        "point_residual": _quantize_i16(frame_residual if frame_residual is not None else np.zeros((n,)), n),
        "point_is_error": np.ascontiguousarray(np.clip(np.asarray(errors, dtype=np.int64).ravel()[:n], 0, 255).astype(np.uint8), dtype=np.uint8),
    }


def restore_point_intelligence_track(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract + normalize the ``point_intelligence_2d_track`` block (None if absent)."""
    if not isinstance(payload, dict):
        return None
    block = payload.get("point_intelligence_2d_track")
    if not isinstance(block, dict) or not block:
        return None
    try:
        n = int(block.get("n_frames", 0) or 0)
    except (TypeError, ValueError):
        return None
    if n < 0 or n > MAX_FRAME_POINTS:
        return None

    scale = block.get("quant_scale", QUANT_SCALE)
    try:
        scale_f = float(scale)
    except (TypeError, ValueError):
        scale_f = QUANT_SCALE
    if not np.isfinite(scale_f) or scale_f <= 0:
        scale_f = QUANT_SCALE

    def _col(name: str, quantized: bool) -> np.ndarray | None:
        v = block.get(name)
        if v is None:
            return None
        try:
            arr = np.asarray(v).ravel()
        except (TypeError, ValueError):
            return None
        if arr.size != n:
            return None
        if quantized and arr.dtype == np.int16:
            return np.ascontiguousarray(arr.astype(np.float32) / np.float32(scale_f), dtype=np.float32)
        try:
            return np.ascontiguousarray(arr.astype(np.float32), dtype=np.float32)
        except (TypeError, ValueError):
            return None

    cols: dict[str, Any] = {}
    for name, q in (("point_x", True), ("point_y", True), ("point_residual", True),
                    ("point_actual", False), ("point_pred", False)):
        arr = _col(name, q)
        if n and arr is None:
            return None
        cols[name] = arr if arr is not None else np.zeros((0,), dtype=np.float32)
    try:
        err_arr = np.asarray(block.get("point_is_error", []), dtype=np.int64).ravel()
    except (TypeError, ValueError):
        return None
    if n and err_arr.size != n:
        return None
    grid = block.get("hexbin_grid", {}) or {}
    cells_raw = grid.get("cells", []) if isinstance(grid, dict) else []
    contours_raw = block.get("density_contours", []) or []
    cells, _, radius = _clean_hexbin_cells(cells_raw if isinstance(cells_raw, list) else [], block.get("hexbin_radius", 0.45))
    contours, _ = _clean_density_contours(contours_raw if isinstance(contours_raw, list) else [])
    return {
        "flags": _clean_flags(block.get("flags")),
        "hexbin_radius": radius,
        "hexbin_grid": {"cells": cells, "truncated": bool(grid.get("truncated", False)) if isinstance(grid, dict) else False},
        "density_contours": contours,
        "contours_truncated": bool(block.get("contours_truncated", False)),
        "n_frames": n,
        "quant_scale": scale_f,
        "point_x": cols["point_x"],
        "point_y": cols["point_y"],
        "point_actual": cols["point_actual"],
        "point_pred": cols["point_pred"],
        "point_residual": cols["point_residual"],
        "point_is_error": np.ascontiguousarray(np.clip(err_arr, 0, 1).astype(np.int32), dtype=np.int32),
    }


def point_track_estimate_bytes(n_frames: int, n_cells: int, n_segments: int) -> int:
    """Rough compressed-size estimate (≈6 float32 cols + hex cells + segments)."""
    n = max(0, min(int(n_frames), MAX_FRAME_POINTS))
    raw = n * 6 * 4 + max(0, int(n_cells)) * 32 + max(0, int(n_segments)) * 16 + 512
    return max(1_024, int(raw * 0.30))

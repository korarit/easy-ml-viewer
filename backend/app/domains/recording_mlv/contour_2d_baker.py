"""Phase 48 — 2D contour cache baker for the ``.mlv`` container (MsgPack + Zstd).

Encodes a Phase-41 ``Contour2DResponse`` payload into the compact
``contour_2d_cache`` block from the phase-48 spec §2.2::

    contour_2d_cache = {
        "x_feature": ..., "y_feature": ..., "grid_resolution": 120,
        "heatmap_float32": <float32 bytes -> handled by BinaryCodec ndarrays>,
        "heatmap_min": ..., "heatmap_max": ...,
        "boundary_polylines": [{"class_id": int, "points": [[x1,y1], ...]}],
        "model_specific_2d": {...} | None,
    }

Design notes (§5.1 memory budget):
- The heatmap is stored as a raw ``np.float32`` array and compressed by the
  shared ``BinaryCodec`` (MsgPack ``__nd__`` + Zstd), so a 120×120 grid costs
  ~45 KB on disk instead of ~230 KB as a JSON float list.
- Decision-boundary polylines are capped (400 segments) with an explicit
  ``truncated`` flag so ``.mlv`` files stay small and instant-loading.
- ``restore_contour_2d_cache`` returns ``None`` for legacy v1.x files without
  the block (E48.3) so callers fall back to on-the-fly recompute.

Max boundary segments and heatmap resolution mirror the live contour path
(``grid_resolution`` ∈ [16, 200]).
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_BOUNDARY_SEGMENTS = 400
MIN_GRID_RESOLUTION = 16
MAX_GRID_RESOLUTION = 200


def _clamp_resolution(res: int) -> int:
    return max(MIN_GRID_RESOLUTION, min(MAX_GRID_RESOLUTION, int(res)))


def bake_contour_2d_cache(contour: dict[str, Any] | Any) -> dict[str, Any]:
    """Encode a contour DTO (dict or ``Contour2DResponse``) into a cache block."""
    data: dict[str, Any] = contour.model_dump(mode="json") if hasattr(contour, "model_dump") else dict(contour or {})
    res = _clamp_resolution(int(data.get("grid_resolution", 120) or 120))
    x_feature = str(data.get("x_feature", ""))
    y_feature = str(data.get("y_feature", ""))
    h_min = float(data.get("heatmap_min", 0.0) or 0.0)
    h_max = float(data.get("heatmap_max", 1.0) or 1.0)
    if not np.isfinite(h_min):
        h_min = 0.0
    if not np.isfinite(h_max) or h_max <= h_min:
        h_max = h_min + 1.0

    quantized = list(data.get("heatmap_quantized", []) or [])
    span = float(h_max - h_min)
    if len(quantized) == res * res and span > 1e-12:
        q = np.asarray(quantized, dtype=np.float64).ravel()[: res * res]
        heatmap = (h_min + np.clip(q / 255.0, 0.0, 1.0) * span).astype(np.float32)
    else:
        heatmap = np.full((res * res,), h_min, dtype=np.float32)
    heatmap = np.ascontiguousarray(heatmap.reshape(res, res), dtype=np.float32)

    # Flatten decision-kind isolines into vector polylines (world coords).
    polylines: list[dict[str, Any]] = []
    truncated = False
    for level in data.get("contours", []) or []:
        if not isinstance(level, dict) or level.get("kind") != "decision":
            continue
        class_id = int(level.get("class_index", 0) or 0)
        for seg in level.get("line_segments", []) or []:
            if len(polylines) >= MAX_BOUNDARY_SEGMENTS:
                truncated = True
                break
            try:
                x1, y1, x2, y2 = (float(v) for v in list(seg)[:4])
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite([x1, y1, x2, y2])):
                continue
            polylines.append({
                "class_id": class_id,
                "points": [[round(x1, 3), round(y1, 3)], [round(x2, 3), round(y2, 3)]],
            })
        if truncated:
            break

    model_specific: dict[str, Any] | None = None
    for key in ("linear_2d", "svm_2d", "knn_2d", "probabilistic_2d", "trees_2d", "boosting_2d", "neural_2d"):
        overlay = data.get(key)
        if isinstance(overlay, dict) and overlay:
            if model_specific is None:
                model_specific = {}
            model_specific[key] = overlay

    return {
        "x_feature": x_feature,
        "y_feature": y_feature,
        "grid_resolution": res,
        "prediction_task": str(data.get("prediction_task", "classification")),
        "heatmap_float32": np.ascontiguousarray(heatmap, dtype=np.float32),
        "heatmap_min": h_min,
        "heatmap_max": h_max,
        "boundary_polylines": polylines,
        "boundary_truncated": truncated,
        "model_specific_2d": model_specific,
    }


def restore_contour_2d_cache(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract + normalize the ``contour_2d_cache`` block (``None`` if absent).

    Legacy ``.mlv`` files (v1.x, no 2D block) return ``None`` so the 2D stage
    falls back to on-the-fly recompute (E48.3).
    """
    if not isinstance(payload, dict):
        return None
    block = payload.get("contour_2d_cache")
    if not isinstance(block, dict) or not block:
        return None
    try:
        res = _clamp_resolution(int(block.get("grid_resolution", 0) or 0))
    except (TypeError, ValueError):
        return None
    heat = block.get("heatmap_float32")
    try:
        heat_arr = np.asarray(heat, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return None
    if heat_arr.size != res * res:
        return None
    h_min = block.get("heatmap_min", 0.0)
    h_max = block.get("heatmap_max", 1.0)
    try:
        h_min_f, h_max_f = float(h_min), float(h_max)
    except (TypeError, ValueError):
        return None
    polylines = block.get("boundary_polylines", []) or []
    if not isinstance(polylines, list):
        return None
    clean: list[dict[str, Any]] = []
    for poly in polylines[: MAX_BOUNDARY_SEGMENTS + 1]:
        if not isinstance(poly, dict):
            continue
        pts = poly.get("points", [])
        if not isinstance(pts, list) or len(pts) < 1:
            continue
        clean.append({"class_id": int(poly.get("class_id", 0) or 0), "points": pts})
    model_specific = block.get("model_specific_2d")
    return {
        "x_feature": str(block.get("x_feature", "")),
        "y_feature": str(block.get("y_feature", "")),
        "grid_resolution": res,
        "prediction_task": str(block.get("prediction_task", "classification")),
        "heatmap_float32": np.ascontiguousarray(heat_arr.reshape(res, res), dtype=np.float32),
        "heatmap_min": h_min_f,
        "heatmap_max": h_max_f,
        "boundary_polylines": clean[:MAX_BOUNDARY_SEGMENTS],
        "boundary_truncated": bool(block.get("boundary_truncated", len(clean) > MAX_BOUNDARY_SEGMENTS)),
        "model_specific_2d": model_specific if isinstance(model_specific, dict) else None,
    }


def contour_cache_estimate_bytes(grid_resolution: int, n_polylines: int) -> int:
    """Rough compressed-size estimate (≈45 KB @120×120 per spec §2.2)."""
    res = _clamp_resolution(grid_resolution)
    raw = res * res * 4 + max(0, int(n_polylines)) * 32 + 512
    return max(1_024, int(raw * 0.35))

"""Phase 41 — Universal 2D decision-boundary / contour engine.

Computes a dense ``grid_resolution x grid_resolution`` evaluation grid on the
chosen ``(x_feature, y_feature)`` pair (other features pinned via
median / mean / zero / current-row), runs the fitted model in chunked
batches, and extracts vector isolines with a Marching-Squares pass.

Pure NumPy + float32, no DataFrame retention: a 120x120 grid holds
14,400 floats (~57 KB) and is evaluated in ``PREDICT_CHUNK_SIZE``-row
batches so the flat-RAM profile (§5.1) is preserved.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.core.math_3d import WorldCoordinateScaler, build_slice_grid_matrix

try:  # Phase 42 — optional exact linear geometry (never breaks the contour path).
    from app.domains.model_engine.adapters.linear.linear_2d_extractor import (
        Linear2DGeometry,
        extract_linear_2d_geometry,
    )
except ImportError:  # pragma: no cover - extractor module missing
    Linear2DGeometry = None  # type: ignore[assignment,misc]
    extract_linear_2d_geometry = None  # type: ignore[assignment]

#: Batch size shared with the Phase-09 surface path (memory budget §5.1).
PREDICT_CHUNK_SIZE = 2_500

#: Default dense-grid resolution for the 2D planar view ($120 x 120$).
DEFAULT_GRID_RESOLUTION = 120

#: Confidence isolines drawn for classification heatmaps.
CONFIDENCE_LEVELS = (0.6, 0.7, 0.8, 0.9)

#: Decision threshold for binary classification boundaries.
DECISION_THRESHOLD = 0.5


class ContourLevel2D(BaseModel):
    """One vector isoline set at a fixed scalar threshold."""

    threshold: float = Field(description="Isoline level (e.g. 0.5 = decision boundary)")
    class_index: int = Field(default=0, ge=0)
    kind: str = Field(default="decision", description="decision | confidence | elevation")
    line_segments: list[list[float]] = Field(
        default_factory=list,
        description="Polyline segments as [x1, y1, x2, y2] in world coords [-5, +5]",
    )


class ResponseCurve1D(BaseModel):
    """Regression helper: X sweep vs predicted target at pinned background."""

    x_values_raw: list[float] = Field(default_factory=list)
    y_pred: list[float] = Field(default_factory=list)


class Contour2DResponse(BaseModel):
    x_feature: str
    y_feature: str
    grid_resolution: int
    x_range_raw: tuple[float, float]
    y_range_raw: tuple[float, float]
    prediction_task: str = Field(description="classification | regression")
    classes: list[Any] = Field(default_factory=list)
    heatmap_min: float = 0.0
    heatmap_max: float = 1.0
    heatmap_quantized: list[int] = Field(
        default_factory=list,
        description=(
            "Row-major Ny*Nx uint8 field (0-255) normalized from the raw "
            "scalar field via heatmap_min/max. Keeps the 120x120 DTO under "
            "the 150 KB budget; decode with min + (q/255)*(max-min)."
        ),
    )
    contours: list[ContourLevel2D] = Field(default_factory=list)
    response_curve_1d: ResponseCurve1D | None = None
    # Phase 42 — exact analytic linear geometry (None for non-linear families).
    linear_2d: dict[str, Any] | None = Field(
        default=None,
        description="Exact 2D decision line / margin / sparsity payload for linear models",
    )
    computation_time_ms: float = 0.0


def _sanitize(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    fill = float(np.median(finite)) if finite.size else float(fallback)
    return np.where(np.isfinite(arr), arr, fill)


def _chunked_predict(adapter: Any, model_obj: Any, matrix: np.ndarray) -> np.ndarray:
    """Regression / class-label predictions in ≤2,500-row chunks."""
    mat = np.ascontiguousarray(matrix, dtype=np.float32)
    out = np.empty((mat.shape[0],), dtype=np.float64)
    out[:] = np.nan
    offset = 0
    while offset < mat.shape[0]:
        chunk = np.ascontiguousarray(mat[offset : offset + PREDICT_CHUNK_SIZE], dtype=np.float32)
        result = adapter.predict_batch(model_obj, chunk)
        vals = np.asarray(list(result.predictions), dtype=np.float64)
        out[offset : offset + vals.shape[0]] = vals
        del chunk, result, vals
        offset += PREDICT_CHUNK_SIZE
    return _sanitize(out)


def _chunked_proba_matrix(model_obj: Any, final_est: Any, matrix: np.ndarray) -> np.ndarray | None:
    """Full ``(N, C)`` probability matrix in chunks, or ``None`` when unsupported."""
    est = model_obj if hasattr(model_obj, "predict_proba") else final_est
    if not hasattr(est, "predict_proba"):
        return None
    try:
        mat = np.ascontiguousarray(matrix, dtype=np.float32)
        first = np.asarray(est.predict_proba(mat[: min(PREDICT_CHUNK_SIZE, mat.shape[0])]), dtype=np.float64)
        if first.ndim != 2 or first.shape[1] == 0:
            return None
        n_classes = first.shape[1]
        out = np.empty((mat.shape[0], n_classes), dtype=np.float64)
        offset = 0
        while offset < mat.shape[0]:
            proba = np.asarray(
                est.predict_proba(mat[offset : offset + PREDICT_CHUNK_SIZE]), dtype=np.float64
            )
            if proba.ndim != 2 or proba.shape[1] != n_classes:
                return None
            out[offset : offset + proba.shape[0]] = proba
            del proba
            offset += PREDICT_CHUNK_SIZE
        if not np.isfinite(out).any():
            return None
        return np.clip(np.where(np.isfinite(out), out, 0.0), 0.0, 1.0)
    except Exception:
        return None


def _marching_squares_segments(
    xs: np.ndarray,
    ys: np.ndarray,
    field: np.ndarray,
    threshold: float,
) -> list[list[float]]:
    """Extract ``[x1, y1, x2, y2]`` isoline segments at ``threshold``.

    Classic 16-case Marching Squares with linear edge interpolation:

        p_edge = p1 + (τ − S1) / (S2 − S1) · (p2 − p1)

    ``xs``/``ys`` are the grid axes (already in world coords), ``field`` is
    the ``(Ny, Nx)`` scalar field. Degenerate cells (``0`` / ``15``) and
    zero-denominator edges are skipped so constant fields yield no segments.
    """
    segments: list[list[float]] = []
    ny, nx = field.shape
    if ny < 2 or nx < 2:
        return segments
    thr = float(threshold)
    above = field >= thr
    # Vectorized 4-bit cell codes; only crossing cells (code not in {0, 15})
    # need the Python interpolation loop — typically O(res), not O(res²).
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
        t = (thr - v_a) / denom
        t = min(1.0, max(0.0, t))
        return p_a + t * (p_b - p_a)

    for j, i in zip(cross_j.tolist(), cross_i.tolist()):
        s0 = float(fld[j, i])
        s1 = float(fld[j, i + 1])
        s2 = float(fld[j + 1, i + 1])
        s3 = float(fld[j + 1, i])
        x0 = float(xs_f[i])
        x1 = float(xs_f[i + 1])
        y0 = float(ys_f[j])
        y1 = float(ys_f[j + 1])
        # Edge crossings: top (s0-s1), right (s1-s2), bottom (s3-s2), left (s0-s3).
        top = (_interp(x0, s0, x1, s1), y0) if (s0 >= thr) != (s1 >= thr) else None
        right = (x1, _interp(y0, s1, y1, s2)) if (s1 >= thr) != (s2 >= thr) else None
        bottom = (_interp(x0, s3, x1, s2), y1) if (s3 >= thr) != (s2 >= thr) else None
        left = (x0, _interp(y0, s0, y1, s3)) if (s0 >= thr) != (s3 >= thr) else None
        pts = [p for p in (top, right, bottom, left) if p is not None]
        if len(pts) == 2:
            segments.append([round(pts[0][0], 3), round(pts[0][1], 3), round(pts[1][0], 3), round(pts[1][1], 3)])
        elif len(pts) == 4:
            # Ambiguous saddle (cases 5 / 10): connect pairs in edge order.
            segments.append([round(pts[0][0], 3), round(pts[0][1], 3), round(pts[1][0], 3), round(pts[1][1], 3)])
            segments.append([round(pts[2][0], 3), round(pts[2][1], 3), round(pts[3][0], 3), round(pts[3][1], 3)])
    return segments


class Contour2DGenerator:
    """Stateless dense-grid contour builder (see module docstring)."""

    @staticmethod
    def detect_task(model_obj: Any, final_est: Any) -> tuple[str, list[Any]]:
        """Return ``(prediction_task, classes)`` for the fitted estimator."""
        est = final_est if final_est is not None else model_obj
        classes_attr = getattr(est, "classes_", None)
        if classes_attr is not None:
            try:
                classes = [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(classes_attr).ravel())]
            except Exception:
                classes = []
            if len(classes) >= 2:
                return "classification", classes
        if hasattr(model_obj, "predict_proba") or hasattr(final_est, "predict_proba"):
            return "classification", []
        return "regression", []

    @staticmethod
    def generate_contour(
        *,
        adapter: Any,
        model_obj: Any,
        final_est: Any,
        X: np.ndarray,
        x_feature_idx: int,
        y_feature_idx: int,
        x_feature_name: str,
        y_feature_name: str,
        grid_resolution: int = DEFAULT_GRID_RESOLUTION,
        pinning_mode: str = "median",
        current_row_values: np.ndarray | None = None,
        use_robust_range: bool = False,
        include_heatmap: bool = True,
    ) -> Contour2DResponse:
        started = time.perf_counter()
        res = max(16, min(200, int(grid_resolution)))
        ref = np.asarray(X, dtype=np.float32)

        grid, (x_min, x_max), (y_min, y_max) = build_slice_grid_matrix(
            ref,
            int(x_feature_idx),
            int(y_feature_idx),
            res,
            pinning_mode,  # type: ignore[arg-type]
            current_row_values=current_row_values,
            use_robust_range=use_robust_range,
        )

        prediction_task, classes = Contour2DGenerator.detect_task(model_obj, final_est)

        x_scale = WorldCoordinateScaler.build_axis_scale(x_feature_name, ref[:, int(x_feature_idx)])
        y_scale = WorldCoordinateScaler.build_axis_scale(y_feature_name, ref[:, int(y_feature_idx)])
        gx_raw = np.linspace(x_min, x_max, res, dtype=np.float32)
        gy_raw = np.linspace(y_min, y_max, res, dtype=np.float32)
        gx_w = WorldCoordinateScaler.to_world_coords(gx_raw, x_scale).astype(np.float64)
        gy_w = WorldCoordinateScaler.to_world_coords(gy_raw, y_scale).astype(np.float64)

        contours: list[ContourLevel2D] = []
        response_curve: ResponseCurve1D | None = None

        # Phase 42 — exact analytic linear overlay (None for non-linear /
        # pipelined models; never allowed to break the sampled contour path).
        linear_2d: dict[str, Any] | None = None
        if extract_linear_2d_geometry is not None:
            try:
                geometry = extract_linear_2d_geometry(
                    model_obj=model_obj,
                    final_est=final_est,
                    X=ref,
                    x_feature_idx=int(x_feature_idx),
                    y_feature_idx=int(y_feature_idx),
                    x_feature_name=x_feature_name,
                    y_feature_name=y_feature_name,
                    x_range_raw=(float(x_min), float(x_max)),
                    y_range_raw=(float(y_min), float(y_max)),
                    x_scale=x_scale,
                    y_scale=y_scale,
                    pinning_mode=str(pinning_mode),
                    current_row_values=current_row_values,
                )
                linear_2d = geometry.model_dump(mode="json") if geometry is not None else None
            except Exception:
                linear_2d = None

        if prediction_task == "classification":
            proba = _chunked_proba_matrix(model_obj, final_est, grid)
            if proba is None:
                # No predict_proba (e.g. LinearSVC): fall back to label field.
                labels = _chunked_predict(adapter, model_obj, grid)
                field = labels.reshape((res, res))
                heat = field
                levels = sorted(set(float(v) for v in np.unique(field).tolist()))
                for k, lv in enumerate(levels):
                    if k == 0:
                        continue
                    segs = _marching_squares_segments(gx_w, gy_w, field, (lv + levels[k - 1]) / 2.0)
                    if segs:
                        contours.append(ContourLevel2D(threshold=round(float(lv), 4), class_index=k, kind="decision", line_segments=segs))
                prediction_task = "classification"
            elif proba.shape[1] == 2:
                field = proba[:, 1].reshape((res, res))
                heat = field
                boundary = _marching_squares_segments(gx_w, gy_w, field, DECISION_THRESHOLD)
                if boundary:
                    contours.append(ContourLevel2D(threshold=DECISION_THRESHOLD, class_index=1, kind="decision", line_segments=boundary))
                for lv in CONFIDENCE_LEVELS:
                    segs = _marching_squares_segments(gx_w, gy_w, field, lv)
                    if segs:
                        contours.append(ContourLevel2D(threshold=lv, class_index=1, kind="confidence", line_segments=segs))
            else:
                winners = np.argmax(proba, axis=1).reshape((res, res)).astype(np.float64)
                max_proba = np.max(proba, axis=1).reshape((res, res))
                heat = max_proba
                n_classes = proba.shape[1]
                for k in range(n_classes):
                    mask = (winners == k).astype(np.float64)
                    segs = _marching_squares_segments(gx_w, gy_w, mask, 0.5)
                    if segs:
                        contours.append(ContourLevel2D(threshold=0.5, class_index=k, kind="decision", line_segments=segs))
                for lv in CONFIDENCE_LEVELS:
                    segs = _marching_squares_segments(gx_w, gy_w, max_proba, lv)
                    if segs:
                        contours.append(ContourLevel2D(threshold=lv, class_index=-1, kind="confidence", line_segments=segs))
        else:
            values = _chunked_predict(adapter, model_obj, grid)
            field = values.reshape((res, res))
            heat = field
            finite = field[np.isfinite(field)]
            if finite.size:
                quantiles = np.quantile(finite, [0.2, 0.4, 0.6, 0.8]).tolist()
                for q in dict.fromkeys(round(float(v), 4) for v in quantiles):
                    segs = _marching_squares_segments(gx_w, gy_w, field, q)
                    if segs:
                        contours.append(ContourLevel2D(threshold=q, class_index=0, kind="elevation", line_segments=segs))
                # 1D feature-vs-target response curve along X at pinned background.
                try:
                    n_curve = 120
                    curve_x_raw = np.linspace(float(x_min), float(x_max), n_curve, dtype=np.float32)
                    base = np.nanmedian(ref, axis=0).astype(np.float32)
                    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    curve_mat = np.tile(base, (n_curve, 1))
                    curve_mat[:, int(x_feature_idx)] = curve_x_raw
                    curve_pred = _chunked_predict(adapter, model_obj, np.ascontiguousarray(curve_mat, dtype=np.float32))
                    response_curve = ResponseCurve1D(
                        x_values_raw=[round(float(v), 4) for v in curve_x_raw.tolist()],
                        y_pred=[round(float(v), 4) for v in curve_pred.tolist()],
                    )
                    del curve_mat, curve_pred
                except Exception:
                    response_curve = None
            else:
                response_curve = None

        heat_arr = np.asarray(heat, dtype=np.float64)
        finite_heat = heat_arr[np.isfinite(heat_arr)]
        h_min = float(np.min(finite_heat)) if finite_heat.size else 0.0
        h_max = float(np.max(finite_heat)) if finite_heat.size else 1.0

        heatmap_quantized: list[int] = []
        if include_heatmap:
            flat = np.nan_to_num(heat_arr.ravel(), nan=float(h_min), posinf=float(h_max), neginf=float(h_min))
            span = float(h_max - h_min)
            if span < 1e-12:
                heatmap_quantized = [128] * int(flat.size)
            else:
                norm = np.clip((flat - float(h_min)) / span, 0.0, 1.0)
                heatmap_quantized = (np.rint(norm * 255.0).astype(np.uint8)).tolist()

        del grid
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return Contour2DResponse(
            x_feature=x_feature_name,
            y_feature=y_feature_name,
            grid_resolution=res,
            x_range_raw=(float(x_min), float(x_max)),
            y_range_raw=(float(y_min), float(y_max)),
            prediction_task=prediction_task,
            classes=list(classes),
            heatmap_min=h_min,
            heatmap_max=h_max,
            heatmap_quantized=heatmap_quantized,
            contours=contours,
            response_curve_1d=response_curve,
            linear_2d=linear_2d,
            computation_time_ms=elapsed_ms,
        )

"""Phase 08–09 — Normalized World Cube math (Shared Kernel).

Maps real-world feature/prediction values (arbitrary scales, e.g.
``area_sqm: 22..450`` vs ``price: 1M..18M``) into the standard Three.js
world cube ``[-5.0, +5.0]`` so no axis collapses to a thread while another
clips the camera frustum. Raw values are preserved separately in
``AxisScaleMeta.tick_raw_labels`` for axis tick labels / tooltips.

Phase 09 adds the projection math core: safe min-max normalization,
axis-explorer slice-grid synthesis (median/mean/zero/current-row pinning),
robust percentile ranges, and the StandardScaler + PCA(3) manifold.

Memory: pure float32 NumPy, no DataFrame retention. A 60x60 surface grid
(3,600 points) costs ~14 KB; wide-column grids are evaluated in mini-batches.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

WORLD_MIN: float = -5.0
WORLD_MAX: float = 5.0
WORLD_SPAN: float = 10.0

#: Phase 09 alias used by the projection specs (same cube half-size).
WORLD_BOUNDS_HALF_SIZE: float = 5.0

#: Hard frustum guard — world coords are clipped here so extreme outliers
#: never fly the active ball out of the camera view (edge case #4).
CLAMP_MIN: float = WORLD_MIN - 2.0
CLAMP_MAX: float = WORLD_MAX + 2.0

_ZERO_SPAN_EPSILON = 1e-7
_SAFE_SPAN_EPSILON = 1e-9


class AxisScaleMeta(BaseModel):
    axis_name: str
    feature_index: int | None = None
    raw_min: float
    raw_max: float
    tick_world_positions: list[float] = Field(default_factory=list)
    tick_raw_labels: list[float] = Field(default_factory=list)


class WorldCoordinateScaler:
    """Stateless normalizer: raw float32 values <-> [-5, +5] world cube."""

    @staticmethod
    def build_axis_scale(
        name: str,
        values: np.ndarray,
        feature_idx: int | None = None,
        n_ticks: int = 5,
    ) -> AxisScaleMeta:
        arr = np.asarray(values, dtype=np.float32).ravel()
        finite = arr[np.isfinite(arr)]
        # NaN-only / empty column (edge: unmapped axis) -> neutral [0, 1].
        if finite.size == 0:
            raw_min, raw_max = 0.0, 1.0
        else:
            raw_min = float(np.min(finite))
            raw_max = float(np.max(finite))
        if abs(raw_max - raw_min) < _ZERO_SPAN_EPSILON:
            # Zero-span guard (edge case #3): constant column must not
            # divide by zero / emit NaN into BufferAttribute.
            raw_min -= 1.0
            raw_max += 1.0

        world_ticks = np.linspace(WORLD_MIN, WORLD_MAX, n_ticks, dtype=np.float32)
        raw_ticks = np.linspace(raw_min, raw_max, n_ticks, dtype=np.float32)
        return AxisScaleMeta(
            axis_name=name,
            feature_index=feature_idx,
            raw_min=raw_min,
            raw_max=raw_max,
            tick_world_positions=[round(float(w), 3) for w in world_ticks],
            tick_raw_labels=[round(float(r), 4) for r in raw_ticks],
        )

    @staticmethod
    def to_world_coords(values: np.ndarray, scale: AxisScaleMeta) -> np.ndarray:
        span = max(scale.raw_max - scale.raw_min, _ZERO_SPAN_EPSILON)
        norm_01 = (np.asarray(values, dtype=np.float32) - np.float32(scale.raw_min)) / np.float32(span)
        world = np.float32(WORLD_MIN) + norm_01 * np.float32(WORLD_SPAN)
        return np.clip(world, CLAMP_MIN, CLAMP_MAX).astype(np.float32)

    @staticmethod
    def to_raw_value(world: float, scale: AxisScaleMeta) -> float:
        """Inverse map (for tooltips / click-to-scrub readouts)."""
        t = (float(world) - WORLD_MIN) / WORLD_SPAN
        return float(scale.raw_min + t * (scale.raw_max - scale.raw_min))


# ---------------------------------------------------------------------------
# Phase 09 — projection math core (pure functions, float32 throughout)
# ---------------------------------------------------------------------------

PinningMode = Literal["median", "mean", "zero", "current_row"]


def safe_min_max_normalize(
    values: np.ndarray,
    v_min: float,
    v_max: float,
    target_min: float = -WORLD_BOUNDS_HALF_SIZE,
    target_max: float = WORLD_BOUNDS_HALF_SIZE,
) -> np.ndarray:
    """Map values into the 3D stage range with a zero-variance guard.

    Edge case #1: a constant column (``v_min == v_max``) returns all zeros
    instead of raising ``ZeroDivisionError`` or emitting ``NaN`` into a
    ``THREE.BufferAttribute``.
    """
    span = float(v_max - v_min)
    if abs(span) < _SAFE_SPAN_EPSILON:
        return np.zeros_like(np.asarray(values), dtype=np.float32)
    normalized_01 = (np.asarray(values, dtype=np.float32) - np.float32(v_min)) / np.float32(span)
    normalized_01 = np.clip(normalized_01, -0.5, 1.5)  # slight out-of-bounds overshoot allowed
    world = normalized_01 * np.float32(target_max - target_min) + np.float32(target_min)
    return np.clip(world, np.float32(CLAMP_MIN), np.float32(CLAMP_MAX)).astype(np.float32)


def robust_percentile_range(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> tuple[float, float]:
    """P01–P99 data range so one extreme outlier cannot flatten the surface (edge #5)."""
    arr = np.asarray(values, dtype=np.float32).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(finite, low_q))
    hi = float(np.nanpercentile(finite, high_q))
    if abs(hi - lo) < _SAFE_SPAN_EPSILON:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def build_slice_grid_matrix(
    reference_X: np.ndarray,
    x_col_idx: int,
    y_col_idx: int,
    grid_resolution: int,
    pinning_mode: PinningMode = "median",
    current_row_values: np.ndarray | None = None,
    use_robust_range: bool = False,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Build a ``(res*res, D)`` float32 grid matrix for model surface prediction.

    Non-axis columns are pinned (median / mean / zero / current-row values);
    the X/Y columns sweep their data range (optionally the robust P01–P99
    range). Edge case #3: when both axes address the same column the second
    write is skipped so the plane never collapses to a 1-D line.
    """
    ref = np.asarray(reference_X, dtype=np.float32)
    if ref.ndim != 2 or ref.shape[1] == 0:
        raise ValueError("reference_X must be a non-empty 2-D float32 matrix")
    num_samples, num_features = ref.shape
    if not (0 <= x_col_idx < num_features and 0 <= y_col_idx < num_features):
        raise ValueError(f"axis column out of range for {num_features} features")
    res = max(2, int(grid_resolution))
    total_grid_points = res * res

    if pinning_mode == "median":
        base_vector = np.nanmedian(ref, axis=0).astype(np.float32)
    elif pinning_mode == "mean":
        with np.errstate(invalid="ignore"):
            base_vector = np.nanmean(ref, axis=0).astype(np.float32)
    elif pinning_mode == "current_row" and current_row_values is not None:
        row = np.asarray(current_row_values, dtype=np.float32).ravel()
        base_vector = np.zeros(num_features, dtype=np.float32)
        n = min(num_features, row.size)
        base_vector[:n] = row[:n]
    else:
        base_vector = np.zeros(num_features, dtype=np.float32)
    base_vector = np.nan_to_num(base_vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _finite_range(col: np.ndarray) -> tuple[float, float]:
        lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 0.0, 1.0
        return lo, hi

    if use_robust_range:
        x_min, x_max = robust_percentile_range(ref[:, x_col_idx])
        y_min, y_max = robust_percentile_range(ref[:, y_col_idx])
    else:
        x_min, x_max = _finite_range(ref[:, x_col_idx])
        y_min, y_max = _finite_range(ref[:, y_col_idx])
    if abs(x_max - x_min) < _SAFE_SPAN_EPSILON:
        x_min -= 1.0
        x_max += 1.0
    if abs(y_max - y_min) < _SAFE_SPAN_EPSILON:
        y_min -= 1.0
        y_max += 1.0

    x_lin = np.linspace(x_min, x_max, res, dtype=np.float32)
    y_lin = np.linspace(y_min, y_max, res, dtype=np.float32)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")

    grid_matrix = np.tile(base_vector, (total_grid_points, 1))
    grid_matrix[:, x_col_idx] = xx.ravel()
    if x_col_idx != y_col_idx:
        grid_matrix[:, y_col_idx] = yy.ravel()
    return np.ascontiguousarray(grid_matrix, dtype=np.float32), (x_min, x_max), (y_min, y_max)


def compute_3d_pca_projection(
    X: np.ndarray,
    *,
    return_fitted: bool = False,
    max_fit_rows: int = 20_000,
) -> tuple[np.ndarray, list[float], np.ndarray] | tuple[np.ndarray, list[float], np.ndarray, Any, Any]:
    """StandardScaler + PCA(n_components=3) manifold, normalized to float32 world coords.

    Edge case #2: ``D < 3`` (or fewer than 2 samples) pads the missing axes
    with ``0.0`` instead of raising ``ValueError``.

    Memory: the PCA fit runs on a deterministic stride subsample of at most
    ``max_fit_rows`` rows; callers transform the full matrix in chunks with
    the returned fitted objects (``return_fitted=True`` appends
    ``(scaler, pca)`` — ``(None, None)`` on the degenerate path).
    """
    from typing import Any as _Any

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    mat = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, n_features = mat.shape
    empty: tuple[np.ndarray, list[float], np.ndarray, _Any, _Any] | tuple[np.ndarray, list[float], np.ndarray]
    if n_samples < 2 or n_features == 0:
        base = (np.zeros((n_samples, 3), dtype=np.float32), [0.0, 0.0, 0.0], np.zeros((0, n_features), dtype=np.float32))
        empty = (*base, None, None) if return_fitted else base
        return empty

    if n_samples > max_fit_rows:
        stride = max(1, n_samples // max_fit_rows)
        fit_mat = mat[::stride][:max_fit_rows]
    else:
        fit_mat = mat
    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_mat)

    n_comp = min(3, n_features, fit_mat.shape[0])
    pca = PCA(n_components=n_comp, svd_solver="auto", random_state=42)
    pca.fit(fit_scaled)
    coords_raw = pca.transform(scaler.transform(mat)).astype(np.float32)

    coords_3d = np.zeros((n_samples, 3), dtype=np.float32)
    for dim in range(n_comp):
        col = coords_raw[:, dim]
        coords_3d[:, dim] = safe_min_max_normalize(col, float(col.min()), float(col.max()))

    variance_ratios = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    while len(variance_ratios) < 3:
        variance_ratios.append(0.0)

    components = np.asarray(pca.components_, dtype=np.float32)
    if return_fitted:
        return coords_3d, variance_ratios, components, scaler, pca
    return coords_3d, variance_ratios, components

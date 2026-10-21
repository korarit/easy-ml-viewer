"""Phase 09 — projection math unit tests (dev spec §4.1, test plan §2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.math_3d import (
    WORLD_BOUNDS_HALF_SIZE,
    build_slice_grid_matrix,
    compute_3d_pca_projection,
    robust_percentile_range,
    safe_min_max_normalize,
)
from app.domains.spatial_projection.value_objects import ProjectionConfigVO


def test_safe_min_max_normalize_standard_range():
    arr = np.array([10.0, 55.0, 100.0], dtype=np.float64)
    norm = safe_min_max_normalize(arr, v_min=10.0, v_max=100.0)

    assert norm.dtype == np.float32
    assert np.isclose(norm[0], -WORLD_BOUNDS_HALF_SIZE, atol=1e-5)
    assert np.isclose(norm[1], 0.0, atol=1e-5)
    assert np.isclose(norm[2], WORLD_BOUNDS_HALF_SIZE, atol=1e-5)


def test_safe_min_max_normalize_constant_column_zero_variance():
    arr = np.full(100, 42.0, dtype=np.float64)
    norm = safe_min_max_normalize(arr, v_min=42.0, v_max=42.0)

    assert norm.dtype == np.float32
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()
    assert np.all(norm == 0.0)


@pytest.mark.parametrize(
    "lod_level,expected_res,expected_points",
    [
        ("low", 30, 900),
        ("standard", 60, 3600),
        ("high", 100, 10000),
    ],
)
def test_build_slice_grid_matrix_lod_resolutions_and_pinning(lod_level, expected_res, expected_points):
    cfg = ProjectionConfigVO(x_feature="f0", y_feature="f1", lod_level=lod_level, pinning_mode="median")
    assert cfg.grid_resolution == expected_res

    rng = np.random.default_rng(42)
    X_ref = rng.normal(loc=10.0, scale=2.0, size=(200, 5)).astype(np.float64)
    expected_f2_median = np.float32(np.nanmedian(X_ref[:, 2]))

    grid_mat, x_range, y_range = build_slice_grid_matrix(
        reference_X=X_ref,
        x_col_idx=0,
        y_col_idx=1,
        grid_resolution=cfg.grid_resolution,
        pinning_mode="median",
    )

    assert grid_mat.shape == (expected_points, 5)
    assert grid_mat.dtype == np.float32
    assert np.allclose(grid_mat[:, 2], expected_f2_median, atol=1e-5)
    assert x_range[0] < x_range[1]
    assert y_range[0] < y_range[1]


def test_build_slice_grid_matrix_current_row_pinning():
    X_ref = np.ones((50, 4), dtype=np.float32)
    current_row = np.array([1.0, 2.0, 99.5, -45.25], dtype=np.float32)

    grid_mat, _, _ = build_slice_grid_matrix(
        reference_X=X_ref,
        x_col_idx=0,
        y_col_idx=1,
        grid_resolution=30,
        pinning_mode="current_row",
        current_row_values=current_row,
    )

    assert np.allclose(grid_mat[:, 2], 99.5)
    assert np.allclose(grid_mat[:, 3], -45.25)


def test_compute_3d_pca_projection_with_fewer_than_3_features():
    X_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 1.0]], dtype=np.float32)
    coords_3d, variance_ratios, components = compute_3d_pca_projection(X_2d)

    assert coords_3d.shape == (4, 3)
    assert coords_3d.dtype == np.float32
    assert np.all(coords_3d[:, 2] == 0.0)
    assert len(variance_ratios) == 3
    assert variance_ratios[2] == 0.0


def test_compute_3d_pca_variance_sums_to_one_for_full_rank():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    coords, variance, _ = compute_3d_pca_projection(X)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()
    assert 0.0 < sum(variance) <= 1.0 + 1e-6
    assert all(v >= 0.0 for v in variance)


def test_nan_and_inf_inputs_never_leak_into_outputs():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    X[::10, 0] = np.nan
    X[::7, 1] = np.inf
    grid_mat, _, _ = build_slice_grid_matrix(X, 0, 1, grid_resolution=10, pinning_mode="median")
    assert np.isfinite(grid_mat).all()
    coords, variance, _ = compute_3d_pca_projection(X)
    assert np.isfinite(coords).all()
    assert all(np.isfinite(v) for v in variance)
    # Contract: NaN entries are sanitized upstream (nan_to_num); the
    # normalizer itself maps finite inputs to finite world coords.
    clean = np.nan_to_num(X[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    norm = safe_min_max_normalize(clean, float(np.nanmin(X[:, 0])), float(np.nanmax(X[:, 0])))
    assert not np.isnan(norm).any()


def test_robust_percentile_range_ignores_single_extreme_outlier():
    vals = np.concatenate([np.linspace(0, 100, 999), [1e9]]).astype(np.float32)
    lo, hi = robust_percentile_range(vals)
    assert lo >= 0.0
    assert hi <= 101.0
    assert hi < 1e9


def test_same_column_axes_do_not_collapse():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    grid_mat, _, _ = build_slice_grid_matrix(X, 0, 0, grid_resolution=8, pinning_mode="zero")
    assert grid_mat.shape == (64, 2)
    assert np.isfinite(grid_mat).all()

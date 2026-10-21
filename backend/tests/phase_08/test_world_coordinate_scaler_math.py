"""Phase 08 — WorldCoordinateScaler unit tests (dev spec §4.2, test plan §2.2)."""

from __future__ import annotations

import numpy as np

from app.core.math_3d import WORLD_MAX, WORLD_MIN, WorldCoordinateScaler


def test_world_coordinate_normalization_maps_min_max_to_minus5_plus5() -> None:
    prices = np.array([1_000_000.0, 5_500_000.0, 10_000_000.0], dtype=np.float32)
    scale = WorldCoordinateScaler.build_axis_scale("selling_price", prices, feature_idx=None, n_ticks=5)

    assert scale.raw_min == 1_000_000.0
    assert scale.raw_max == 10_000_000.0
    assert scale.tick_world_positions == [-5.0, -2.5, 0.0, 2.5, 5.0]
    assert scale.tick_raw_labels[0] == 1_000_000.0
    assert scale.tick_raw_labels[-1] == 10_000_000.0

    world_coords = WorldCoordinateScaler.to_world_coords(prices, scale)
    assert world_coords.dtype == np.float32
    assert np.allclose(world_coords, [-5.0, 0.0, 5.0], atol=1e-4)


def test_zero_span_constant_feature_does_not_divide_by_zero() -> None:
    constant_vals = np.array([42.0, 42.0, 42.0], dtype=np.float32)
    scale = WorldCoordinateScaler.build_axis_scale("const_col", constant_vals, feature_idx=0)

    assert scale.raw_min == 41.0
    assert scale.raw_max == 43.0
    world_coords = WorldCoordinateScaler.to_world_coords(constant_vals, scale)
    assert not np.isnan(world_coords).any()
    assert np.allclose(world_coords, [0.0, 0.0, 0.0], atol=1e-4)


def test_extreme_outlier_clamped_within_safe_frustum_bounds() -> None:
    normal_vals = np.array([0.0, 100.0], dtype=np.float32)
    scale = WorldCoordinateScaler.build_axis_scale("x", normal_vals, feature_idx=0)

    extreme_vals = np.array([-5000.0, 50.0, 9000.0], dtype=np.float32)
    world_coords = WorldCoordinateScaler.to_world_coords(extreme_vals, scale)
    assert np.allclose(world_coords, [-7.0, 0.0, 7.0], atol=1e-4)


def test_world_bounds_constants() -> None:
    assert WORLD_MIN == -5.0
    assert WORLD_MAX == 5.0


def test_nan_only_column_falls_back_to_neutral_scale() -> None:
    vals = np.array([np.nan, np.nan], dtype=np.float32)
    scale = WorldCoordinateScaler.build_axis_scale("empty", vals, feature_idx=1)
    assert scale.raw_min == 0.0
    assert scale.raw_max == 1.0
    world = WorldCoordinateScaler.to_world_coords(np.array([0.5], dtype=np.float32), scale)
    assert np.allclose(world, [0.0], atol=1e-4)

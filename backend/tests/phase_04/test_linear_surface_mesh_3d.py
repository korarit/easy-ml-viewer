"""Phase 04 — 3D hyperplane surface mesh tests (50x50, 1D ribbon, perf)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LinearRegression

from app.domains.model_engine.adapters.linear.linear_regression_adapter import (
    LinearRegressionAdapter,
)
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType


def test_compute_3d_surface_hyperplane_and_1d_model_ribbon() -> None:
    X_1d = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    y_1d = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    model_1d = LinearRegression().fit(X_1d, y_1d)

    adapter = LinearRegressionAdapter()
    spec = GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=0,
        x_min=10.0,
        x_max=30.0,
        y_min=-5.0,
        y_max=5.0,
        resolution=50,
        fixed_feature_medians={},
    )
    surface = adapter.compute_3d_surface(model_1d, spec, n_features=1)
    assert surface.visual_type == Visual3DType.FLAT_HYPERPLANE
    assert surface.resolution == 50
    assert len(surface.x_grid) == 50
    assert len(surface.y_grid) == 50
    assert len(surface.z_values) == 2500


def test_surface_generation_under_10ms_and_float32_contiguous() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((100, 20), dtype=np.float32)
    y = rng.random(100).astype(np.float32)
    model = LinearRegression().fit(X, y)
    adapter = LinearRegressionAdapter()

    spec = GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=1,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        resolution=50,
        fixed_feature_medians={i: 0.5 for i in range(2, 20)},
    )

    t0 = time.perf_counter()
    surface = adapter.compute_3d_surface(model, spec, n_features=20)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 10.0
    assert len(surface.z_values) == 2500


def test_surface_resolution_clamped_to_100() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float32)
    y = np.array([0.0, 2.0, 2.5], dtype=np.float32)
    model = LinearRegression().fit(X, y)
    adapter = LinearRegressionAdapter()
    spec = GridSpec3D(
        axis_x_feature_index=0,
        axis_y_feature_index=1,
        x_min=0.0,
        x_max=2.0,
        y_min=0.0,
        y_max=1.0,
        resolution=200,  # over limit -> clamped to 100
        fixed_feature_medians={},
    )
    surface = adapter.compute_3d_surface(model, spec, n_features=2)
    assert surface.resolution == 100
    assert len(surface.z_values) == 100 * 100

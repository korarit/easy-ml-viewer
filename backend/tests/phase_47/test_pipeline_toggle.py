"""Phase 47 — Pipeline Raw <-> Scaled toggle: exactness + honest fallback."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.domains.model_engine.adapters.neural_onnx.neural_2d_extractor import (
    Neural2DExtractor,
)

from conftest import make_ctx


def test_standard_scaler_toggle_exact(fitted_standard_pipeline, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=fitted_standard_pipeline, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and geo.family == "pipeline"
    assert geo.pipeline is not None and geo.pipeline.has_toggle
    assert geo.pipeline.scaler_kind == "standard"
    scaler = fitted_standard_pipeline.named_steps["scaler"]
    assert geo.pipeline.mu_x == float(scaler.mean_[0])
    assert geo.pipeline.sigma_x == float(scaler.scale_[0])
    # Scaled range must equal (raw - mu) / sigma exactly.
    assert geo.pipeline.scaled_x_range is not None
    assert abs(geo.pipeline.scaled_x_range[0] - (xr[0] - geo.pipeline.mu_x) / geo.pipeline.sigma_x) < 1e-3
    assert abs(geo.pipeline.scaled_x_range[1] - (xr[1] - geo.pipeline.mu_x) / geo.pipeline.sigma_x) < 1e-3


def test_minmax_scaler_toggle_exact(fitted_minmax_pipeline, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=fitted_minmax_pipeline, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and geo.pipeline is not None
    assert geo.pipeline.has_toggle and geo.pipeline.scaler_kind == "minmax"


def test_column_transformer_honest_fallback():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=120), rng.integers(0, 3, size=120).astype(float)])
    y = (X[:, 0] > 0).astype(int)
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), [0]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [1]),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)
    from app.core.math_3d import WorldCoordinateScaler

    xs = WorldCoordinateScaler.build_axis_scale("f0", X[:, 0])
    ys = WorldCoordinateScaler.build_axis_scale("f1", X[:, 1])
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=pipe, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=(float(X[:, 0].min()), float(X[:, 0].max())),
        y_range_raw=(float(X[:, 1].min()), float(X[:, 1].max())),
        x_scale=xs, y_scale=ys)
    assert geo is not None and geo.family == "pipeline"
    assert geo.pipeline is not None and not geo.pipeline.has_toggle

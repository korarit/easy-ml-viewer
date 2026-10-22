"""Phase 47 — MLP first-layer hyperplanes: exactness + Dead ReLU flags."""

from __future__ import annotations

import numpy as np

from app.domains.model_engine.adapters.neural_onnx.neural_2d_extractor import (
    Neural2DExtractor,
)

from conftest import make_ctx


def test_mlp_hyperplane_count_matches_hidden(fitted_mlp, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=fitted_mlp, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and geo.family == "mlp"
    assert len(geo.hyperplanes) == int(fitted_mlp.coefs_[0].shape[1])
    assert not geo.hyperplanes_truncated
    assert geo.dead_count + geo.active_count == len(geo.hyperplanes)


def test_mlp_hyperplane_equation_exact(fitted_mlp, blobs_2d):
    """Every clipped endpoint must satisfy z_k = 0 within float tolerance."""
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=fitted_mlp, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None
    for h in geo.hyperplanes:
        if h.segment_raw is None:
            continue
        x1, y1, x2, y2 = h.segment_raw
        assert abs(h.w1 * x1 + h.w2 * y1 + h.b_eff) < 1e-3
        assert abs(h.w1 * x2 + h.w2 * y2 + h.b_eff) < 1e-3
        assert h.segment_world is not None
        assert all(-5.5 <= v <= 5.5 for v in h.segment_world)


def test_mlp_dead_flag_matches_corners(fitted_mlp, blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=fitted_mlp, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and geo.activation == "relu"
    corners = [(xr[0], yr[0]), (xr[0], yr[1]), (xr[1], yr[0]), (xr[1], yr[1])]
    for h in geo.hyperplanes:
        zs = [h.w1 * cx + h.w2 * cy + h.b_eff for cx, cy in corners]
        if max(zs) < 0:
            assert h.is_dead
        if min(zs) > 0:
            assert h.is_always_active
    assert 0.0 <= (geo.active_ratio or 0.0) <= 1.0


def test_mlp_prototype_signature(fitted_mlp, blobs_2d):
    """Spec §5.2 prototype returns raw (w1, w2, b_eff) per neuron."""
    X, _ = blobs_2d
    pinned = np.median(X, axis=0)
    planes = Neural2DExtractor.extract_mlp_first_layer_hyperplanes(
        fitted_mlp, 0, 1, pinned)
    assert len(planes) == int(fitted_mlp.coefs_[0].shape[1])
    assert set(planes[0].keys()) == {"neuron_idx", "w1", "w2", "b_eff"}

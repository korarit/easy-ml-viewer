"""Phase 47 — ONNX classical summary + deep latent SVD projection plane."""

from __future__ import annotations

import numpy as np

from app.domains.model_engine.adapters.neural_onnx.neural_2d_extractor import (
    Neural2DExtractor,
    project_latent_svd,
)


class _FakeNode:
    def __init__(self, domain, op_type):
        self.domain = domain
        self.op_type = op_type


class _FakeGraph:
    def __init__(self, nodes):
        self.node = nodes
        self.input = []


class _FakeProto:
    def __init__(self, nodes):
        self.graph = _FakeGraph(nodes)


def _ctx(X):
    from conftest import make_ctx

    return make_ctx(X)


def test_onnx_classical_summary(blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = _ctx(X)
    proto = _FakeProto([
        _FakeNode("ai.onnx.ml", "TreeEnsembleClassifier"),
        _FakeNode("ai.onnx.ml", "ZipMap"),
    ])
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=proto, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    assert geo is not None and geo.family == "onnx_classical"
    assert geo.onnx is not None
    assert geo.onnx.operator == "TreeEnsembleClassifier"
    assert not geo.onnx.latent_available
    assert geo.onnx.n_nodes == 2


def test_latent_svd_projection_plane():
    rng = np.random.default_rng(11)
    H = np.vstack([rng.normal(loc=-3.0, scale=0.5, size=(60, 8)),
                   rng.normal(loc=+3.0, scale=0.5, size=(60, 8))])
    pts, evr = project_latent_svd(H)
    assert pts.shape == (120, 2) and pts.dtype == np.float32
    assert np.all(np.isfinite(pts))
    assert len(evr) == 2 and evr[0] >= evr[1] >= 0.0 and sum(evr) <= 1.0 + 1e-6
    assert evr[0] > 0.9  # first axis dominates this two-cluster fixture
    # Two clusters must stay separated along the first SVD axis.
    assert float(np.mean(pts[60:, 0]) - np.mean(pts[:60, 0])) > 1.0


def test_latent_svd_degenerate_safe():
    H = np.ones((10, 6), dtype=np.float64)
    pts, evr = project_latent_svd(H)
    assert pts.shape == (10, 2)
    assert np.all(np.isfinite(pts)) and evr == [0.0, 0.0]


def test_onnx_deep_fallback_without_ort(blobs_2d):
    X, _ = blobs_2d
    xs, ys, xr, yr = _ctx(X)
    proto = _FakeProto([_FakeNode("", "Gemm"), _FakeNode("", "Relu"), _FakeNode("", "Gemm")])
    geo = Neural2DExtractor.extract_neural_2d_geometry(
        model_obj=proto, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)
    # With a fake proto (no ORT session), geometry degrades to the honest
    # operator summary instead of raising.
    assert geo is not None and geo.family == "onnx_deep"
    assert geo.onnx is not None and geo.onnx.n_nodes == 3

"""Phase 41 — contour-2d engine unit tests (Marching Squares + grid inference)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.domains.model_engine.value_objects import PredictionBatchResult
from app.domains.simulation.contour_2d_generator import (
    Contour2DGenerator,
    _marching_squares_segments,
)


class _SklearnAdapter:
    """Minimal BaseModelAdapter-shaped stub around a fitted sklearn estimator."""

    def __init__(self, model):
        self._model = model

    def predict_batch(self, model_obj, X_float32):
        preds = np.asarray(model_obj.predict(np.asarray(X_float32, dtype=np.float32))).ravel()
        return PredictionBatchResult(predictions=[float(v) for v in preds.tolist()])


def test_marching_squares_half_plane_single_line():
    xs = np.array([-5.0, 0.0, 5.0])
    ys = np.array([-5.0, 0.0, 5.0])
    field = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    segs = _marching_squares_segments(xs, ys, field, 0.5)
    assert len(segs) == 2  # one crossing per row-band
    for x1, y1, x2, y2 in segs:
        assert -5.0 <= x1 <= 5.0 and -5.0 <= x2 <= 5.0
        assert 2.4 < x1 < 2.6 and 2.4 < x2 < 2.6  # interpolated mid-edge


def test_marching_squares_constant_field_has_no_segments():
    xs = np.linspace(-5, 5, 8)
    ys = np.linspace(-5, 5, 8)
    field = np.full((8, 8), 0.9)
    assert _marching_squares_segments(xs, ys, field, 0.5) == []


def test_marching_squares_ambiguous_saddle_yields_two_segments():
    xs = np.array([-5.0, 5.0])
    ys = np.array([-5.0, 5.0])
    field = np.array([[1.0, 0.0], [0.0, 1.0]])
    segs = _marching_squares_segments(xs, ys, field, 0.5)
    assert len(segs) == 2


def test_binary_classification_contour_contract():
    X, y = make_blobs(n_samples=300, centers=2, n_features=3, random_state=7)
    X = X.astype(np.float32)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    adapter = _SklearnAdapter(clf)
    res = Contour2DGenerator.generate_contour(
        adapter=adapter,
        model_obj=clf,
        final_est=clf,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=60,
    )
    assert res.prediction_task == "classification"
    assert len(res.heatmap_quantized) == 60 * 60
    assert all(0 <= v <= 255 for v in res.heatmap_quantized)
    kinds = {c.kind for c in res.contours}
    assert "decision" in kinds
    decision = [c for c in res.contours if c.kind == "decision" and c.threshold == 0.5]
    assert decision and len(decision[0].line_segments) > 0
    for seg in decision[0].line_segments:
        assert len(seg) == 4
        assert all(-5.0 - 1e-6 <= v <= 5.0 + 1e-6 for v in seg)
    assert res.x_range_raw[0] < res.x_range_raw[1]


def test_regression_contour_has_elevation_and_response_curve():
    X, y = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=11)
    X = X.astype(np.float32)
    reg = LinearRegression().fit(X, y)
    adapter = _SklearnAdapter(reg)
    res = Contour2DGenerator.generate_contour(
        adapter=adapter,
        model_obj=reg,
        final_est=reg,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=40,
    )
    assert res.prediction_task == "regression"
    assert len(res.heatmap_quantized) == 40 * 40
    assert any(c.kind == "elevation" for c in res.contours)
    assert res.response_curve_1d is not None
    assert len(res.response_curve_1d.x_values_raw) == 120
    assert len(res.response_curve_1d.y_pred) == 120

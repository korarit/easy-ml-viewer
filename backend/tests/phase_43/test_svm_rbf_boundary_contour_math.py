"""Phase 43 — SVM RBF boundary / margin contour math + score-field parity."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC, SVR, LinearSVC

from app.core.math_3d import WorldCoordinateScaler
from app.domains.model_engine.value_objects import PredictionBatchResult
from app.domains.simulation.contour_2d_generator import Contour2DGenerator


class _Adapter:
    def __init__(self, model):
        self._model = model

    def predict_batch(self, model_obj, X_float32):
        preds = np.asarray(model_obj.predict(np.asarray(X_float32, dtype=np.float32))).ravel()
        return PredictionBatchResult(predictions=[float(v) for v in preds.tolist()])


def _world_to_raw(w: float, lo: float, hi: float) -> float:
    return float(lo + ((w + 5.0) / 10.0) * (hi - lo))


def _contour(model, X, res=60, **kw):
    return Contour2DGenerator.generate_contour(
        adapter=_Adapter(model),
        model_obj=model,
        final_est=model,
        X=X,
        x_feature_idx=0,
        y_feature_idx=1,
        x_feature_name="f0",
        y_feature_name="f1",
        grid_resolution=res,
        **kw,
    )


def test_rbf_decision_and_margin_levels_present(moons_rbf) -> None:
    X, y = moons_rbf
    model = SVC(kernel="rbf", C=1.0).fit(X, y)
    out = _contour(model, X)
    assert out.prediction_task == "classification"
    kinds = {(c.kind, c.threshold) for c in out.contours}
    assert ("decision", 0.0) in kinds
    assert ("margin", 1.0) in kinds and ("margin", -1.0) in kinds
    for c in out.contours:
        assert c.kind in ("decision", "margin", "confidence", "elevation")
        for seg in c.line_segments:
            assert all(-5.0 - 1e-6 <= v <= 5.0 + 1e-6 for v in seg)


def test_rbf_boundary_score_parity(moons_rbf) -> None:
    """Sampled f = 0 midpoints must sit near the true kernel zero crossing."""
    X, y = moons_rbf
    model = SVC(kernel="rbf", C=1.0).fit(X, y)
    out = _contour(model, X)
    decision = next(c for c in out.contours if c.kind == "decision")
    assert len(decision.line_segments) > 0
    x_lo, x_hi = out.x_range_raw
    y_lo, y_hi = out.y_range_raw
    mags = []
    for seg in decision.line_segments[:40]:
        mx_w, my_w = (seg[0] + seg[2]) / 2.0, (seg[1] + seg[3]) / 2.0
        raw = np.array(
            [[_world_to_raw(mx_w, x_lo, x_hi), _world_to_raw(my_w, y_lo, y_hi)]], dtype=np.float32
        )
        mags.append(abs(float(model.decision_function(raw)[0])))
    # Marching-squares interpolation error stays well below the margin scale.
    assert float(np.median(mags)) < 0.25
    margins = [c for c in out.contours if c.kind == "margin"]
    assert margins
    m = margins[0]
    seg = m.line_segments[0]
    mx_w, my_w = (seg[0] + seg[2]) / 2.0, (seg[1] + seg[3]) / 2.0
    raw = np.array(
        [[_world_to_raw(mx_w, x_lo, x_hi), _world_to_raw(my_w, y_lo, y_hi)]], dtype=np.float32
    )
    assert 0.5 <= abs(float(model.decision_function(raw)[0])) <= 1.5


def test_high_gamma_islands_detected() -> None:
    """E43.3: severe RBF overfitting fragments into islands — all extracted."""
    X, y = make_blobs(n_samples=120, centers=2, n_features=2, random_state=5)
    X = X.astype(np.float32)
    model = SVC(kernel="rbf", C=10.0, gamma=50.0).fit(X, y)
    out = _contour(model, X)
    decision = [c for c in out.contours if c.kind == "decision"]
    assert decision and len(decision[0].line_segments) >= 2  # fragmented, not one loop


def test_high_degree_poly_kernel_stays_bounded() -> None:
    """E43.6: poly d=5 completes with finite heat and world-clamped vectors."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 2)).astype(np.float32)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.5).astype(np.int32)
    model = SVC(kernel="poly", degree=5, C=1.0).fit(X, y)
    out = _contour(model, X)
    assert np.all(np.isfinite(np.asarray(out.heatmap_quantized)))
    for c in out.contours:
        for seg in c.line_segments:
            assert all(-5.0 - 1e-6 <= v <= 5.0 + 1e-6 for v in seg)


def test_svr_regression_has_no_margins() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 2)).astype(np.float32)
    y = X[:, 0] * 3.0 - X[:, 1]
    model = SVR(kernel="rbf", C=1.0).fit(X, y)
    out = _contour(model, X)
    assert out.prediction_task == "regression"
    assert not any(c.kind == "margin" for c in out.contours)
    assert any(c.kind == "elevation" for c in out.contours)


def test_linearsvc_score_path_margins() -> None:
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    model = LinearSVC(max_iter=5000).fit(X, y)
    out = _contour(model, X)
    assert out.prediction_task == "classification"
    assert any(c.kind == "decision" for c in out.contours)
    assert any(c.kind == "margin" for c in out.contours)
    # LinearSVC keeps no support vectors — static payload still reports why.
    assert out.svm_2d is not None and out.svm_2d["is_linear_no_sv"] is True


def test_rbf_contour_embeds_svm_halos(moons_rbf) -> None:
    X, y = moons_rbf
    model = SVC(kernel="rbf", C=1.0).fit(X, y)
    out = _contour(model, X, res=40)
    assert out.svm_2d is not None
    assert out.svm_2d["kernel"] == "rbf"
    assert out.svm_2d["n_support_total"] == len(model.support_vectors_)
    assert out.knn_2d is None
    for h in out.svm_2d["support_vectors_2d"]:
        assert -5.5 - 1e-6 <= h["x_world"] <= 5.5 + 1e-6
        assert -5.5 - 1e-6 <= h["y_world"] <= 5.5 + 1e-6
        assert 0.0 <= h["halo_scale"] <= 1.0

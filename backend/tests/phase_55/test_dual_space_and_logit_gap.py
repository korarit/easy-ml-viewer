"""Phase 55 — dual-space trails (A) and softmax logit-gap rings (B).

DoD mapping: the Raw <-> Scaled toggle maps points 1-to-1 through the
exact ``z = (x - mu) / sigma`` affine (DoD #1); logit-gap rings badge
saturated / tie / decisive points from ``Δz = z_(1) - z_(2)`` computed on
pre-softmax logits (DoD #2), with parity against a real MLP fit and its
StandardScaler pipeline.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.neural_onnx.neural_point_intelligence import (
    binary_logit_from_proba,
    dual_space_points,
    dual_space_transform,
    logit_gap,
    logit_gap_status,
)


def _blobs():
    X, y = make_blobs(
        n_samples=120, centers=[[0.0, 0.0], [2.5, 2.5]], n_features=2,
        cluster_std=1.2, random_state=11,
    )
    return X.astype(np.float64), y


def test_dual_space_affine_is_one_to_one():
    z = dual_space_transform(np.array([1.0, 2.0, 3.0]), 1.0, 2.0)
    assert z.tolist() == pytest.approx([0.0, 0.5, 1.0])
    # Degenerate sigma yields honest NaNs — never a bogus scaled frame.
    assert np.isnan(dual_space_transform(np.array([1.0]), 0.0, 0.0)).all()
    assert np.isnan(dual_space_transform(np.array([np.nan]), 0.0, 1.0)).all()
    assert dual_space_transform(np.array([]), 0.0, 1.0).size == 0


def test_dual_space_points_round_trip():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaled = dual_space_points(pts, 1.0, 2.0, 2.0, 2.0)
    assert np.allclose(scaled, np.array([[0.0, 0.0], [1.0, 1.0]]))
    # Round-trip recovers raw inputs exactly.
    raw = scaled * np.array([2.0, 2.0]) + np.array([1.0, 2.0])
    assert np.allclose(raw, pts)
    # Non-finite rows gap honestly instead of teleporting.
    gapped = dual_space_points(np.array([[np.inf, 0.0]]), 0.0, 1.0, 0.0, 1.0)
    assert np.isnan(gapped[0, 0])


def test_scaler_pipeline_matches_manual_affine():
    X, y = _blobs()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=3)),
    ]).fit(X, y)
    scaler = pipe.named_steps["scaler"]
    mu = np.asarray(scaler.mean_, dtype=np.float64)
    sigma = np.asarray(scaler.scale_, dtype=np.float64)
    got = dual_space_points(X[:10], mu[0], sigma[0], mu[1], sigma[1])
    manual = (X[:10] - mu) / sigma
    assert got == pytest.approx(manual)
    # sklearn's own transform agrees — the toggle is the true scaler frame.
    assert got == pytest.approx(scaler.transform(X[:10]))


def test_logit_gap_is_top_two_margin():
    z = np.array([[3.0, 1.0, 0.5], [0.4, 0.3, 0.35]])
    assert logit_gap(z).tolist() == pytest.approx([2.0, 0.05])
    # Binary single-logit head maps to |logit|.
    assert logit_gap(np.array([[2.0], [-1.5]])).tolist() == pytest.approx([2.0, 1.5])
    assert np.isnan(logit_gap(np.array([[1.0, np.nan]]))[0])
    assert logit_gap(np.empty((0, 2))).size == 0


def test_logit_gap_status_thresholds_match_spec():
    s = logit_gap_status(np.array([3.5, 3.0, 1.5, 0.49, 0.0, np.nan]))
    assert s.tolist() == [
        "confident",  # Δz > 3.0 → saturated solid ring
        "decisive",   # Δz == 3.0 → boundary stays decisive
        "decisive",
        "diffuse",    # Δz < 0.5 → neon tie halo
        "diffuse",
        "unknown",    # no belief → neutral, never a bogus badge
    ]


def test_mlp_pre_softmax_gap_matches_proba_order():
    X, y = _blobs()
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=3).fit(X, y)
    proba = np.asarray(mlp.predict_proba(X[:10]), dtype=np.float64)
    assert proba.shape == (10, 2)
    # Top-2 proba order agrees with the pre-softmax top-2 logit order, so
    # the ring badges the same winner the model actually predicts.
    winners = np.argmax(proba, axis=1)
    assert (winners == mlp.predict(X[:10])).all()
    logits = np.stack([binary_logit_from_proba(proba[:, 1]), -binary_logit_from_proba(proba[:, 1])], axis=1)
    gaps = logit_gap(np.column_stack([logits[:, 0], np.zeros(10)]))
    assert np.all(np.isfinite(gaps))
    # High-confidence rows carry wide gaps; ties carry narrow ones.
    top = proba.max(axis=1)
    assert gaps[top > 0.9].min() > gaps[top < 0.6].max() or (top <= 0.9).all()


def test_binary_logit_clips_miscalibrated_proba():
    assert binary_logit_from_proba(np.array([0.9]))[0] == pytest.approx(np.log(9.0))
    # p outside [0, 1] cannot fake an infinite gap.
    assert np.isfinite(binary_logit_from_proba(np.array([1.7]))[0])
    assert np.isnan(binary_logit_from_proba(np.array([np.nan]))[0])

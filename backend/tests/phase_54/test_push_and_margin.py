"""Phase 54 — staged SHAP push trails (A) and functional-margin rings (B).

DoD mapping: the comet trail replays cumulative TreeSHAP pushes stage by
stage (DoD #1); margin rings badge safe / critical / negative points from
``M = y·F(x)`` (DoD #2), with parity against real sklearn GBM staged
predictions and XGBoost ``predcontrib`` additivity.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingClassifier

from app.domains.model_engine.adapters.boosting.boosting_point_intelligence import (
    cumulative_push,
    functional_margin,
    margin_status,
    push_step_magnitudes,
    to_signed,
)


def _blobs():
    X, y = make_blobs(
        n_samples=120, centers=[[0.0, 0.0], [2.5, 2.5]], n_features=2,
        cluster_std=1.2, random_state=11,
    )
    return X.astype(np.float64), y


def test_cumulative_push_walks_stage_by_stage():
    stages = np.array([[1.0, 0.0], [0.0, 2.0], [-0.5, 0.5]])
    path = cumulative_push(stages)
    assert path.shape == (4, 2)  # origin + one vertex per stage
    assert path[0].tolist() == [0.0, 0.0]
    assert path == pytest.approx(np.vstack([np.zeros((1, 2)), np.cumsum(stages, axis=0)]))
    assert cumulative_push(np.empty((0, 2))).tolist() == [[0.0, 0.0]]
    # Non-finite stages step zero — the trail stays drawable, never gapped.
    gapped = cumulative_push(np.array([[1.0, np.nan], [0.0, 1.0]]))
    assert gapped[1].tolist() == pytest.approx([0.0, 0.0])
    assert gapped[2].tolist() == pytest.approx([0.0, 1.0])


def test_push_magnitudes_flag_the_hardest_stage():
    mags = push_step_magnitudes(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert mags.tolist() == pytest.approx([5.0, 0.0])
    assert np.isnan(push_step_magnitudes(np.array([[np.nan, 0.0]]))[0])
    assert push_step_magnitudes(np.empty((0, 2))).size == 0


def test_gbm_staged_predictions_converge_to_final():
    X, y = _blobs()
    gbm = GradientBoostingClassifier(n_estimators=20, random_state=3).fit(X, y)
    staged = list(gbm.staged_predict(X[:10]))
    assert len(staged) == 20
    assert (np.asarray(staged[-1]) == gbm.predict(X[:10])).all()
    # Early stages already disagree somewhere — the trail has somewhere to go.
    assert any((np.asarray(s) != gbm.predict(X[:10])).any() for s in staged[:-1])


def test_xgb_predcontrib_sums_to_margin():
    xgb = pytest.importorskip("xgboost")
    X, y = _blobs()
    clf = xgb.XGBClassifier(
        n_estimators=15, max_depth=2, learning_rate=0.3,
        reg_lambda=1.0, random_state=3, n_jobs=1,
    ).fit(X, y)
    booster = clf.get_booster()
    dm = xgb.DMatrix(X[:10])
    contribs = np.asarray(booster.predict(dm, pred_contribs=True), dtype=np.float64)
    margin = np.asarray(booster.predict(dm, output_margin=True), dtype=np.float64)
    assert contribs.shape == (10, 3)  # 2 features + bias
    assert contribs.sum(axis=1) == pytest.approx(margin, rel=1e-5, abs=1e-6)


def test_functional_margin_is_y_times_score():
    m = functional_margin(np.array([1.0, -1.0, 1.0]), np.array([2.0, 2.0, -0.5]))
    assert m.tolist() == pytest.approx([2.0, -2.0, -0.5])
    assert np.isnan(functional_margin(np.array([np.nan]), np.array([1.0]))[0])
    assert np.isnan(functional_margin(np.array([0.0]), np.array([1.0]))[0])  # y=0 is not a side
    assert functional_margin(np.array([]), np.array([])).size == 0


def test_margin_status_thresholds_match_spec():
    s = margin_status(np.array([2.1, 1.5, 0.4, 0.0, -0.01, -3.0, np.nan]))
    assert s.tolist() == [
        "safe",       # M > 1.5 → solid class-color ring
        "critical",   # M == 1.5 → still pulsing orange
        "critical",
        "critical",   # on the boundary — near the flip
        "negative",   # M < 0 → red cross, next round's target
        "negative",
        "unknown",    # no belief → neutral, never a bogus badge
    ]


def test_signed_labels_split_binary_classes():
    assert to_signed(np.array([0, 1, 1, 0]), 1).tolist() == [-1.0, 1.0, 1.0, -1.0]
    assert to_signed(np.array(["a", "b"]), "b").tolist() == [-1.0, 1.0]
    assert np.isnan(to_signed(np.array([np.nan]), 1)[0])

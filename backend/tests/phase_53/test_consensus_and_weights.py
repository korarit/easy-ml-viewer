"""Phase 53 — forest voting consensus (B) and AdaBoost weight rings (B).

Covers mechanism B's forest half (vote shares, sigma, ring style with the
60% fuzzy threshold) plus the Manhattan trail kernel (A), with parity
against real RandomForest / ExtraTrees ``estimators_`` votes on 2D data.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from app.domains.model_engine.adapters.trees.trees_point_intelligence import (
    adaboost_ring_scale,
    consensus_ring_style,
    forest_consensus,
    leaf_transition,
    manhattan_elbow,
    manhattan_length,
)


def _blobs():
    X, y = make_blobs(n_samples=120, centers=2, n_features=2, random_state=7)
    return X.astype(np.float64), y


def _tree_votes(forest, X):
    return np.column_stack([t.predict(X) for t in forest.estimators_])


def test_manhattan_elbow_routes_orthogonally():
    elbow = manhattan_elbow(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert elbow.tolist() == [3.0, 0.0]  # x-first: horizontal then vertical
    elbow_v = manhattan_elbow(
        np.array([0.0, 0.0]), np.array([3.0, 4.0]), x_first=False
    )
    assert elbow_v.tolist() == [0.0, 4.0]
    assert np.all(np.isnan(manhattan_elbow(np.array([np.nan, 0.0]), np.array([1.0, 1.0]))))
    assert manhattan_length(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(7.0)
    assert np.isnan(manhattan_length(np.array([np.nan, 0.0]), np.array([1.0, 1.0])))


def test_leaf_transition_pulses_only_on_branch_switch():
    assert leaf_transition(3, 5) is True
    assert leaf_transition(3, 3) is False
    assert leaf_transition(None, 3) is False
    assert leaf_transition(float("nan"), 3) is False


def test_forest_consensus_matches_estimators_votes():
    X, y = _blobs()
    forest = RandomForestClassifier(n_estimators=25, random_state=3).fit(X, y)
    votes = _tree_votes(forest, X[:30])
    max_share, sigma = forest_consensus(votes)
    assert max_share.shape == (30,)
    # Manual recomputation from the same estimator votes.
    for i in range(30):
        labels, counts = np.unique(votes[i], return_counts=True)
        assert max_share[i] == pytest.approx(counts.max() / votes.shape[1])
    assert sigma == pytest.approx(1.0 - max_share)
    assert bool(np.all((max_share >= 0.5) & (max_share <= 1.0)))


def test_extra_trees_consensus_agrees_with_majority_predict():
    X, y = _blobs()
    forest = ExtraTreesClassifier(n_estimators=25, random_state=3).fit(X, y)
    votes = _tree_votes(forest, X[:30])
    max_share, _ = forest_consensus(votes)
    majority = np.array(
        [np.unique(row, return_counts=True)[0][int(np.argmax(np.unique(row, return_counts=True)[1]))]
         for row in votes]
    )
    assert (majority == forest.predict(X[:30])).all()
    assert bool(np.all(max_share >= 0.5))


def test_consensus_ring_turns_fuzzy_below_sixty_percent():
    fuzzy, thick = consensus_ring_style(np.array([1.0, 0.6, 0.59, 0.52]))
    assert fuzzy.tolist() == [False, False, True, True]  # DoD #1: <60% → double-dashed
    assert thick[0] == pytest.approx(1.0)  # unanimous → thin solid
    assert thick[3] == pytest.approx(1.0 + (1.0 - 0.52))  # coin flip → ~2x
    assert thick[2] > thick[1]  # thickness grows as agreement falls
    fuzzy_nan, thick_nan = consensus_ring_style(np.array([np.nan]))
    assert fuzzy_nan.tolist() == [False]  # no belief → neutral, never fuzzy
    assert thick_nan[0] == pytest.approx(1.0)


def test_adaboost_ring_scales_with_sqrt_weight_ratio():
    w = np.array([0.25, 1.0, 4.0])
    scales = adaboost_ring_scale(w, base_thickness=0.04)
    mean = w.mean()
    assert scales == pytest.approx(0.04 * np.sqrt(w / mean))
    assert scales[2] > scales[1] > scales[0]  # DoD #2: heavy points stand out
    # Degenerate inputs never spike.
    assert adaboost_ring_scale(np.array([np.nan, np.inf, -1.0])) == pytest.approx(
        np.full((3,), 1.0)
    )
    assert adaboost_ring_scale(np.zeros(3)) == pytest.approx(np.full((3,), 1.0))
    assert adaboost_ring_scale(np.array([])).size == 0


def test_consensus_handles_empty_and_ragged_votes():
    ms, sg = forest_consensus(np.empty((0, 5), dtype=int))
    assert ms.size == 0 and sg.size == 0
    ms, _ = forest_consensus(np.array([[0, 0, 0]]))
    assert ms[0] == pytest.approx(1.0)  # unanimous single row

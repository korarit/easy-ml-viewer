"""Phase 53 — leaf minority spokes (C), stepwise residuals (C) and
leaf-bounded hexbin purity (D).

DoD mapping: minority points in impure leaves grow leaf-centre spokes
(DoD #3); regressor residuals anchor to the leaf constant; hexbin purity
flags mixed-leaf blocks while everything stays vectorized for 60 FPS
(DoD #4).
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.domains.model_engine.adapters.trees.trees_point_intelligence import (
    hex_purity,
    is_leaf_minority,
    is_mixed_leaf,
    leaf_center_spoke,
    stepwise_residual,
)


def test_minority_detection_matches_leaf_majority():
    X, y = make_blobs(
        n_samples=150,
        centers=[[0.0, 0.0], [1.5, 1.5]],
        n_features=2,
        cluster_std=2.0,
        random_state=9,
    )
    tree = DecisionTreeClassifier(max_depth=1, random_state=1).fit(X, y)
    leaf_ids = tree.apply(X)
    leaf_pred = np.asarray(
        [tree.tree_.value[leaf, 0].argmax() for leaf in leaf_ids], dtype=np.float64
    )
    minority = is_leaf_minority(y.astype(np.float64), leaf_pred)
    assert minority.shape == (150,)
    # Every flagged point truly disagrees with its leaf majority.
    assert bool(np.all(y[minority].astype(int) != leaf_pred[minority].astype(int)))
    # Pure-leaf residents never flag.
    assert bool(np.all(y[~minority].astype(int) == leaf_pred[~minority].astype(int)))
    assert bool(minority.any())  # depth-2 stump strands minorities
    # NaN truth is suspect → minority, never silently hidden.
    assert bool(is_leaf_minority(np.array([np.nan]), np.array([0.0]))[0])


def test_leaf_center_spoke_points_to_box_centre():
    v = leaf_center_spoke(np.array([1.0, 2.0]), np.array([4.0, 6.0]))
    assert v.tolist() == pytest.approx([3.0, 4.0])
    assert leaf_center_spoke(
        np.array([np.nan, 2.0]), np.array([4.0, 6.0])
    ).tolist() == [0.0, 0.0]
    assert leaf_center_spoke(np.array([1.0]), np.array([4.0, 6.0])).tolist() == [
        0.0,
        0.0,
    ]


def test_stepwise_residual_anchors_to_leaf_constant():
    X, y = make_regression(n_samples=100, n_features=2, noise=5.0, random_state=4)
    tree = DecisionTreeRegressor(max_depth=3, random_state=2).fit(X, y)
    leaf_ids = tree.apply(X)
    y_leaf = np.asarray(
        [tree.tree_.value[leaf, 0, 0] for leaf in leaf_ids], dtype=np.float64
    )
    resid = stepwise_residual(y.astype(np.float64), y_leaf)
    assert resid == pytest.approx(y - y_leaf)
    assert np.isnan(stepwise_residual(np.array([np.nan]), np.array([1.0]))[0])
    assert stepwise_residual(np.array([]), np.array([])).size == 0


def test_hex_purity_scores_leaf_blocks():
    purity = hex_purity(np.array([10.0, 5.0, 0.0]), np.array([10.0, 10.0, 10.0]))
    assert purity[0] == pytest.approx(1.0)  # pure leaf block → solid
    assert purity[1] == pytest.approx(0.5)  # mixed block → warn
    assert purity[2] == pytest.approx(0.0)
    bad = hex_purity(np.array([3.0, np.nan]), np.array([0.0, 5.0]))
    assert np.isnan(bad[0])  # empty bin → desert badge, never a score
    assert np.isnan(bad[1])


def test_mixed_leaf_flags_impure_boxes_only():
    assert is_mixed_leaf(np.array([0.0, 0.32, 0.5])).tolist() == [False, True, True]
    assert is_mixed_leaf(np.array([np.nan])).tolist() == [False]

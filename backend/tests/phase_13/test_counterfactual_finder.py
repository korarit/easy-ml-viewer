"""Phase 13 — CounterfactualFinderService unit tests (spec §4.1 + §7 edges)."""

from __future__ import annotations

import numpy as np

from app.domains.whatif.services import (
    CounterfactualFinderService,
    detect_feature_kinds,
    subsample_reference_pool,
)


def _linear_threshold_predict(threshold: float = 0.0):
    def _fn(X: np.ndarray) -> np.ndarray:
        return (X[:, 0] + X[:, 1] > threshold).astype(np.float32)

    return _fn


def test_classification_flip_finds_minimum_change():
    """Nearest opposite-class row + bisection flips the decision."""
    rng = np.random.default_rng(7)
    X = (rng.random((400, 2), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    predict_fn = _linear_threshold_predict(0.0)
    y = predict_fn(X)
    base = np.array([-1.5, -1.5], dtype=np.float32)

    found, recipe, message = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=predict_fn,
        X_dataset=X,
        y_predictions=y,
        base_row=base,
        locked_feature_indices=[],
        target_class_or_threshold=1.0,
        is_classification=True,
    )
    assert found is True, message
    assert float(predict_fn(recipe.reshape(1, -1))[0]) == 1.0
    # Bisection pulls the recipe close to the x0+x1=0 boundary.
    assert abs(float(recipe[0] + recipe[1])) < 0.6


def test_locked_features_are_never_touched():
    """Immutable locks survive projection (spec §2 UX) and bisection."""
    rng = np.random.default_rng(11)
    X = (rng.random((300, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    def _fn(batch: np.ndarray) -> np.ndarray:
        return (batch[:, 0] + batch[:, 1] + batch[:, 2] > 0.0).astype(np.float32)

    y = _fn(X)
    base = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    found, recipe, _ = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=_fn,
        X_dataset=X,
        y_predictions=y,
        base_row=base,
        locked_feature_indices=[0],
        target_class_or_threshold=1.0,
        is_classification=True,
    )
    assert found is True
    assert float(recipe[0]) == float(base[0])
    assert float(_fn(recipe.reshape(1, -1))[0]) == 1.0


def test_all_locked_returns_thai_unlock_message():
    """Edge #3: every column locked -> guided failure, base row untouched."""
    rng = np.random.default_rng(3)
    X = (rng.random((100, 2), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    predict_fn = _linear_threshold_predict(0.0)
    base = np.array([-0.5, -0.5], dtype=np.float32)

    found, recipe, message = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=predict_fn,
        X_dataset=X,
        y_predictions=predict_fn(X),
        base_row=base,
        locked_feature_indices=[0, 1],
        target_class_or_threshold=1.0,
        is_classification=True,
    )
    assert found is False
    assert "ปลดล็อก" in message
    assert np.allclose(recipe, base)


def test_single_class_pool_uses_synthetic_grid_fallback():
    """Edge #6: no opposite-class row in CSV -> 500-point synthetic search."""
    X = np.zeros((200, 2), dtype=np.float32)

    def _fn(batch: np.ndarray) -> np.ndarray:
        return ((batch[:, 0] > 0.5) & (batch[:, 1] > 0.5)).astype(np.float32)

    base = np.array([0.0, 0.0], dtype=np.float32)
    found, recipe, _ = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=_fn,
        X_dataset=X,
        y_predictions=_fn(X),
        base_row=base,
        locked_feature_indices=[],
        target_class_or_threshold=1.0,
        is_classification=True,
    )
    assert found is True
    assert float(recipe[0]) > 0.5 and float(recipe[1]) > 0.5


def test_integer_and_binary_columns_round_to_whole_numbers():
    """Edge #2: int/binary columns never return fractional recipes."""
    rng = np.random.default_rng(5)
    int_col = rng.integers(0, 10, size=(300, 1)).astype(np.float32)
    bin_col = rng.integers(0, 2, size=(300, 1)).astype(np.float32)
    cont_col = (rng.random((300, 1), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    X = np.hstack([int_col, bin_col, cont_col])

    def _fn(batch: np.ndarray) -> np.ndarray:
        return (batch[:, 0] + 2.0 * batch[:, 1] + batch[:, 2] > 5.0).astype(np.float32)

    base = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    found, recipe, _ = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=_fn,
        X_dataset=X,
        y_predictions=_fn(X),
        base_row=base,
        locked_feature_indices=[],
        target_class_or_threshold=1.0,
        is_classification=True,
    )
    assert found is True
    assert float(recipe[0]) == round(float(recipe[0]))
    assert float(recipe[1]) in (0.0, 1.0)


def test_subsampled_pool_stays_bounded_and_finds_flip():
    """§5.1: >20k rows still searches a bounded pool and flips."""
    rng = np.random.default_rng(9)
    X = (rng.random((2_500, 2), dtype=np.float32) * 6.0 - 3.0).astype(np.float32)
    assert X.nbytes < 4 * 1024 * 1024  # reference pool budget sanity
    predict_fn = _linear_threshold_predict(1.0)
    base = np.array([-2.0, -2.0], dtype=np.float32)

    found, recipe, _ = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=predict_fn,
        X_dataset=X,
        y_predictions=predict_fn(X),
        base_row=base,
        locked_feature_indices=[],
        target_class_or_threshold=1.0,
        is_classification=True,
    )
    assert found is True
    assert float(predict_fn(recipe.reshape(1, -1))[0]) == 1.0


def test_subsample_reference_pool_caps_rows():
    """Stride subsample never exceeds the 5,000-row cap."""
    rng = np.random.default_rng(13)
    X = rng.random((21_000, 3), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    pool_X, pool_y = subsample_reference_pool(X, y)
    assert len(pool_X) <= 5_000
    assert len(pool_X) == len(pool_y)
    assert set(np.unique(pool_y).tolist()) <= {0.0, 1.0}


def test_regression_goal_seek_reaches_threshold():
    """Regression mode: recipe prediction satisfies the goal threshold."""

    def _fn(batch: np.ndarray) -> np.ndarray:
        return (2.0 * batch[:, 0] - batch[:, 1] + 0.5).astype(np.float32)

    rng = np.random.default_rng(17)
    X = (rng.random((300, 2), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y = _fn(X)
    base = np.array([-1.0, 1.0], dtype=np.float32)
    base_pred = float(_fn(base.reshape(1, -1))[0])

    found, recipe, _ = CounterfactualFinderService.find_minimum_change_counterfactual(
        predict_fn=_fn,
        X_dataset=X,
        y_predictions=y,
        base_row=base,
        locked_feature_indices=[],
        target_class_or_threshold=base_pred + 3.0,
        is_classification=False,
    )
    assert found is True
    assert float(_fn(recipe.reshape(1, -1))[0]) >= base_pred + 3.0 - 1e-4


def test_detect_feature_kinds_binary_integer_float():
    X = np.column_stack([
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([3.0, 7.0, 12.0, 0.0], dtype=np.float32),
        np.array([0.5, 1.25, 3.75, 9.1], dtype=np.float32),
    ])
    assert detect_feature_kinds(X) == ["binary", "integer", "float"]

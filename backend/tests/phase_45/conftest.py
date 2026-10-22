"""Phase 45 — shared fixtures: blob datasets + fitted tree ensembles."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.core.math_3d import WorldCoordinateScaler


@pytest.fixture()
def blobs_2d():
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=7)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def blobs_4d():
    X, y = make_blobs(n_samples=400, centers=2, n_features=4, random_state=11)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def hard_blobs():
    """Heavily overlapping classes — boosters keep learning for many rounds."""
    X, y = make_blobs(n_samples=300, centers=2, n_features=2,
                      random_state=7, cluster_std=3.0)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def fitted_tree(blobs_2d):
    X, y = blobs_2d
    return DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)


@pytest.fixture()
def fitted_forest(blobs_2d):
    X, y = blobs_2d
    return RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42).fit(X, y)


@pytest.fixture()
def fitted_extratrees(blobs_2d):
    X, y = blobs_2d
    return ExtraTreesClassifier(n_estimators=10, max_depth=4, random_state=42).fit(X, y)


@pytest.fixture()
def fitted_adaboost(hard_blobs):
    X, y = hard_blobs
    return AdaBoostClassifier(n_estimators=10, random_state=42).fit(X, y)


@pytest.fixture()
def fitted_regressor(blobs_2d):
    X, y = blobs_2d
    return DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y.astype(np.float64))


def make_ctx(X, x_idx=0, y_idx=1, xn="f0", yn="f1"):
    xs = WorldCoordinateScaler.build_axis_scale(xn, X[:, x_idx])
    ys = WorldCoordinateScaler.build_axis_scale(yn, X[:, y_idx])
    xr = (float(X[:, x_idx].min()), float(X[:, x_idx].max()))
    yr = (float(X[:, y_idx].min()), float(X[:, y_idx].max()))
    return xs, ys, xr, yr

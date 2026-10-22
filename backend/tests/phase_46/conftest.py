"""Phase 46 — shared fixtures: blob datasets + fitted boosting heavyweights."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from app.core.math_3d import WorldCoordinateScaler


@pytest.fixture()
def blobs_2d():
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=7)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def blobs_4d():
    X, y = make_blobs(n_samples=200, centers=2, n_features=4, random_state=11)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def fitted_gbm(blobs_2d):
    X, y = blobs_2d
    return GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42).fit(X, y)


@pytest.fixture()
def fitted_hist(blobs_2d):
    X, y = blobs_2d
    return HistGradientBoostingClassifier(max_iter=10, max_depth=3, random_state=42).fit(X, y)


@pytest.fixture()
def fitted_xgb(blobs_2d):
    xgb = pytest.importorskip("xgboost")
    X, y = blobs_2d
    m = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42,
                          nthread=2, verbosity=0)
    return m.fit(X, y)


@pytest.fixture()
def fitted_lgbm(blobs_2d):
    lgbm = pytest.importorskip("lightgbm")
    X, y = blobs_2d
    m = lgbm.LGBMClassifier(n_estimators=10, max_depth=3, random_state=42,
                            num_threads=2, verbose=-1)
    return m.fit(X, y)


@pytest.fixture()
def fitted_catboost(blobs_2d):
    cb = pytest.importorskip("catboost")
    X, y = blobs_2d
    m = cb.CatBoostClassifier(depth=3, n_estimators=5, random_seed=42,
                              thread_count=2, verbose=False)
    return m.fit(X, y)


def make_ctx(X, x_idx=0, y_idx=1, xn="f0", yn="f1"):
    xs = WorldCoordinateScaler.build_axis_scale(xn, X[:, x_idx])
    ys = WorldCoordinateScaler.build_axis_scale(yn, X[:, y_idx])
    xr = (float(X[:, x_idx].min()), float(X[:, x_idx].max()))
    yr = (float(X[:, y_idx].min()), float(X[:, y_idx].max()))
    return xs, ys, xr, yr

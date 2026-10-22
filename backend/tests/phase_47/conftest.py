"""Phase 47 — shared fixtures: blobs + fitted MLP / pipelines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from app.core.math_3d import WorldCoordinateScaler


@pytest.fixture()
def blobs_2d():
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=7)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def fitted_mlp(blobs_2d):
    X, y = blobs_2d
    m = MLPClassifier(hidden_layer_sizes=(8,), activation="relu",
                      max_iter=800, random_state=42)
    return m.fit(X, y)


@pytest.fixture()
def fitted_standard_pipeline(blobs_2d):
    from sklearn.linear_model import LogisticRegression

    X, y = blobs_2d
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    return pipe.fit(X, y)


@pytest.fixture()
def fitted_minmax_pipeline(blobs_2d):
    from sklearn.linear_model import LogisticRegression

    X, y = blobs_2d
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    return pipe.fit(X, y)


def make_ctx(X, x_idx=0, y_idx=1, xn="f0", yn="f1"):
    xs = WorldCoordinateScaler.build_axis_scale(xn, X[:, x_idx])
    ys = WorldCoordinateScaler.build_axis_scale(yn, X[:, y_idx])
    xr = (float(X[:, x_idx].min()), float(X[:, x_idx].max()))
    yr = (float(X[:, y_idx].min()), float(X[:, y_idx].max()))
    return xs, ys, xr, yr

"""Phase 43 — shared fixtures for distance & kernel 2D geometry tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC


@pytest.fixture()
def spec_blobs_3d():
    """Exact arrays from the Phase-43 test blueprint (§2.2 prototype)."""
    X_train = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [3.0, 4.0, 1.0],
            [5.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    query = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    return X_train, y_train, query


@pytest.fixture()
def spec_knn3(spec_blobs_3d):
    X_train, y_train, _ = spec_blobs_3d
    return KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)


@pytest.fixture()
def spec_radius():
    X_train = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [3.0, 4.0, 1.0],
            [5.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    return RadiusNeighborsClassifier(radius=2.5).fit(X_train, y_train)


@pytest.fixture()
def spec_svc_linear():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    return SVC(kernel="linear", C=1.0).fit(X, y)


@pytest.fixture()
def moons_rbf():
    X, y = make_blobs(n_samples=240, centers=2, n_features=2, random_state=43)
    X = (X / np.abs(X).max() * 2.0).astype(np.float32)
    return X, y.astype(np.int32)

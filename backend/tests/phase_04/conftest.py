"""Phase 04 shared fixtures — sample OLS model + 3-feature dataset."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture()
def house_X() -> np.ndarray:
    rng = np.random.default_rng(7)
    n = 60
    area = rng.uniform(25.0, 250.0, size=n).astype(np.float32)
    bedrooms = rng.integers(1, 6, size=n).astype(np.float32)
    dist = rng.uniform(0.2, 18.0, size=n).astype(np.float32)
    return np.column_stack([area, bedrooms, dist]).astype(np.float32)


@pytest.fixture()
def house_y(house_X: np.ndarray) -> np.ndarray:
    # Noise-free ground truth: y = 150000 + 42500*x0 + 350000*x1 - 185000*x2
    return (
        150000.0 + 42500.0 * house_X[:, 0] + 350000.0 * house_X[:, 1] - 185000.0 * house_X[:, 2]
    ).astype(np.float32)


@pytest.fixture()
def house_model(house_X: np.ndarray, house_y: np.ndarray) -> LinearRegression:
    return LinearRegression().fit(house_X, house_y)


@pytest.fixture()
def feature_names() -> list[str]:
    return ["area_sqm", "bedrooms", "distance_bts_km"]

"""Phase 49 — shared fixtures: clustered world points + binary labels."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def clustered_world():
    rng = np.random.default_rng(49)
    a = rng.normal(loc=[-2.5, 1.5], scale=0.8, size=(150, 2))
    b = rng.normal(loc=[2.5, -1.5], scale=0.8, size=(150, 2))
    pts = np.vstack([a, b]).astype(np.float32)
    y_true = np.array([0] * 150 + [1] * 150, dtype=np.int64)
    y_pred = y_true.copy()
    # Inject a 20% error pocket inside cluster B.
    y_pred[200:230] = 0
    return pts, y_true, y_pred


@pytest.fixture()
def uniform_world():
    rng = np.random.default_rng(7)
    return rng.uniform(-5, 5, size=(300, 2)).astype(np.float32)

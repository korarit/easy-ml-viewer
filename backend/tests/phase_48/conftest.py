"""Phase 48 — shared fixtures: two blobs + fitted classifiers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons


@pytest.fixture()
def blobs_2d():
    X, y = make_blobs(n_samples=240, centers=2, n_features=2, random_state=7)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def moons_2d():
    X, y = make_moons(n_samples=240, noise=0.12, random_state=11)
    return X.astype(np.float64), y.astype(np.int64)


@pytest.fixture()
def fitted_rf(blobs_2d):
    from sklearn.ensemble import RandomForestClassifier

    X, y = blobs_2d
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)


@pytest.fixture()
def fitted_svc(blobs_2d):
    from sklearn.svm import SVC

    X, y = blobs_2d
    return SVC(kernel="rbf", probability=True, random_state=42).fit(X, y)


def make_contour_dict(res: int = 24, shift: float = 0.0, task: str = "classification"):
    """Small synthetic contour DTO (same shape as Contour2DResponse JSON)."""
    rng = np.random.default_rng(48)
    field = rng.uniform(0.0, 1.0, size=(res, res)).astype(np.float64)
    field = np.clip(field + shift, 0.0, 1.0)
    h_min, h_max = float(field.min()), float(field.max())
    span = max(h_max - h_min, 1e-9)
    quantized = (np.rint((field - h_min) / span * 255.0).astype(np.uint8)).ravel().tolist()
    return {
        "x_feature": "income",
        "y_feature": "debt_ratio",
        "grid_resolution": res,
        "prediction_task": task,
        "heatmap_min": h_min,
        "heatmap_max": h_max,
        "heatmap_quantized": quantized,
        "contours": [
            {"threshold": 0.5, "class_index": 1, "kind": "decision",
             "line_segments": [[-2.5, 1.0, -1.0, 1.5], [-1.0, 1.5, 0.5, 2.0]]},
        ],
    }

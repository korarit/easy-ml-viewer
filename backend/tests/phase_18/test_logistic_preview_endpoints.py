"""Phase 18 — trace-preview / surface-preview REST tests for logistic models."""

from __future__ import annotations

import io

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from app.main import create_app

client = TestClient(create_app())


def _upload(model: LogisticRegression, name: str) -> str:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, buf, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == "logistic_regression"
    return body["model_id"]


def _binary_model() -> tuple[LogisticRegression, np.ndarray]:
    X, y = make_classification(
        n_samples=80, n_features=4, n_informative=3, n_redundant=0, random_state=11
    )
    X = X.astype(np.float32)
    return LogisticRegression(max_iter=1000).fit(X, y), X


def test_trace_preview_log_odds_waterfall_identity() -> None:
    model, X = _binary_model()
    model_id = _upload(model, "logreg_binary.joblib")
    rows = X[:2].tolist()
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": rows, "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    scores = np.asarray(model.decision_function(X[:2])).reshape(-1)
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {f"feature_{j}" for j in range(4)}
        assert np.isclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            scores[i],
            atol=1e-4,
        )
        assert tr["extra_mechanics"]["is_binary"] is True
        assert tr["extra_mechanics"]["decision_threshold"] == 0.5


def test_surface_preview_sigmoid_probability_range() -> None:
    model, _ = _binary_model()
    model_id = _upload(model, "logreg_surface.joblib")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -3.0,
            "x_max": 3.0,
            "y_min": -3.0,
            "y_max": 3.0,
            "resolution": 50,
            "fixed_feature_medians": {"2": 0.0, "3": 0.0},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "probability_sigmoid_surface"
    assert len(body["z_values"]) == 2500
    assert min(body["z_values"]) >= 0.0
    assert max(body["z_values"]) <= 1.0
    assert body["telemetry"]["grid_points"] == 2500

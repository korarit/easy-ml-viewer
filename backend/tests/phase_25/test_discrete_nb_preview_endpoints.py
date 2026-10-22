"""Phase 25 — trace-preview / surface-preview REST tests for discrete NB."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB

from app.main import create_app

client = TestClient(create_app())


def _upload(model, name: str) -> str:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, buf, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == "discrete_nb", body
    assert body["visual_3d_type"] == "discrete_nb_towers", body
    return body["model_id"]


def _counts() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(25)
    lam = np.array([[4.0, 3.0, 0.2, 0.3], [0.1, 0.2, 3.0, 2.5]])
    y = (rng.random(80) < 0.5).astype(np.int32)
    return np.stack([rng.poisson(lam[k]).astype(np.float32) for k in y]), y


def _binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(26)
    y = (rng.random(80) < 0.4).astype(np.int32)
    X = (rng.random((80, 4)) < np.where(y[:, None] == 1, 0.7, 0.2)).astype(np.float32)
    return X, y


def test_trace_preview_multinomial_waterfall() -> None:
    X, y = _counts()
    model = MultinomialNB(alpha=1.0).fit(X, y)
    model_id = _upload(model, "discrete_multi.joblib")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    joint = np.asarray(model.predict_joint_log_proba(X[:2]), dtype=np.float64)
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {f"feature_{j}" for j in range(4)}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        mech = tr["extra_mechanics"]
        assert mech["nb_variant"] == "multinomial"
        assert mech["tower_heights"] == pytest.approx(mech["probabilities"], rel=1e-9)


def test_trace_preview_bernoulli_presence_flags() -> None:
    X, y = _binary()
    model = BernoulliNB(binarize=0.5).fit(X, y)
    model_id = _upload(model, "discrete_bern.joblib")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    joint = np.asarray(model.predict_joint_log_proba(X[:2]), dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr["extra_mechanics"]["nb_variant"] == "bernoulli"
        assert len(tr["extra_mechanics"]["presence_flags"]) == 4


def test_trace_preview_categorical_target_switch() -> None:
    rng = np.random.default_rng(28)
    X = np.stack(
        [
            rng.integers(0, 3, size=120),
            rng.integers(0, 2, size=120),
            rng.integers(0, 4, size=120),
            rng.integers(0, 2, size=120),
        ],
        axis=1,
    ).astype(np.float32)
    y = ((X[:, 0] == 2).astype(np.int32) + (X[:, 2] >= 2).astype(np.int32)) % 3
    model = CategoricalNB().fit(X, y)
    model_id = _upload(model, "discrete_cat.joblib")
    joint = np.asarray(model.predict_joint_log_proba(X[:1]), dtype=np.float64)
    for k in range(3):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": X[:1].tolist(), "row_indices": [0], "target_idx": k},
        )
        assert res.status_code == 200, res.text
        tr = res.json()["traces"][0]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(joint[0, k]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k


def test_surface_preview_discrete_towers() -> None:
    X, y = _counts()
    model = MultinomialNB().fit(X, y)
    model_id = _upload(model, "discrete_surface.joblib")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -3.0,
            "x_max": 8.0,
            "y_min": -2.0,
            "y_max": 6.0,
            "resolution": 50,
            "fixed_feature_medians": {"2": 1.0, "3": 0.5},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "discrete_nb_towers"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    assert body["extra_overlays"]["clamped_non_negative"] is True
    assert body["extra_overlays"]["boundary_threshold"] == 0.5
    # Negative requested bounds never leak into a count-domain grid.
    assert body["x_grid"][0] >= 0.0 and body["y_grid"][0] >= 0.0

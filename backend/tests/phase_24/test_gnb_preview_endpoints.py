"""Phase 24 — trace-preview / surface-preview REST tests for GaussianNB."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from app.main import create_app

client = TestClient(create_app())


def _upload(model, name: str, expected_key: str) -> str:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, buf, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == expected_key, body
    return body["model_id"]


def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=80, n_features=4, n_informative=2, n_redundant=1, random_state=11
    )
    return StandardScaler().fit_transform(X).astype(np.float32), y


def _iris_like() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    centers = np.array(
        [[5.0, 3.4, 1.5, 0.2], [6.0, 2.8, 4.3, 1.3], [6.6, 3.0, 5.5, 2.0]],
        dtype=np.float64,
    )
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for k, c in enumerate(centers):
        xs.append(rng.normal(loc=c, scale=[0.35, 0.3, 0.35, 0.15], size=(30, 4)))
        ys.extend([k] * 30)
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int32)


def test_trace_preview_gnb_binary_waterfall() -> None:
    X, y = _binary_data()
    model = GaussianNB().fit(X, y)
    model_id = _upload(model, "gnb_binary.joblib", "gaussian_nb")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    joint = np.asarray(model._joint_log_likelihood(X[:2]), dtype=np.float64)
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {f"feature_{j}" for j in range(4)}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(joint[i, 1] - joint[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        mech = tr["extra_mechanics"]
        assert mech["is_binary"] is True
        assert mech["joint_log_probs"] == pytest.approx(joint[i].tolist(), rel=1e-6)
        assert len(mech["winning_class_z_scores"]) == 4


def test_trace_preview_gnb_multiclass_target_switch() -> None:
    X, y = _iris_like()
    model = GaussianNB().fit(X, y)
    model_id = _upload(model, "gnb_multi.joblib", "gaussian_nb")
    rows = X[:1].tolist()
    for k in range(3):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": rows, "row_indices": [0], "target_idx": k},
        )
        assert res.status_code == 200, res.text
        tr = res.json()["traces"][0]
        joint = np.asarray(model._joint_log_likelihood(X[:1]), dtype=np.float64)
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(joint[0, k]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k


def test_trace_preview_gnb_outlier_flag() -> None:
    X, y = _binary_data()
    model = GaussianNB().fit(X, y)
    model_id = _upload(model, "gnb_outlier.joblib", "gaussian_nb")
    far = np.full((1, 4), 100.0, dtype=np.float32).tolist()
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview", json={"rows": far, "row_indices": [0]}
    )
    assert res.status_code == 200, res.text
    assert res.json()["traces"][0]["extra_mechanics"]["is_low_density_outlier"] is True


def test_surface_preview_gnb_bell_hills() -> None:
    X, y = _iris_like()
    model = GaussianNB().fit(X, y)
    model_id = _upload(model, "gnb_surface.joblib", "gaussian_nb")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 2,
            "axis_y_feature_index": 3,
            "x_min": 0.5,
            "x_max": 7.0,
            "y_min": 0.0,
            "y_max": 2.6,
            "resolution": 50,
            "fixed_feature_medians": {"0": 5.8, "1": 3.0},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "gaussian_ellipsoid_domes"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    assert set(body["extra_overlays"]["bell_hills_per_class"]) == {"0", "1", "2"}
    assert body["extra_overlays"]["boundary_threshold"] == 0.5
    assert len(body["extra_overlays"]["peaks_2d"]) == 3
    assert len(body["extra_overlays"]["stds_2d"]) == 3

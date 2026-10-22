"""Phase 26 — trace-preview / surface-preview REST tests for LDA/QDA."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

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
    assert body["algorithm_key"] == "discriminant", body
    assert body["visual_3d_type"] == "discriminant_ellipsoids", body
    return body["model_id"]


def _blobs2() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(26)
    a = rng.normal(size=(60, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(60, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    return np.vstack([a, b]).astype(np.float32), np.array([0] * 60 + [1] * 60, dtype=np.int32)


def _blobs3() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(40, 3)) for c in [[0, 0, 0], [3, 1, 0], [-2, 2, 3]]]
    ).astype(np.float32)
    return X, np.array([0] * 40 + [1] * 40 + [2] * 40, dtype=np.int32)


def test_trace_preview_lda_linear_waterfall() -> None:
    X, y = _blobs2()
    model = LinearDiscriminantAnalysis().fit(X, y)
    model_id = _upload(model, "disc_lda.joblib")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    dec = np.asarray(model.decision_function(X[:2]), dtype=np.float64).ravel()
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {"feature_0", "feature_1"}
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(dec[i]),
            rtol=1e-5,
            atol=1e-5,
        )
        mech = tr["extra_mechanics"]
        assert mech["discriminant"] == "lda"
        assert len(mech["mahalanobis_distances"]) == 2
        assert isinstance(mech["paradox_detected"], bool)


def test_trace_preview_qda_multiclass_target_switch() -> None:
    X, y = _blobs3()
    model = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)
    model_id = _upload(model, "disc_qda.joblib")
    dec = np.asarray(model.decision_function(X[:1]), dtype=np.float64)
    for k in range(3):
        res = client.post(
            f"/api/v1/models/{model_id}/trace-preview",
            json={"rows": X[:1].tolist(), "row_indices": [0], "target_idx": k},
        )
        assert res.status_code == 200, res.text
        tr = res.json()["traces"][0]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(dec[0, k]),
            rtol=1e-5,
            atol=1e-5,
        )
        assert tr["extra_mechanics"]["target_class_idx"] == k
        assert tr["extra_mechanics"]["discriminant"] == "qda"


def test_surface_preview_discriminant_ellipsoids() -> None:
    X, y = _blobs3()
    model = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(X, y)
    model_id = _upload(model, "disc_surface.joblib")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -4.0,
            "x_max": 6.0,
            "y_min": -3.0,
            "y_max": 5.0,
            "resolution": 50,
            "fixed_feature_medians": {"2": 0.5},
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "discriminant_ellipsoids"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    ells = body["extra_overlays"]["ellipsoids_2d"]
    assert len(ells) == 3
    assert all(len(e["radii_1sigma"]) == 2 for e in ells)
    assert body["extra_overlays"]["boundary_threshold"] == 0.5
    dets = [float(np.linalg.det(np.asarray(e["eigenvectors_2d"]))) for e in ells]
    assert dets == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)

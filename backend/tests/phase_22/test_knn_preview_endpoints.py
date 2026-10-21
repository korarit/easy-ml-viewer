"""Phase 22 — trace-preview / surface-preview REST tests for KNN models."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
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


def _clf_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=80, n_features=4, n_informative=2, n_redundant=1, random_state=11
    )
    return StandardScaler().fit_transform(X).astype(np.float32), y


def _reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=80, n_features=4, n_informative=3, noise=5.0, random_state=11)
    return StandardScaler().fit_transform(X).astype(np.float32), y


def test_trace_preview_knn_classifier_radar() -> None:
    X, y = _clf_data()
    model = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X, y)
    model_id = _upload(model, "knn_clf.joblib", "knn_classifier")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    dists, _ = model.kneighbors(X[:2], return_distance=True)
    for i, tr in enumerate(traces):
        assert set(tr["feature_contributions"]) == {f"feature_{j}" for j in range(4)}
        mech = tr["extra_mechanics"]
        assert mech["n_neighbors"] == 5
        assert mech["radar_radius"] == pytest.approx(float(np.max(dists[i])))
        assert mech["radar_geometry_shape"] == "sphere"
        assert len(mech["neighbor_distances"]) == 5
        assert len(mech["neighbor_original_indices"]) == 5
        cloud_idx = set(mech["reference_cloud"]["original_indices"])
        for orig in mech["neighbor_original_indices"]:
            assert orig in cloud_idx


def test_trace_preview_knn_manhattan_octahedron() -> None:
    X, y = _clf_data()
    model = KNeighborsClassifier(n_neighbors=3, metric="manhattan").fit(X, y)
    model_id = _upload(model, "knn_manhattan.joblib", "knn_classifier")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:1].tolist(), "row_indices": [0]},
    )
    assert res.status_code == 200, res.text
    mech = res.json()["traces"][0]["extra_mechanics"]
    assert mech["effective_metric"] == "manhattan"
    assert mech["radar_geometry_shape"] == "octahedron"


def test_trace_preview_radius_outlier() -> None:
    X, y = _clf_data()
    model = RadiusNeighborsClassifier(radius=0.01, outlier_label=-1).fit(X, y)
    model_id = _upload(model, "knn_radius.joblib", "knn_classifier")
    far = np.full((1, 4), 100.0, dtype=np.float32).tolist()
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview", json={"rows": far, "row_indices": [0]}
    )
    assert res.status_code == 200, res.text
    tr = res.json()["traces"][0]
    assert tr["extra_mechanics"]["is_outlier"] is True
    assert tr["final_prediction"] == -1


def test_trace_preview_knn_regressor_mean() -> None:
    X, y = _reg_data()
    model = KNeighborsRegressor(n_neighbors=3).fit(X, y)
    model_id = _upload(model, "knn_reg.joblib", "knn_regressor")
    rows = X[:2].tolist()
    res = client.post(f"/api/v1/models/{model_id}/trace-preview", json={"rows": rows, "row_indices": [0, 1]})
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    preds = np.asarray(model.predict(X[:2])).ravel()
    for i, tr in enumerate(traces):
        assert tr["final_prediction"] == pytest.approx(float(preds[i]), rel=1e-5)
        assert tr["extra_mechanics"]["weighted_neighbor_mean"] == pytest.approx(
            float(preds[i]), rel=1e-5
        )


def test_surface_preview_knn_voronoi() -> None:
    X, y = _clf_data()
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    model_id = _upload(model, "knn_surface.joblib", "knn_classifier")
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
    assert body["visual_type"] == "voronoi_knn_proximity"
    assert len(body["z_values"]) == 2500
    assert body["telemetry"]["grid_points"] == 2500
    assert body["extra_overlays"]["radar_geometry_shape"] == "sphere"
    assert body["extra_overlays"]["n_neighbors"] == 5
    assert body["extra_overlays"]["boundary_threshold"] == 0.5

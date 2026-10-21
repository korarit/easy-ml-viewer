"""Phase 09 — surface-mesh endpoint E2E (sample regression slice)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.domains.spatial_projection.repository import SurfaceMeshCacheRepository
from app.main import app


def _bootstrap(client: TestClient, target: str | None = "selling_price"):
    ds = client.post("/api/v1/datasets/load-sample/house_prices").json()
    mdl = client.post("/api/v1/models/load-sample/house_price_linreg").json()
    auto = client.post(
        "/api/v1/mappings/auto-suggest",
        json={"dataset_id": ds["dataset_id"], "model_id": mdl["model_id"]},
    ).json()
    items = auto["mapping"]["items"]
    commit = client.post(
        "/api/v1/mappings/validate-and-save",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "items": items,
            "target_column_name": target,
        },
    )
    assert commit.status_code == 200, commit.text
    mapping = commit.json()["mapping"]
    cols = [i["dataset_column_name"] or i["model_feature_name"] for i in mapping["items"]]
    return ds, mdl, mapping, cols


def _payload(ds, mdl, mapping, cols, **overrides):
    body = {
        "dataset_id": ds["dataset_id"],
        "model_id": mdl["model_id"],
        "mapping_id": mapping["mapping_id"],
        "mode": "axis_explorer",
        "x_feature": cols[0],
        "y_feature": cols[1],
        "z_mode": "prediction",
        "target_class_index": 0,
        "pinning_mode": "median",
        "lod_level": "standard",
        "current_frame_index": 0,
        "selected_row_indices": [0, 1, 2],
    }
    body.update(overrides)
    return body


def test_axis_explorer_standard_lod_contract():
    SurfaceMeshCacheRepository.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post("/api/v1/projection/surface-mesh", json=_payload(ds, mdl, mapping, cols))
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["grid_resolution"] == 60
    assert len(data["z_heights_world"]) == 3600
    assert len(data["raw_predictions"]) == 3600
    assert data["has_constant_axis_warning"] is False
    assert data["resolved_z_mode"] == "prediction"
    assert data["x_feature"] == cols[0]
    assert data["y_feature"] == cols[1]
    assert data["x_range"][0] < data["x_range"][1]
    z = np.asarray(data["z_heights_world"], dtype=np.float64)
    assert np.isfinite(z).all()
    assert z.min() >= -7.0 - 1e-6 and z.max() <= 7.0 + 1e-6
    assert len(data["point_positions_world"]) == 3
    assert all(len(p) == 3 for p in data["point_positions_world"])
    assert len(data["trajectory_points"]) == 3
    assert data["dataset_row_count"] > 0
    print(f"\n[phase09] standard LOD compute: {data['computation_time_ms']} ms")


def test_standard_lod_latency_sla():
    """Gate: warm Standard LOD (3,600 pts) bakes in well under the 50 ms SLA
    budget (asserted at 10x headroom so CI machines never flake)."""
    SurfaceMeshCacheRepository.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    # Warmup (BLAS thread-pool spin-up + parquet page cache).
    first = client.post(
        "/api/v1/projection/surface-mesh", json=_payload(ds, mdl, mapping, cols)
    )
    assert first.status_code == 200, first.text
    SurfaceMeshCacheRepository.clear()
    second = client.post(
        "/api/v1/projection/surface-mesh", json=_payload(ds, mdl, mapping, cols)
    )
    assert second.status_code == 200, second.text
    assert second.json()["computation_time_ms"] < 500.0


def test_low_lod_and_cache_hit():
    SurfaceMeshCacheRepository.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    first = client.post(
        "/api/v1/projection/surface-mesh", json=_payload(ds, mdl, mapping, cols, lod_level="low")
    )
    assert first.status_code == 200, first.text
    assert first.json()["grid_resolution"] == 30
    assert len(first.json()["z_heights_world"]) == 900
    second = client.post(
        "/api/v1/projection/surface-mesh", json=_payload(ds, mdl, mapping, cols, lod_level="low")
    )
    assert second.status_code == 200, second.text
    assert second.json()["z_heights_world"] == first.json()["z_heights_world"]
    assert second.json()["computation_time_ms"] < 100.0


def test_pca_mode_returns_variance_and_loadings():
    SurfaceMeshCacheRepository.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post(
        "/api/v1/projection/surface-mesh",
        json=_payload(ds, mdl, mapping, cols, mode="pca_3d", lod_level="low"),
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["pca_variance_ratio"] is not None
    assert len(data["pca_variance_ratio"]) == 3
    assert 0.0 < sum(data["pca_variance_ratio"]) <= 1.0 + 1e-6
    assert data["pca_loadings"] is not None
    assert len(data["pca_loadings"]) == 3
    assert data["x_feature"].startswith("PC1")
    z = np.asarray(data["z_heights_world"], dtype=np.float64)
    assert np.isfinite(z).all()


def test_probability_falls_back_to_prediction_on_regression_model():
    SurfaceMeshCacheRepository.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post(
        "/api/v1/projection/surface-mesh",
        json=_payload(ds, mdl, mapping, cols, z_mode="probability", lod_level="low"),
    )
    assert res.status_code == 200, res.text
    assert res.json()["resolved_z_mode"] == "prediction"


def test_residual_z_mode_is_nonnegative():
    SurfaceMeshCacheRepository.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post(
        "/api/v1/projection/surface-mesh",
        json=_payload(ds, mdl, mapping, cols, z_mode="residual_error", lod_level="low"),
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["resolved_z_mode"] == "residual_error"
    raw = np.asarray(data["raw_predictions"], dtype=np.float64)
    assert (raw >= 0).all()


def test_duplicate_axes_rejected():
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    res = client.post(
        "/api/v1/projection/surface-mesh",
        json=_payload(ds, mdl, mapping, cols, y_feature=cols[0], lod_level="low"),
    )
    assert res.status_code == 422


def test_current_row_pinning_uses_frame_values():
    SurfaceMeshCacheRepository.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap(client)
    static = client.post(
        "/api/v1/projection/surface-mesh",
        json=_payload(ds, mdl, mapping, cols, pinning_mode="median", lod_level="low"),
    )
    dynamic = client.post(
        "/api/v1/projection/surface-mesh",
        json=_payload(
            ds, mdl, mapping, cols,
            pinning_mode="current_row", current_frame_index=2, lod_level="low",
        ),
    )
    assert static.status_code == 200 and dynamic.status_code == 200
    assert static.json()["raw_predictions"] != dynamic.json()["raw_predictions"]

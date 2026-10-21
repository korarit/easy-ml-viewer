"""Phase 08 — initialize-scene endpoint E2E (sample regression slice)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

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
    return ds, mdl, commit.json()["mapping"]


def test_initialize_scene_e2e_flat_hyperplane() -> None:
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    n_features = len(mapping["items"])
    assert n_features >= 2

    res = client.post(
        "/api/v1/stage3d/initialize-scene",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1 if n_features > 1 else 0,
            "selected_row_indices": [0, 1, 2],
            "grid_resolution": 10,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()

    # Axes carry raw scales + world ticks.
    for key in ("x", "y", "z"):
        axis = body["axes"][key]
        assert axis["tick_world_positions"] == [-5.0, -2.5, 0.0, 2.5, 5.0]
        assert len(axis["tick_raw_labels"]) == 5
        assert axis["raw_max"] >= axis["raw_min"]
    assert body["axes"]["x"]["feature_index"] == 0

    # Flat interleaved float32 vertex buffer: R*R*3 floats.
    surface = body["surface"]
    assert surface["visual_type"] == "flat_hyperplane"
    assert surface["resolution"] == 10
    assert len(surface["world_vertices_xyz"]) == 10 * 10 * 3
    verts = np.asarray(surface["world_vertices_xyz"], dtype=np.float64)
    assert np.isfinite(verts).all()
    assert verts.min() >= -7.0 - 1e-6 and verts.max() <= 7.0 + 1e-6

    # Trajectory points: world coords bounded, residual = actual - pred.
    points = body["trajectory_points"]
    assert len(points) == 3
    for t, p in enumerate(points):
        assert p["frame_index"] == t
        assert p["original_row_index"] == t
        for k in ("world_x", "world_y", "world_z_pred"):
            assert -7.0 - 1e-6 <= p[k] <= 7.0 + 1e-6
        if p["raw_z_actual"] is not None and p["raw_z_pred"] is not None:
            assert p["residual_error"] == abs(p["raw_z_actual"] - p["raw_z_pred"]) * (
                1.0 if p["raw_z_actual"] >= p["raw_z_pred"] else -1.0
            ) or True  # sign check below
            assert abs(p["residual_error"] - (p["raw_z_actual"] - p["raw_z_pred"])) < 1e-3


def test_initialize_scene_rejects_duplicate_axes() -> None:
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    res = client.post(
        "/api/v1/stage3d/initialize-scene",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 0,
            "selected_row_indices": [0],
            "grid_resolution": 10,
        },
    )
    assert res.status_code == 422


def test_initialize_scene_rejects_row_out_of_range() -> None:
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    res = client.post(
        "/api/v1/stage3d/initialize-scene",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "selected_row_indices": [999_999_999],
            "grid_resolution": 10,
        },
    )
    assert res.status_code == 422


def test_initialize_scene_clamps_grid_resolution() -> None:
    client = TestClient(app)
    ds, mdl, mapping = _bootstrap(client)
    res = client.post(
        "/api/v1/stage3d/initialize-scene",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "selected_row_indices": [],
            "grid_resolution": 500,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["surface"]["resolution"] == 80
    assert len(body["surface"]["world_vertices_xyz"]) == 80 * 80 * 3
    assert body["trajectory_points"] == []

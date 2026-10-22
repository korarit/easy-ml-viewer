"""Phase 43 — KNN spider-line endpoint coordinates + knn-frame endpoint."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.application.projection_orchestrator import _ContourCache
from app.domains.model_engine.adapters.distance.distance_2d_extractor import Distance2DExtractor
from app.main import app


def test_spider_endpoints_match_fit_rows(spec_knn3, spec_blobs_3d) -> None:
    """Each spider line lands exactly on its ``_fit_X`` reference row."""
    X_train, _, query = spec_blobs_3d
    res = Distance2DExtractor.extract_knn_2d_state(spec_knn3, query, x_idx=0, y_idx=1)
    assert res is not None
    dists, indices = spec_knn3.kneighbors(query)
    for nb, ref_pos, d_full in zip(res["neighbors_2d"], indices[0].tolist(), dists[0].tolist()):
        assert nb["ref_index"] == int(ref_pos)
        assert abs(nb["x"] - float(X_train[int(ref_pos), 0])) < 1e-9
        assert abs(nb["y"] - float(X_train[int(ref_pos), 1])) < 1e-9
        assert abs(nb["distance_full"] - float(d_full)) < 1e-9
        # Planar distance never exceeds the full-space distance.
        assert nb["distance_2d"] <= nb["distance_full"] + 1e-9


def test_overlapping_2d_neighbors_keep_unique_ranks() -> None:
    """E43.4: neighbors stacked on the same 2D point stay distinguishable."""
    # Same (x1, x2), different x3 → identical planar position, different depth.
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 9.0],
            [40.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    res = Distance2DExtractor.extract_knn_2d_state(knn, np.array([[0.0, 0.0, 1.0]]), x_idx=0, y_idx=1)
    assert res is not None and len(res["neighbors_2d"]) == 3
    ranks = [nb["rank"] for nb in res["neighbors_2d"]]
    assert ranks == [1, 2, 3]  # strict closeness order, no ties
    assert all(nb["distance_2d"] == 0.0 for nb in res["neighbors_2d"])
    assert res["radar_radius_2d"] == 0.0  # degenerate but valid circle


def test_large_k_spider_payload_is_capped() -> None:
    """K = 200 ships the closest 100 with an explicit cap flag."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=200).fit(X, y)
    res = Distance2DExtractor.extract_knn_2d_state(knn, np.array([[0.0, 0.0]]), x_idx=0, y_idx=1)
    assert res is not None and res["display_capped"] is True
    assert res["total_neighbors"] == 200 and len(res["neighbors_2d"]) == 100
    assert abs(sum(nb["vote_weight"] for nb in res["neighbors_2d"]) - 1.0) < 1e-9


def _bootstrap_knn(client: TestClient):
    ds = client.post("/api/v1/datasets/load-sample/iris").json()
    assert "dataset_id" in ds, ds
    mdl = client.post("/api/v1/models/load-sample/sample_iris_knn").json()
    assert "model_id" in mdl, mdl
    auto = client.post(
        "/api/v1/mappings/auto-suggest",
        json={"dataset_id": ds["dataset_id"], "model_id": mdl["model_id"]},
    ).json()
    commit = client.post(
        "/api/v1/mappings/validate-and-save",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "items": auto["mapping"]["items"],
        },
    )
    assert commit.status_code == 200, commit.text
    mapping = commit.json()["mapping"]
    cols = [i["dataset_column_name"] or i["model_feature_name"] for i in mapping["items"]]
    return ds, mdl, mapping, cols


def test_knn_frame_endpoint_contract() -> None:
    """POST /knn-frame-2d: world-mapped radar + spider payload for one dataset row."""
    _ContourCache.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap_knn(client)
    res = client.post(
        "/api/v1/projection/knn-frame-2d",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "x_feature": cols[0],
            "y_feature": cols[1],
            "row_index": 0,
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["row_index"] == 0
    assert len(data["query_world"]) == 2
    assert all(-5.5 - 1e-6 <= v <= 5.5 + 1e-6 for v in data["query_world"])
    assert len(data["neighbors"]) >= 1
    # Radar containment holds in world space (isotropic scale per axis pair
    # is preserved by the ellipse radii; raw containment is the invariant).
    for nb in data["neighbors"]:
        assert nb["rank"] >= 1 and 0.0 <= nb["vote_weight"] <= 1.0
        assert -5.5 - 1e-6 <= nb["x_world"] <= 5.5 + 1e-6
        assert -5.5 - 1e-6 <= nb["y_world"] <= 5.5 + 1e-6
    assert abs(sum(nb["vote_weight"] for nb in data["neighbors"]) - 1.0) < 1e-9
    assert data["radar_radius_raw"] >= 0.0
    assert data["computation_time_ms"] >= 0.0


def test_knn_frame_endpoint_rejects_row_out_of_range() -> None:
    _ContourCache.clear()
    client = TestClient(app)
    ds, mdl, mapping, cols = _bootstrap_knn(client)
    res = client.post(
        "/api/v1/projection/knn-frame-2d",
        json={
            "dataset_id": ds["dataset_id"],
            "model_id": mdl["model_id"],
            "mapping_id": mapping["mapping_id"],
            "x_feature": cols[0],
            "y_feature": cols[1],
            "row_index": 10_000_000,
        },
    )
    assert res.status_code == 422

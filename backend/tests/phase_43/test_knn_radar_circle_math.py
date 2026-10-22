"""Phase 43 — KNN 2D radar-circle math (radius containment + vote weights)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.domains.model_engine.adapters.distance.distance_2d_extractor import Distance2DExtractor


def test_knn_2d_radar_radius_contains_all_k_neighbors(spec_knn3, spec_blobs_3d) -> None:
    """Blueprint §2.2 prototype: every neighbor sits inside the radar circle."""
    _, _, query = spec_blobs_3d
    res = Distance2DExtractor.extract_knn_2d_state(spec_knn3, query, x_idx=0, y_idx=1)
    assert res is not None

    radar_radius = res["radar_radius_2d"]
    neighbors_2d = res["neighbors_2d"]

    assert len(neighbors_2d) == 3
    for nb in neighbors_2d:
        dist_2d = np.hypot(nb["x"] - query[0, 0], nb["y"] - query[0, 1])
        assert dist_2d <= radar_radius + 1e-5
    # Radius is tight: it equals the farthest neighbor's planar distance.
    assert abs(radar_radius - max(
        np.hypot(nb["x"] - query[0, 0], nb["y"] - query[0, 1]) for nb in neighbors_2d
    )) < 1e-9


def test_knn_uniform_weights_and_vote_tally(spec_knn3, spec_blobs_3d) -> None:
    _, _, query = spec_blobs_3d
    res = Distance2DExtractor.extract_knn_2d_state(spec_knn3, query, x_idx=0, y_idx=1)
    assert res is not None
    weights = [nb["vote_weight"] for nb in res["neighbors_2d"]]
    assert all(abs(w - 1.0 / 3) < 1e-12 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-12
    assert abs(sum(res["votes"].values()) - 1.0) < 1e-12
    assert res["predicted"] == 0  # 2 of 3 neighbors are class 0
    assert res["k"] == 3 and res["weights_mode"] == "uniform"
    assert res["outlier"] is False and res["display_capped"] is False


def test_knn_distance_weights_with_zero_distance_guard() -> None:
    """E43.1: query exactly on a reference row (d = 0) never divides by zero."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, y)
    res = Distance2DExtractor.extract_knn_2d_state(knn, np.array([[0.0, 0.0]]), x_idx=0, y_idx=1)
    assert res is not None
    weights = [nb["vote_weight"] for nb in res["neighbors_2d"]]
    assert all(np.isfinite(weights)) and abs(sum(weights) - 1.0) < 1e-12
    # The coincident point (d = 0) dominates the distance-weighted vote.
    assert res["neighbors_2d"][0]["distance_full"] == 0.0
    assert weights[0] > 0.99


def test_knn_k_equals_one_single_laser() -> None:
    """E43.2: K = 1 draws one laser line and reports a 100% vote."""
    X = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]], dtype=np.float32)
    y = np.array([0, 1, 1], dtype=np.int32)
    knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    res = Distance2DExtractor.extract_knn_2d_state(knn, np.array([[0.5, 0.0]]), x_idx=0, y_idx=1)
    assert res is not None and len(res["neighbors_2d"]) == 1
    assert res["neighbors_2d"][0]["vote_weight"] == 1.0
    assert res["neighbors_2d"][0]["rank"] == 1
    assert res["radar_radius_2d"] == res["neighbors_2d"][0]["distance_2d"]


def test_radius_model_empty_neighborhood_reports_outlier(spec_radius) -> None:
    """Radius with zero neighbors: OUTLIER telemetry, no crash."""
    far = np.array([[100.0, 100.0, 100.0]])
    res = Distance2DExtractor.extract_knn_2d_state(spec_radius, far, x_idx=0, y_idx=1)
    assert res is not None and res["is_radius_model"] is True
    assert res["outlier"] is True and res["neighbors_2d"] == []
    assert res["radar_radius_2d"] == 0.0


def test_knn_regressor_weighted_mean_matches_predict() -> None:
    rng = np.random.default_rng(43)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    y = X[:, 0] * 2.0 - X[:, 1]
    reg = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(X, y)
    q = rng.normal(size=(1, 3))
    res = Distance2DExtractor.extract_knn_2d_state(reg, q, x_idx=0, y_idx=1)
    assert res is not None
    assert abs(float(res["predicted"]) - float(reg.predict(q)[0])) < 1e-6


def test_knn_pipeline_and_non_knn_return_none(spec_blobs_3d) -> None:
    from sklearn.decomposition import PCA

    X_train, y_train, query = spec_blobs_3d
    # Scaler pipeline folds back to raw space exactly.
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
    res = Distance2DExtractor.extract_knn_2d_state(pipe, query, x_idx=0, y_idx=1)
    assert res is not None and len(res["neighbors_2d"]) == 3
    bare = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    ref = Distance2DExtractor.extract_knn_2d_state(bare, query, x_idx=0, y_idx=1)
    assert ref is not None
    assert [n["ref_index"] for n in res["neighbors_2d"]] == [n["ref_index"] for n in ref["neighbors_2d"]]
    # Dimensionality-changing pipelines have no exact raw mapping → None.
    pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=3))
    pca_pipe.fit(X_train, y_train)
    assert Distance2DExtractor.extract_knn_2d_state(pca_pipe, query) is None
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    assert Distance2DExtractor.extract_knn_2d_state(tree, query) is None

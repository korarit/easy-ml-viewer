"""Phase 51 — KNN trail elasticity (A) and vote-breakdown rings (B).

Covers mechanism A (elastic centroid + swap ripples) and mechanism B's KNN
half (vote arc segments that must tile the full turn and mirror the
classifier's own majority vote).
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from app.domains.model_engine.adapters.distance.distance_point_intelligence import (
    knn_centroid,
    knn_discordant_indices,
    knn_vote_arc_segments,
    neighbor_swap_events,
    trail_elastic_vector,
)


def _blobs():
    X, y = make_blobs(n_samples=120, centers=2, n_features=2, random_state=7)
    return X.astype(np.float64), y


def test_uniform_centroid_of_targets_equals_prediction():
    X, _ = _blobs()
    yt = X[:, 0] * 2.0 + X[:, 1]
    reg = KNeighborsRegressor(n_neighbors=5).fit(X, yt)
    q = X[:3]
    _, idx = reg.kneighbors(q)
    for r in range(3):
        # Parity identity (mirrors knn.py): uniform mean of neighbor
        # *targets* reproduces predict; the 2D elastic centroid below is
        # the same kernel applied to planar neighbor coordinates.
        c = knn_centroid(yt[idx[r]].reshape(-1, 1))
        assert float(c[0]) == pytest.approx(float(reg.predict(q[r : r + 1])[0]), rel=1e-9)
    # Feature-space hand check: mean of the K neighbor points.
    c2 = knn_centroid(np.array([[0.0, 0.0], [2.0, 4.0]]))
    assert c2 == pytest.approx([1.0, 2.0])


def test_distance_weighted_centroid_matches_predict():
    X, _ = _blobs()
    yt = X[:, 0] * 2.0 + X[:, 1]
    reg = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(X, yt)
    q = X[:4]
    dists, idx = reg.kneighbors(q)
    for r in range(4):
        w = 1.0 / np.maximum(dists[r], 1e-12)
        w = w / w.sum()
        c = knn_centroid(yt[idx[r]].reshape(-1, 1), w)
        assert float(c[0]) == pytest.approx(float(reg.predict(q[r : r + 1])[0]), rel=1e-9)


def test_swap_events_detect_entered_and_exited():
    ev = neighbor_swap_events(np.array([1, 2, 3]), np.array([2, 3, 4]))
    assert ev == {"entered": [4], "exited": [1], "kept": [2, 3]}
    ev2 = neighbor_swap_events(np.array([5, 6]), np.array([5, 6]))
    assert ev2 == {"entered": [], "exited": [], "kept": [5, 6]}


def test_elastic_vector_points_from_active_to_centroid():
    v = trail_elastic_vector(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert v == pytest.approx([3.0, 4.0])
    bad = trail_elastic_vector(np.array([np.nan, 0.0]), np.array([1.0, 1.0]))
    assert (bad == 0.0).all()  # non-finite input → no tether


def test_vote_arcs_tile_full_turn_and_match_majority():
    X, y = _blobs()
    clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    q = X[:6]
    dists, idx = clf.kneighbors(q)
    for r in range(6):
        labs = [str(v) for v in y[idx[r]]]
        segs = knn_vote_arc_segments(labs, np.full(5, 1.0 / 5))
        assert sum(s["sweep"] for s in segs) == pytest.approx(1.0)
        winner = max(segs, key=lambda s: s["share"])["label"]
        assert str(clf.predict(q[r : r + 1])[0]) == winner


def test_discordant_indices_flag_votes_against_truth():
    idx = knn_discordant_indices(["a", "b", "b", "a", "b"], "a")
    assert idx == [1, 2, 4]
    assert knn_discordant_indices(["a", "a"], "a") == []

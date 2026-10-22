"""Phase 45 — leaf-box partition geometry + active-leaf walk parity."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from app.domains.model_engine.adapters.trees.tree_2d_extractor import Tree2DExtractor
from conftest import make_ctx


def test_active_leaf_bounding_box_contains_query_point() -> None:
    """Test-plan §2.2 — the active leaf box always covers the query point."""
    X = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
        [40.0, 400.0]
    ], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
    query_point = np.array([15.0, 150.0], dtype=np.float32)

    leaf_box = Tree2DExtractor.compute_active_leaf_box(
        tree, query_point, x_idx=0, y_idx=1, x_range=(0.0, 50.0), y_range=(50.0, 500.0)
    )

    assert leaf_box["x_min"] <= query_point[0] <= leaf_box["x_max"]
    assert leaf_box["y_min"] <= query_point[1] <= leaf_box["y_max"]


def test_active_leaf_matches_sklearn_apply(fitted_tree, blobs_2d):
    """The walked leaf id equals sklearn's own ``apply`` on every training row."""
    X, _ = blobs_2d
    _, _, xr, yr = make_ctx(X)
    native = fitted_tree.apply(X)
    for row, leaf in zip(X[:50], native[:50]):
        box = Tree2DExtractor.compute_active_leaf_box(
            fitted_tree, row, x_idx=0, y_idx=1, x_range=xr, y_range=yr,
            extra_values={},
        )
        assert box["leaf_id"] == int(leaf)
        assert box["x_min"] <= row[0] <= box["x_max"]
        assert box["y_min"] <= row[1] <= box["y_max"]
        assert box["off_axis_ambiguous"] is False


def test_exact_threshold_goes_left(fitted_tree, blobs_2d):
    """E45.4 — a query exactly on a threshold follows ``<=`` (left), like sklearn."""
    X, _ = blobs_2d
    _, _, xr, yr = make_ctx(X)
    tree = fitted_tree.tree_
    feat = np.asarray(tree.feature).ravel()
    thr = np.asarray(tree.threshold).ravel()
    node = next(i for i in range(feat.size) if int(feat[i]) == 0)
    t = float(thr[node])
    q = X[0].copy()
    q[0] = t  # exactly on the X threshold
    box = Tree2DExtractor.compute_active_leaf_box(
        fitted_tree, q, x_idx=0, y_idx=1, x_range=xr, y_range=yr, extra_values={})
    native = int(fitted_tree.apply(q.reshape(1, -1))[0])
    assert box["leaf_id"] == native
    assert box["x_max"] <= t + 1e-9  # left branch taken


def test_leaf_boxes_partition_the_rect(fitted_tree, blobs_2d, fitted_regressor):
    """Leaves are non-overlapping and every box carries prediction + stats."""
    from app.domains.model_engine.adapters.trees.tree_2d_extractor import Tree2DExtractor as T

    X, _ = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = dict(X=X, x_feature_idx=0, y_feature_idx=1, x_feature_name="f0",
              y_feature_name="f1", x_range_raw=xr, y_range_raw=yr,
              x_scale=xs, y_scale=ys)
    for model, is_reg in ((fitted_tree, False), (fitted_regressor, True)):
        geo = T.extract_trees_2d_geometry(model_obj=model, **kw)
        assert geo is not None
        assert len(geo.leaves) == int(model.tree_.n_leaves)
        assert geo.leaves_truncated is False
        for leaf in geo.leaves:
            xmin, xmax, ymin, ymax = leaf.box_raw
            assert xmin < xmax and ymin < ymax
            assert leaf.samples > 0
            assert leaf.impurity >= 0.0
            assert len(leaf.box_world) == 4
            if not is_reg:
                assert leaf.prediction in (0, 1)
        # Pairwise disjoint interiors (shared edges allowed).
        boxes = [l.box_raw for l in geo.leaves]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                inter_x = min(a[1], b[1]) - max(a[0], b[0])
                inter_y = min(a[3], b[3]) - max(a[2], b[2])
                assert inter_x <= 1e-9 or inter_y <= 1e-9
        # Leaf prediction matches the fitted tree.
        for leaf in geo.leaves[:8]:
            cx = (leaf.box_raw[0] + leaf.box_raw[1]) / 2
            cy = (leaf.box_raw[2] + leaf.box_raw[3]) / 2
            row = np.median(X, axis=0)
            row[0], row[1] = cx, cy
            native = model.predict(row.reshape(1, -1))[0]
            try:
                native = native.item()
            except (ValueError, TypeError, AttributeError):
                pass
            assert leaf.prediction == native or is_reg and abs(float(leaf.prediction) - float(native)) < 1e-6


def test_off_axis_walk_flags_ambiguity(blobs_4d):
    """Off-axis splits without values: left branch + honest ambiguity flag."""
    X, y = blobs_4d
    tree = None
    for seed in range(10):
        cand = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(X, y)
        feats = {int(f) for f in np.asarray(cand.tree_.feature).ravel() if int(f) >= 0}
        if feats - {0, 1}:
            tree = cand
            break
    assert tree is not None, "need a tree with an off-axis split"
    _, _, xr, yr = make_ctx(X, 0, 1)
    box = Tree2DExtractor.compute_active_leaf_box(
        tree, X[0, :2], x_idx=0, y_idx=1, x_range=xr, y_range=yr)
    assert box["off_axis_ambiguous"] is True
    # With full values the walk resolves exactly like sklearn.
    box2 = Tree2DExtractor.compute_active_leaf_box(
        tree, X[0, :2], x_idx=0, y_idx=1, x_range=xr, y_range=yr,
        extra_values={2: float(X[0, 2]), 3: float(X[0, 3])})
    assert box2["off_axis_ambiguous"] is False
    assert box2["leaf_id"] == int(tree.apply(X[0].reshape(1, -1))[0])

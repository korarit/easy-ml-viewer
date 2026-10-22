"""Phase 45 — on-axis split extraction, stump + off-axis edge cases."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from app.domains.model_engine.adapters.trees.tree_2d_extractor import Tree2DExtractor
from conftest import make_ctx


def test_splits_match_tree_nodes(fitted_tree, blobs_2d):
    X, _ = blobs_2d
    _, _, xr, yr = make_ctx(X)
    out = Tree2DExtractor.extract_tree_partitions(fitted_tree, 0, 1, xr, yr)
    tree = fitted_tree.tree_
    feat = np.asarray(tree.feature).ravel()
    on_axis = {i for i in range(feat.size) if int(feat[i]) in (0, 1)}
    assert out["node_count"] == int(tree.node_count)
    assert out["leaf_count"] == int(tree.n_leaves)
    assert len(out["splits"]) == len(on_axis)
    for s in out["splits"]:
        node = s["node_id"]
        assert s["threshold"] == float(np.asarray(tree.threshold).ravel()[node])
        assert s["axis"] == ("x" if s["feature_index"] == 0 else "y")


def test_stump_single_cut_two_zones(blobs_2d):
    """E45.2 — a depth-1 stump renders one crisp cut splitting the rect in two."""
    from app.domains.model_engine.adapters.trees.tree_2d_extractor import Tree2DExtractor as T

    X, y = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    geo = T.extract_trees_2d_geometry(
        model_obj=stump, X=X, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys,
    )
    assert geo is not None and geo.family == "decision_tree"
    assert len(geo.segments) == 1
    assert len(geo.leaves) == 2
    seg = geo.segments[0].segment_raw
    # The cut spans its axis fully: vertical spans y_range, horizontal spans x_range.
    if geo.segments[0].axis == "x":
        assert abs(seg[1] - yr[0]) < 1e-6 and abs(seg[3] - yr[1]) < 1e-6
        assert seg[0] == seg[2]
    else:
        assert abs(seg[0] - xr[0]) < 1e-6 and abs(seg[2] - xr[1]) < 1e-6
        assert seg[1] == seg[3]


def test_no_axis_splits_flag(blobs_4d):
    """E45.5 — splits only on non-visualized axes: no segments + honest flag."""
    from app.domains.model_engine.adapters.trees.tree_2d_extractor import Tree2DExtractor as T

    X, y = blobs_4d
    xs, ys, _, _ = make_ctx(X, 0, 1)
    # Axes 2/3 carry the class signal; find a seed whose tree splits there only.
    informative = np.hstack([X[:, 2:3] * 0.01, X[:, 1:2] * 0.01, X[:, 2:]])
    tree = None
    for seed in range(10):
        cand = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(informative, y)
        feats = set(int(f) for f in np.asarray(cand.tree_.feature).ravel() if int(f) >= 0)
        if feats and feats.isdisjoint({0, 1}):
            tree = cand
            break
    assert tree is not None, "could not grow an off-axis-only tree in 10 seeds"
    geo = T.extract_trees_2d_geometry(
        model_obj=tree, X=informative, x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        x_range_raw=(float(informative[:, 0].min()), float(informative[:, 0].max())),
        y_range_raw=(float(informative[:, 1].min()), float(informative[:, 1].max())),
        x_scale=xs, y_scale=ys,
    )
    assert geo is not None
    assert geo.segments == []
    assert geo.no_axis_splits is True


def test_foreign_and_pipeline_models_yield_none(blobs_2d):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from app.domains.model_engine.adapters.trees.tree_2d_extractor import Tree2DExtractor as T

    X, y = blobs_2d
    xs, ys, xr, yr = make_ctx(X)
    kw = dict(X=X, x_feature_idx=0, y_feature_idx=1, x_feature_name="f0",
              y_feature_name="f1", x_range_raw=xr, y_range_raw=yr,
              x_scale=xs, y_scale=ys)
    assert T.extract_trees_2d_geometry(model_obj=LogisticRegression().fit(X, y), **kw) is None
    assert T.extract_trees_2d_geometry(model_obj=GaussianNB().fit(X, y), **kw) is None
    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=2).fit(X, y))
    pipe.fit(X, y)
    assert T.extract_trees_2d_geometry(model_obj=pipe, **kw) is None

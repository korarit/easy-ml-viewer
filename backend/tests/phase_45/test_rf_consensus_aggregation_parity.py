"""Phase 45 — forest consensus parity, AdaBoost stages, caps & contour binding."""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from app.domains.model_engine.adapters.trees.tree_2d_extractor import (
    MAX_FOREST_TREES,
    Tree2DExtractor,
)
from conftest import make_ctx


def _kw(X, x_idx=0, y_idx=1):
    xs, ys, xr, yr = make_ctx(X, x_idx, y_idx)
    return dict(X=X, x_feature_idx=x_idx, y_feature_idx=y_idx,
                x_feature_name="f0", y_feature_name="f1",
                x_range_raw=xr, y_range_raw=yr, x_scale=xs, y_scale=ys)


def test_forest_exports_top_trees_with_caps(fitted_forest, blobs_2d):
    X, _ = blobs_2d
    geo = Tree2DExtractor.extract_trees_2d_geometry(model_obj=fitted_forest, **_kw(X))
    assert geo is not None and geo.family == "random_forest"
    assert geo.n_estimators == 10 and geo.trees_exported == 10
    assert geo.leaves == []  # overlapping forest boxes are never exported
    assert len(geo.segments) > 0
    assert {s.tree_index for s in geo.segments} <= set(range(10))
    assert geo.segments_truncated is False
    # Every segment lies on its axis inside the data rect (the extractor
    # outward-rounds rect edges to 6dp, hence the 2e-6 tolerance).
    _, _, xr, yr = make_ctx(X)
    for s in geo.segments:
        x1, y1, x2, y2 = s.segment_raw
        if s.axis == "x":
            assert x1 == x2 and yr[0] - 2e-6 <= y1 <= y2 <= yr[1] + 2e-6
        else:
            assert y1 == y2 and xr[0] - 2e-6 <= x1 <= x2 <= xr[1] + 2e-6
        assert len(s.segment_world) == 4


def test_forest_consensus_matches_mean_tree_vote(fitted_forest, blobs_2d):
    """Consensus parity: forest predict == majority of exported trees' votes."""
    X, y = blobs_2d
    proba = fitted_forest.predict_proba(X[:20])
    votes = np.asarray([t.predict_proba(X[:20]) for t in fitted_forest.estimators_])
    assert np.allclose(proba, votes.mean(axis=0), atol=1e-6)
    geo = Tree2DExtractor.extract_trees_2d_geometry(model_obj=fitted_forest, **_kw(X))
    assert geo is not None
    assert f"over {fitted_forest.n_estimators} trees" in geo.formula


def test_extratrees_family_label(fitted_extratrees, blobs_2d):
    X, _ = blobs_2d
    geo = Tree2DExtractor.extract_trees_2d_geometry(model_obj=fitted_extratrees, **_kw(X))
    assert geo is not None and geo.family == "extra_trees"
    assert geo.algorithm == "ExtraTreesClassifier"


def test_adaboost_stages_match_weak_learners(fitted_adaboost, hard_blobs):
    """Stage weights/thresholds mirror estimators_ + estimator_weights_."""
    X, _ = hard_blobs
    geo = Tree2DExtractor.extract_trees_2d_geometry(model_obj=fitted_adaboost, **_kw(X))
    assert geo is not None and geo.family == "adaboost"
    n_stages = len(fitted_adaboost.estimators_)
    assert n_stages > 1, "hard blobs should keep boosting past round 1"
    assert geo.n_estimators == n_stages and len(geo.stages) == n_stages
    assert geo.stages_truncated is False
    assert geo.base_value == 0.5  # binary base prediction (E45.6)
    for stage, sub, w in zip(geo.stages, fitted_adaboost.estimators_,
                             fitted_adaboost.estimator_weights_):
        assert stage.stage >= 1 and abs(stage.weight - float(w)) < 1e-6
        f0 = int(np.asarray(sub.tree_.feature).ravel()[0])
        assert stage.feature_index == f0
        if f0 in (0, 1):
            assert stage.axis in ("x", "y") and stage.segment_world is not None
            assert abs(stage.threshold - float(np.asarray(sub.tree_.threshold).ravel()[0])) < 1e-6
        else:
            assert stage.axis == "off"
    # Staged parity: alpha-weighted stump votes track staged_predict.
    staged = list(fitted_adaboost.staged_predict(X[:30]))
    assert staged[-1].shape == (30,)
    assert (staged[-1] == fitted_adaboost.predict(X[:30])).all()


def test_adaboost_stage_cap(hard_blobs):
    X, y = hard_blobs
    big = AdaBoostClassifier(n_estimators=60, random_state=0).fit(X, y)
    assert len(big.estimators_) > 50, "hard blobs should run all 60 rounds"
    geo = Tree2DExtractor.extract_trees_2d_geometry(model_obj=big, **_kw(X))
    assert geo is not None and geo.n_estimators == 60
    assert len(geo.stages) == 50 and geo.stages_truncated is True


def test_deep_tree_caps_and_prefers_shallow_cuts():
    """E45.1 — depth-30 trees stay bounded; shallow principal cuts come first."""
    from sklearn.datasets import make_blobs
    from sklearn.tree import DecisionTreeClassifier

    from app.domains.model_engine.adapters.trees.tree_2d_extractor import (
        MAX_LEAVES,
        MAX_SEGMENTS,
    )

    # Pure label noise on uniform data: the tree keeps splitting to depth 30
    # and grows thousands of nodes (worst case for the 2D overlay).
    rng = np.random.default_rng(3)
    X = rng.uniform(-10.0, 10.0, size=(8000, 2)).astype(np.float64)
    y = rng.integers(0, 2, size=8000).astype(np.int64)
    deep = DecisionTreeClassifier(max_depth=30, random_state=0).fit(X, y)
    assert deep.tree_.node_count > MAX_SEGMENTS, "test needs a big tree"
    geo = Tree2DExtractor.extract_trees_2d_geometry(model_obj=deep, **_kw(X))
    assert geo is not None
    assert len(geo.segments) <= MAX_SEGMENTS and geo.segments_truncated is True
    assert len(geo.leaves) <= MAX_LEAVES + 1 and geo.leaves_truncated is True
    depths = [s.depth for s in geo.segments]
    assert depths == sorted(depths), "pre-order must emit shallow cuts first"


def test_large_forest_extraction_is_fast(blobs_2d):
    """E45.3 — structure-only walk stays far under budget (no re-inference)."""
    X, y = blobs_2d
    forest = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0,
                                    n_jobs=-1).fit(X, y)
    started = time.perf_counter()
    geo = Tree2DExtractor.extract_trees_2d_geometry(model_obj=forest, **_kw(X))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert geo is not None and geo.trees_exported == MAX_FOREST_TREES
    assert elapsed_ms < 2000.0, f"extraction took {elapsed_ms:.1f} ms"


def test_contour_integration_and_dto_budget(fitted_tree, fitted_forest, blobs_2d):
    import json

    from app.domains.model_engine.registry import ModelRegistry
    from app.domains.simulation.contour_2d_generator import Contour2DGenerator
    from sklearn.linear_model import LogisticRegression

    X, y = blobs_2d
    X32 = X.astype(np.float32)
    for model, family in ((fitted_tree, "decision_tree"), (fitted_forest, "random_forest")):
        adapter = ModelRegistry.resolve_adapter(model, ".joblib")
        res = Contour2DGenerator.generate_contour(
            adapter=adapter, model_obj=model, final_est=model, X=X32,
            x_feature_idx=0, y_feature_idx=1,
            x_feature_name="f0", y_feature_name="f1",
            grid_resolution=32, pinning_mode="median",
        )
        assert res.trees_2d is not None and res.trees_2d["family"] == family
        assert len(json.dumps(res.trees_2d).encode("utf-8")) < 40_960
    lin = LogisticRegression(max_iter=500).fit(X32, y)
    lin_adapter = ModelRegistry.resolve_adapter(lin, ".joblib")
    res_lin = Contour2DGenerator.generate_contour(
        adapter=lin_adapter, model_obj=lin, final_est=lin, X=X32,
        x_feature_idx=0, y_feature_idx=1,
        x_feature_name="f0", y_feature_name="f1",
        grid_resolution=32, pinning_mode="median",
    )
    assert res_lin.trees_2d is None

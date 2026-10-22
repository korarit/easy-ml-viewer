"""Phase 27 — decision-tree adapter math identity tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

from app.domains.model_engine.adapters.trees.decision_tree import (
    DecisionTreeClassifierAdapter,
    DecisionTreeRegressorAdapter,
    _split_depths,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import GridSpec3D, Visual3DType

FEATURES_4 = ["f0", "f1", "f2", "f3"]


@pytest.fixture()
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(27)
    a = rng.normal(size=(120, 2)) @ np.array([[1.6, 0.9], [0.9, 0.6]])
    b = rng.normal(size=(120, 2)) @ np.array([[0.5, -0.8], [-0.8, 1.8]]) + np.array([3.0, 1.0])
    X = np.vstack([a, b]).astype(np.float32)
    return X, np.array([0] * 120 + [1] * 120, dtype=np.int32)


@pytest.fixture()
def multi_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(28)
    X = np.vstack(
        [rng.normal(c, 0.6, size=(50, 4)) for c in [[0, 0, 0, 0], [3, 1, 0, 2], [-2, 2, 3, 0]]]
    ).astype(np.float32)
    return X, np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int32)


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    X = rng.normal(size=(150, 3)).astype(np.float32)
    return X, (X[:, 0] * 2.0 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": -4.0,
        "x_max": 6.0,
        "y_min": -3.0,
        "y_max": 5.0,
        "resolution": 20,
        "fixed_feature_medians": {2: 0.5, 3: 1.0},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def test_registry_resolves_all_tree_variants(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    cases = [
        (DecisionTreeClassifier(max_depth=4).fit(Xb, yb), "decision_tree"),
        (ExtraTreeClassifier(max_depth=4).fit(Xb, yb), "decision_tree"),
        (DecisionTreeRegressor(max_depth=4).fit(Xr, yr), "decision_tree_regressor"),
        (ExtraTreeRegressor(max_depth=4).fit(Xr, yr), "decision_tree_regressor"),
    ]
    for model, key in cases:
        adapter = ModelRegistry.resolve_adapter(model)
        assert adapter.algorithm_key == key
        assert adapter.visual_3d_type == Visual3DType.STEPPED_AXIS_BLOCKS


def test_classifier_waterfall_matches_leaf_proba(binary_data, multi_data) -> None:
    Xb, yb = binary_data
    for cls in (DecisionTreeClassifier, ExtraTreeClassifier):
        model = cls(max_depth=5, random_state=1).fit(Xb, yb)
        adapter = DecisionTreeClassifierAdapter()
        batch = adapter.predict_batch(model, Xb[:10])
        np.testing.assert_array_equal(np.asarray(batch.predictions), model.predict(Xb[:10]))
        np.testing.assert_allclose(
            np.asarray(batch.probabilities), np.asarray(model.predict_proba(Xb[:10])), rtol=1e-9
        )
        proba = np.asarray(model.predict_proba(Xb[:10]), dtype=np.float64)[:, 1]
        traces = adapter.trace_internal_flow(model, Xb[:10], list(range(10)), ["a", "b"])
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(proba[i]),
                rtol=1e-9,
                atol=1e-9,
            )
            mech = tr.extra_mechanics
            assert mech["leaf_id"] == int(model.apply(Xb[:10])[i])
            assert mech["leaf_expectation"] == pytest.approx(float(proba[i]), rel=1e-9)
            assert len(mech["active_path"]) >= 0
            assert len(mech["path_node_ids"]) == len(mech["active_path"]) + 1
            # CSR slice: path starts at root 0 and ends at the leaf.
            assert mech["path_node_ids"][0] == 0
            assert mech["path_node_ids"][-1] == mech["leaf_id"]
            assert mech["criterion"] == "gini"

    Xm, ym = multi_data
    multi = DecisionTreeClassifier(max_depth=5, random_state=1).fit(Xm, ym)
    for k in range(3):
        traces = DecisionTreeClassifierAdapter().trace_internal_flow(
            multi, Xm[:5], list(range(5)), FEATURES_4, target_class_idx=k
        )
        proba_k = np.asarray(multi.predict_proba(Xm[:5]), dtype=np.float64)[:, k]
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(proba_k[i]),
                rtol=1e-9,
                atol=1e-9,
            )
            assert tr.extra_mechanics["resolved_class_idx"] == k


def test_regressor_waterfall_matches_prediction(reg_data) -> None:
    X, y = reg_data
    for cls in (DecisionTreeRegressor, ExtraTreeRegressor):
        model = cls(max_depth=5, random_state=1).fit(X, y)
        adapter = DecisionTreeRegressorAdapter()
        preds = np.asarray(model.predict(X[:10]), dtype=np.float64)
        traces = adapter.trace_internal_flow(model, X[:10], list(range(10)), ["a", "b", "c"])
        for i, tr in enumerate(traces):
            np.testing.assert_allclose(
                tr.base_value + sum(tr.feature_contributions.values()),
                float(preds[i]),
                rtol=1e-9,
                atol=1e-9,
            )
            assert tr.extra_mechanics["is_regressor"] is True
            assert tr.extra_mechanics["probabilities"] == []


def test_stump_nan_pipeline_calibrated(binary_data) -> None:
    Xb, yb = binary_data
    adapter = DecisionTreeClassifierAdapter()
    # Stump: single root-leaf node.
    stump = DecisionTreeClassifier().fit(
        np.ones((20, 2), dtype=np.float32), np.zeros(20, dtype=int)
    )
    assert stump.tree_.node_count == 1
    traces = adapter.trace_internal_flow(
        stump, np.ones((2, 2), dtype=np.float32), [0, 1], ["a", "b"]
    )
    for tr in traces:
        assert tr.extra_mechanics["active_path"] == []
        assert tr.extra_mechanics["leaf_id"] == 0
        assert sum(tr.feature_contributions.values()) == 0.0
    # NaN rows route natively; went_left follows the taken child.
    model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(Xb, yb)
    nan_row = Xb[:1].copy()
    nan_row[0, 0] = np.nan
    tr_nan = adapter.trace_internal_flow(model, nan_row, [0], ["a", "b"])[0]
    assert np.all(np.isfinite(list(tr_nan.feature_contributions.values())))
    for step in tr_nan.extra_mechanics["active_path"]:
        tl = model.tree_.children_left[step["node_id"]]
        assert step["went_left"] == (step["next_node_id"] == int(tl))
    # Pipeline wrapper resolves and predicts through the full chain.
    pipe = Pipeline([("scaler", StandardScaler()), ("tree", DecisionTreeClassifier(max_depth=4))])
    pipe.fit(Xb, yb)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "decision_tree"
    traces_pipe = adapter.trace_internal_flow(pipe, Xb[:3], [0, 1, 2], ["a", "b"])
    assert [t.final_prediction for t in traces_pipe] == list(pipe.predict(Xb[:3]))
    # Calibrated wrapper unwraps to the inner tree for tracing.
    cal = CalibratedClassifierCV(DecisionTreeClassifier(max_depth=3), cv=2).fit(Xb, yb)
    assert ModelRegistry.resolve_adapter(cal).algorithm_key == "decision_tree"
    tr_cal = adapter.trace_internal_flow(cal, Xb[:2], [0, 1], ["a", "b"])
    assert all(t.extra_mechanics["calibrated_wrapper"] is True for t in tr_cal)
    assert len(tr_cal) == 2


def test_split_depths_and_topology_cap(multi_data) -> None:
    Xm, ym = multi_data
    deep = DecisionTreeClassifier(random_state=1).fit(Xm, ym)
    n = int(deep.tree_.node_count)
    depth = _split_depths(
        np.asarray(deep.tree_.children_left), np.asarray(deep.tree_.children_right)
    )
    assert depth[0] == 0 and np.all(depth >= 0)
    leaves = np.where(np.asarray(deep.tree_.children_left) == -1)[0]
    assert np.all(depth[leaves] >= 1) or n == 1
    assert len(np.unique(depth[leaves])) >= 1
    topo = DecisionTreeClassifierAdapter().trace_internal_flow(
        deep, Xm[:1], [0], FEATURES_4
    )[0].extra_mechanics["tree_topology"]
    assert topo["node_count"] == n
    assert topo["exported_nodes"] == min(n, 512)
    assert topo["truncated"] == (n > 512)
    assert len(topo["children_left"]) == min(n, 512)


def test_surface_terraces_parity_and_splits(binary_data, reg_data) -> None:
    Xb, yb = binary_data
    Xr, yr = reg_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=1).fit(Xb, yb)
    adapter = DecisionTreeClassifierAdapter()
    mesh = adapter.compute_3d_surface(clf, _spec(), n_features=2)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    proba = np.asarray(clf.predict_proba(np.zeros((400, 2), dtype=np.float32)))
    grid = np.zeros((400, 2), dtype=np.float32)
    xs = np.linspace(-4.0, 6.0, 20, dtype=np.float32)
    ys = np.linspace(-3.0, 5.0, 20, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid[:, 0], grid[:, 1] = xx.ravel(), yy.ravel()
    expected = np.asarray(clf.predict_proba(grid), dtype=np.float64)[:, 1]
    np.testing.assert_allclose(np.asarray(mesh.z_values), expected, rtol=1e-6)
    # Piecewise-constant: z takes at most n_leaves distinct values.
    assert len(np.unique(np.asarray(mesh.z_values))) <= int(clf.tree_.n_leaves)
    topo = mesh.extra_overlays["tree_topology"]
    assert topo["node_count"] == int(clf.tree_.node_count)
    for key in ("split_lines_x", "split_lines_y"):
        lines = mesh.extra_overlays[key]
        assert all("threshold" in e and "depth" in e for e in lines)
        depths = [e["depth"] for e in lines]
        assert depths == sorted(depths)
    assert mesh.extra_overlays["boundary_threshold"] == 0.5

    reg = DecisionTreeRegressor(max_depth=5, random_state=1).fit(Xr, yr)
    mesh_r = DecisionTreeRegressorAdapter().compute_3d_surface(
        reg,
        GridSpec3D(
            axis_x_feature_index=0,
            axis_y_feature_index=1,
            x_min=-2.0,
            x_max=2.0,
            y_min=-2.0,
            y_max=2.0,
            resolution=20,
            fixed_feature_medians={2: 0.0},
        ),
        n_features=3,
    )
    assert mesh_r.extra_overlays["is_regressor"] is True
    assert mesh_r.extra_overlays["boundary_threshold"] is None
    z = np.asarray(mesh_r.z_values)
    assert np.all(np.isfinite(z))
    lo, hi = mesh_r.extra_overlays["regression_z_range"]
    assert lo == pytest.approx(float(z.min())) and hi == pytest.approx(float(z.max()))

    big = _spec(resolution=200)
    assert adapter.compute_3d_surface(clf, big, n_features=2).resolution == 100

    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)
    assert len(adapter.trace_internal_flow(joblib.load(buf), Xb[:2], [0, 1], ["a", "b"])) == 2

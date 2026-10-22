"""Phase 38 — ONNX classical ML inspector math tests (ai.onnx.ml decompiler).

Covers TreeEnsemble / Linear / SVM graphs converted via ``skl2onnx``:
ZipMap bypass parity, waterfall identity, split-plane extraction,
single-vs-multi input tensors, and registry/schema wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")

from skl2onnx import convert_sklearn  # noqa: E402
from skl2onnx.common.data_types import FloatTensorType, StringTensorType  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

import app.domains.model_engine.adapters.neural_onnx.onnx_classical as oc  # noqa: E402
from app.domains.model_engine.registry import ModelRegistry  # noqa: E402
from app.domains.model_engine.schema_detector import UniversalSchemaDetector  # noqa: E402
from app.domains.model_engine.value_objects import GridSpec3D  # noqa: E402


def _convert(model: object, n_features: int):  # type: ignore[no-untyped-def]
    return convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        target_opset=12,
    )


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "resolution": 20,
        "fixed_feature_medians": {},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def _binary_data(seed: int = 38, n: int = 80, width: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, width)).astype(np.float32)
    y = (X[:, 0] * 1.5 - X[:, 1] > 0).astype(int)
    return X, y


def test_inspect_detects_tree_linear_svm_families() -> None:
    X, y = _binary_data()
    cases = {
        "tree_ensemble": DecisionTreeClassifier(max_depth=3).fit(X, y),
        "linear": LogisticRegression(max_iter=200).fit(X, y),
        "svm": SVC(kernel="rbf", probability=True).fit(X, y),
    }
    for family, model in cases.items():
        proto = _convert(model, 4)
        meta = oc.inspect_onnx_classical(proto)
        assert meta.operator_family == family, meta
        assert meta.has_zipmap is True
        assert meta.n_features == 4
        assert meta.classes == [0, 1]
        assert any("ai.onnx.ml" in step for step in meta.node_chain)


def test_linear_zipmap_bypass_parity_and_waterfall_identity() -> None:
    X, y = _binary_data()
    model = LogisticRegression(max_iter=200).fit(X, y)
    proto = _convert(model, 4)
    adapter = ModelRegistry.resolve_adapter(proto, ".onnx")
    assert adapter.algorithm_key == "onnx_classical"
    resolved = adapter._build_optimized_session(proto)
    assert resolved.bypass_active is True
    batch = adapter.predict_batch(proto, X[:8])
    assert batch.probabilities is not None
    mat = np.asarray(batch.probabilities, dtype=np.float64)
    assert mat.shape == (8, 2)
    assert mat.dtype == np.float64  # payload; session itself is float32
    np.testing.assert_allclose(mat, model.predict_proba(X[:8]), rtol=1e-5, atol=1e-5)
    traces = adapter.trace_internal_flow(proto, X[:8], [0, 1], ["a", "b", "c", "d"])
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(mat[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr.extra_mechanics["operator_family"] == "linear"
        assert tr.extra_mechanics["domain_badge"].startswith("ONNX ai.onnx.ml")


def test_linear_regressor_exact_decomposition() -> None:
    rng = np.random.default_rng(41)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.5).astype(np.float32)
    model = LinearRegression().fit(X, y)
    proto = _convert(model, 3)
    adapter = ModelRegistry.resolve_adapter(proto, ".onnx")
    assert adapter.algorithm_key == "onnx_classical_regressor"
    traces = adapter.trace_internal_flow(proto, X[:4], [0, 2], ["a", "b", "c"])
    preds = np.asarray(model.predict(X[:4]), dtype=np.float64)
    for i, idx in enumerate([0, 2]):
        tr = traces[i]
        assert tr.row_index == idx
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(preds[idx]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr.extra_mechanics["attribution_method"] == "linear_exact"
    weights = adapter.extract_feature_weights(proto, 3)
    assert len(weights) == 3 and all(w is not None for w in weights)


def test_tree_ensemble_path_share_identity_and_splits() -> None:
    X, y = _binary_data()
    model = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=38, n_jobs=1).fit(X, y)
    proto = _convert(model, 4)
    adapter = ModelRegistry.resolve_adapter(proto, ".onnx")
    assert adapter.algorithm_key == "onnx_classical"
    batch = adapter.predict_batch(proto, X[:6])
    assert batch.probabilities is not None
    mat = np.asarray(batch.probabilities, dtype=np.float64)
    np.testing.assert_allclose(mat, model.predict_proba(X[:6]), rtol=1e-5, atol=1e-5)
    traces = adapter.trace_internal_flow(proto, X[:6], [0, 1], ["a", "b", "c", "d"])
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(mat[i, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        mech = tr.extra_mechanics
        assert mech["attribution_method"] == "tree_path_share"
        assert mech["n_trees"] == 5
        assert mech["n_path_steps"] >= 5  # >= one split-visit per tree path
        assert mech["active_leaf_count"] == 5
        assert len(mech["leaf_trace"]) == 5
    splits = adapter.decompile_tree_ensemble_splits(proto, 0, 1)
    assert set(splits) == {"axis_x_thresholds", "axis_y_thresholds"}
    assert all(isinstance(v, float) for v in splits["axis_x_thresholds"])
    weights = adapter.extract_feature_weights(proto, 4)
    assert len(weights) == 4
    np.testing.assert_allclose(sum(w for w in weights if w is not None), 1.0, rtol=1e-9, atol=1e-9)


def test_svm_rbf_kernel_gradient_identity() -> None:
    X, y = _binary_data()
    model = SVC(kernel="rbf", gamma="scale", probability=True).fit(X, y)
    proto = _convert(model, 4)
    adapter = ModelRegistry.resolve_adapter(proto, ".onnx")
    batch = adapter.predict_batch(proto, X[:4])
    assert batch.probabilities is not None
    mat = np.asarray(batch.probabilities, dtype=np.float64)
    traces = adapter.trace_internal_flow(proto, X[:4], [0], ["a", "b", "c", "d"])
    tr = traces[0]
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()),
        float(mat[0, 0]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert tr.extra_mechanics["kernel_type"] == "RBF"
    assert tr.extra_mechanics["attribution_method"] == "svm_kernel_gradient"
    assert tr.extra_mechanics["n_support_vectors"] > 0


def test_multi_input_named_tensors_end_to_end() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame({"age": X[:, 0], "city": np.where(X[:, 1] > 0, "a", "b")})
    pre = ColumnTransformer([
        ("num", StandardScaler(), ["age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200).fit(pre.fit_transform(df), y)),
    ])
    proto = convert_sklearn(
        pipe,
        initial_types=[("age", FloatTensorType([None, 1])), ("city", StringTensorType([None, 1]))],
        target_opset=12,
    )
    meta = oc.inspect_onnx_classical(proto)
    assert meta.n_features == 2
    assert meta.feature_names == ["age", "city"]
    adapter = ModelRegistry.resolve_adapter(proto, ".onnx")
    Xobj = np.empty((6, 2), dtype=object)
    Xobj[:, 0] = X[:6, 0].astype(object)
    Xobj[:, 1] = np.where(X[:6, 1] > 0, "a", "b")
    batch = adapter.predict_batch(proto, Xobj)
    assert batch.probabilities is not None
    assert np.asarray(batch.probabilities).shape == (6, 2)
    traces = adapter.trace_internal_flow(proto, Xobj, [0], ["age", "city"])
    tr = traces[0]
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()),
        float(np.asarray(batch.probabilities)[0, 0]),
        rtol=1e-4,
        atol=1e-4,
    )
    assert tr.extra_mechanics["attribution_method"] == "input_x_gradient_session"
    assert tr.extra_mechanics["non_numeric_columns"] == ["city"]


def test_zipmap_fallback_unpacker_without_bypass() -> None:
    raw = [{0: 0.2, 1: 0.8}, {0: 0.6, 1: 0.4}, {"0": 0.1, "1": 0.9}]
    mat = oc.unpack_zipmap_sequence(raw, [0, 1])
    assert mat.dtype == np.float32
    np.testing.assert_allclose(
        mat, np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]], dtype=np.float32), rtol=1e-6, atol=1e-6
    )


def test_schema_registry_and_contiguous_feed() -> None:
    X, y = _binary_data()
    model = LogisticRegression(max_iter=200).fit(X, y)
    proto = _convert(model, 4)
    UniversalSchemaDetector.validate_is_fitted(proto)  # must not raise
    schema = UniversalSchemaDetector.extract_schema(proto)
    assert schema.n_features_in == 4
    assert [f.name for f in schema.features] == ["feature_0", "feature_1", "feature_2", "feature_3"]
    assert schema.classes == [0, 1]
    assert ModelRegistry.resolve_adapter(proto, ".onnx").algorithm_key == "onnx_classical"
    from sklearn.linear_model import LinearRegression as _LR  # noqa: PLC0415

    reg = _LR().fit(X[:, :3], X[:, 0] * 2.0)
    proto_reg = _convert(reg, 3)
    assert ModelRegistry.resolve_adapter(proto_reg, ".onnx").algorithm_key == "onnx_classical_regressor"
    adapter = ModelRegistry.resolve_adapter(proto, ".onnx")
    resolved = adapter._build_optimized_session(proto)
    feed = adapter._build_feed_dict(resolved.session, X[:7, :].copy(), ["a", "b", "c", "d"])
    assert feed["float_input"].dtype == np.float32
    assert bool(feed["float_input"].flags["C_CONTIGUOUS"])


def test_surface_visual_mapping_per_family() -> None:
    X, y = _binary_data()
    tree_proto = _convert(DecisionTreeClassifier(max_depth=3).fit(X, y), 4)
    lin_proto = _convert(LogisticRegression(max_iter=200).fit(X, y), 4)
    svm_proto = _convert(SVC(kernel="rbf", probability=True).fit(X, y), 4)
    tree_mesh = ModelRegistry.resolve_adapter(tree_proto, ".onnx").compute_3d_surface(tree_proto, _spec(), 4)
    assert tree_mesh.visual_type.value == "stepped_axis_blocks"
    assert tree_mesh.extra_overlays["n_trees"] >= 1
    assert "split_thresholds" in tree_mesh.extra_overlays
    lin_mesh = ModelRegistry.resolve_adapter(lin_proto, ".onnx").compute_3d_surface(lin_proto, _spec(), 4)
    assert lin_mesh.visual_type.value == "probability_sigmoid_surface"
    assert lin_mesh.extra_overlays["post_transform"] in ("LOGISTIC", "SOFTMAX", "NONE")
    svm_mesh = ModelRegistry.resolve_adapter(svm_proto, ".onnx").compute_3d_surface(svm_proto, _spec(), 4)
    assert svm_mesh.visual_type.value == "svm_margin_tube"
    assert svm_mesh.extra_overlays["kernel_type"] == "RBF"
    for mesh in (tree_mesh, lin_mesh, svm_mesh):
        assert mesh.resolution == 20
        assert len(mesh.z_values) == 400
        assert all(0.0 <= v <= 1.0 for v in mesh.z_values)
    big = ModelRegistry.resolve_adapter(lin_proto, ".onnx").compute_3d_surface(lin_proto, _spec(resolution=200), 4)
    assert big.resolution == 100

"""Phase 39 — ONNX deep network intermediate-probe math tests.

Covers helper-built ``Gemm -> Relu -> Gemm -> Relu -> Gemm`` graphs (no
torch needed): deep-vs-classical detection, in-memory injection without
touching the source proto, dynamic/static batch parity, raw-logit
softmax wrapping, penultimate SVD manifold, >2-D pooling, probe caps,
waterfall identity, and registry/schema wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import app.domains.model_engine.adapters.neural_onnx.onnx_deep as od  # noqa: E402
from app.domains.model_engine.registry import ModelRegistry  # noqa: E402
from app.domains.model_engine.schema_detector import UniversalSchemaDetector  # noqa: E402
from app.domains.model_engine.value_objects import GridSpec3D  # noqa: E402


def _build_mlp(n_features=4, hidden=(8, 6), n_outputs=2, batch=None, seed=39):  # type: ignore[no-untyped-def]
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    inits = []
    nodes = []
    prev = "input"
    width = int(n_features)
    for i, h in enumerate(hidden):
        w = (rng.normal(0.0, 0.6, size=(width, h)) / np.sqrt(max(width, 1))).astype(np.float32)
        b = np.zeros((h,), dtype=np.float32)
        inits.append(numpy_helper.from_array(w, f"W{i}"))
        inits.append(numpy_helper.from_array(b, f"B{i}"))
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"B{i}"], [f"h{i}"]))
        nodes.append(helper.make_node("Relu", [f"h{i}"], [f"r{i}"]))
        prev = f"r{i}"
        width = int(h)
    w = (rng.normal(0.0, 0.6, size=(width, n_outputs)) / np.sqrt(max(width, 1))).astype(np.float32)
    b = np.zeros((n_outputs,), dtype=np.float32)
    inits.append(numpy_helper.from_array(w, "Wout"))
    inits.append(numpy_helper.from_array(b, "Bout"))
    nodes.append(helper.make_node("Gemm", [prev, "Wout", "Bout"], ["logits"]))
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, n_features])
    Y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, n_outputs])
    graph = helper.make_graph(nodes, "tabular_dnn", [X], [Y], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "resolution": 12,
        "fixed_feature_medians": {},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


def _data(seed=39, n=24, width=4) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)


def test_inspect_detects_deep_and_rejects_classical_overlap() -> None:
    deep = _build_mlp()
    meta = od.inspect_onnx_deep(deep)
    assert meta.n_features == 4
    assert meta.is_classifier is True
    assert meta.output_width == 2
    assert meta.classes == [0, 1]
    assert meta.penultimate_name == "r1"
    assert meta.probe_names[-1] == "r1"
    assert od.is_deep_learning_proto(deep) is True
    assert od.ONNXDeepLearningAdapter.can_handle(deep, ".onnx") is True

    from skl2onnx import convert_sklearn  # noqa: PLC0415
    from skl2onnx.common.data_types import FloatTensorType  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    X = _data()
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    classical = convert_sklearn(
        clf, initial_types=[("float_input", FloatTensorType([None, 4]))], target_opset=12
    )
    assert od.is_deep_learning_proto(classical) is False
    assert od.ONNXDeepLearningAdapter.can_handle(classical, ".onnx") is False
    from app.domains.model_engine.adapters.neural_onnx.onnx_classical import (  # noqa: PLC0415
        ONNXClassicalAdapter,
    )

    assert ONNXClassicalAdapter.can_handle(deep, ".onnx") is False
    assert ModelRegistry.resolve_adapter(deep, ".onnx").algorithm_key == "onnx_deep"


def test_injection_leaves_source_proto_untouched() -> None:
    deep = _build_mlp()
    before = deep.SerializeToString()
    n_out_before = len(list(deep.graph.output))
    adapter = od.ONNXDeepLearningAdapter()
    working, added = adapter._inject_intermediate_outputs(deep)
    assert deep.SerializeToString() == before
    assert len(list(deep.graph.output)) == n_out_before
    assert len(added) >= 2
    assert len(list(working.graph.output)) == n_out_before + len(added)


def test_dynamic_batch_parity_with_manual_forward() -> None:
    deep = _build_mlp()
    X = _data()
    adapter = ModelRegistry.resolve_adapter(deep, ".onnx")
    batch = adapter.predict_batch(deep, X[:8])
    assert batch.probabilities is not None
    mat = np.asarray(batch.probabilities, dtype=np.float64)
    assert mat.shape == (8, 2)
    np.testing.assert_allclose(mat.sum(axis=1), np.ones(8), rtol=1e-5, atol=1e-5)
    # Manual NumPy forward through the known initializers.
    from onnx import numpy_helper  # noqa: PLC0415

    inits = {i.name: numpy_helper.to_array(i) for i in deep.graph.initializer}
    h = X[:8].astype(np.float64) @ inits["W0"].astype(np.float64) + inits["B0"]
    h = np.maximum(h, 0.0)
    h = h @ inits["W1"].astype(np.float64) + inits["B1"]
    h = np.maximum(h, 0.0)
    logits = h @ inits["Wout"].astype(np.float64) + inits["Bout"]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(mat, expected, rtol=1e-5, atol=1e-5)


def test_static_batch_unrolled_matches_dynamic() -> None:
    dyn = _build_mlp(batch=None)
    static = _build_mlp(batch=1)
    meta = od.inspect_onnx_deep(static)
    assert meta.is_static_batch is True
    assert meta.input_static_batch == 1
    X = _data(n=5)
    dyn_batch = ModelRegistry.resolve_adapter(dyn, ".onnx").predict_batch(dyn, X)
    static_batch = ModelRegistry.resolve_adapter(static, ".onnx").predict_batch(static, X)
    assert static_batch.probabilities is not None and dyn_batch.probabilities is not None
    np.testing.assert_allclose(
        np.asarray(static_batch.probabilities),
        np.asarray(dyn_batch.probabilities),
        rtol=1e-5,
        atol=1e-5,
    )


def test_micro_batch_unroller_covers_static_fallback() -> None:
    """Force the §2.3 attempt-2 loop on an unrelaxed static session."""
    static = _build_mlp(batch=1, seed=43)
    raw_sess = ort.InferenceSession(
        static.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [str(o.name) for o in raw_sess.get_outputs()]
    forced = od._ResolvedDeepSession(
        session=raw_sess,
        input_name="input",
        input_static_batch=1,
        is_static_batch=True,
        output_names=names,
        final_output="logits",
        probe_names=[],
        penultimate_name=None,
        providers_used=["CPUExecutionProvider"],
        injection_active=False,
        output_width=2,
        is_classifier=True,
    )
    X = _data(n=5)
    outs = od.run_session_outputs(forced, X)
    assert outs and outs[0].shape == (5, 2)
    ref = np.concatenate(
        [np.asarray(raw_sess.run(["logits"], {"input": X[i : i + 1]})) for i in range(5)],
        axis=1,
    )
    np.testing.assert_allclose(np.asarray(outs[0]), np.asarray(ref[0]), rtol=1e-6, atol=1e-6)


def test_raw_logits_wrapped_and_waterfall_identity() -> None:
    deep = _build_mlp()
    X = _data()
    adapter = ModelRegistry.resolve_adapter(deep, ".onnx")
    batch = adapter.predict_batch(deep, X[:4])
    assert batch.probabilities is not None
    mat = np.asarray(batch.probabilities, dtype=np.float64)
    traces = adapter.trace_internal_flow(deep, X[:4], [0, 1], ["a", "b", "c", "d"])
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(mat[i, 0]),
            rtol=1e-4,
            atol=1e-4,
        )
        mech = tr.extra_mechanics
        assert mech["model_family"] == "onnx_deep"
        assert mech["attribution_method"] == "finite_difference_input_sensitivity"
        assert mech["output_transform"] == "softmax"
        assert mech["injection_active"] is True
        assert mech["provider"] == "CPUExecutionProvider"
        assert mech["batch_mode"] == "dynamic"
        assert len(mech["latent_coords_3d"]) == 3
        assert all(isinstance(v, float) for v in mech["latent_coords_3d"])
        assert mech["n_probed_layers"] >= 2
        assert mech["penultimate_layer"] == "r1"
        assert len(mech["layer_telemetry"]) == mech["n_probed_layers"]


def test_latent_pca_manifold_geometry() -> None:
    rng = np.random.default_rng(7)
    H = rng.normal(size=(16, 6)).astype(np.float32)
    adapter = od.ONNXDeepLearningAdapter()
    z = adapter.compute_latent_pca_manifold(H)
    assert z.dtype == np.float32
    assert z.shape == (16, 3)
    assert bool(np.all(np.isfinite(z)))
    # Centered projection: column means ~0 when N >> D spread.
    np.testing.assert_allclose(z.mean(axis=0), np.zeros(3), rtol=1e-3, atol=1e-3)
    # Degenerate single-row input pads instead of crashing.
    z1 = adapter.compute_latent_pca_manifold(np.ones((1, 6), dtype=np.float32))
    assert z1.shape == (1, 3) and bool(np.all(np.isfinite(z1)))


def test_intermediate_pooling_and_probe_cap() -> None:
    cube = np.ones((4, 5, 7), dtype=np.float32)
    normed = od.normalize_intermediate(cube, 4)
    assert normed.shape == (4, 5)
    np.testing.assert_allclose(normed, np.ones((4, 5), dtype=np.float32))
    deep12 = _build_mlp(hidden=tuple([4] * 12), seed=11)
    meta = od.inspect_onnx_deep(deep12, 8)
    assert len(meta.probe_names) == 8
    assert meta.probe_names[-1] == "r11"
    resolved = od.build_probed_session(deep12, 8)
    assert len(resolved.probe_names) <= 8
    assert resolved.injection_active is True


def test_regressor_end_to_end_and_schema() -> None:
    reg = _build_mlp(n_features=3, hidden=(6, 4), n_outputs=1, seed=41)
    meta = od.inspect_onnx_deep(reg)
    assert meta.is_regressor is True
    adapter = ModelRegistry.resolve_adapter(reg, ".onnx")
    assert adapter.algorithm_key == "onnx_deep_regressor"
    X = _data(seed=41, n=6, width=3)
    batch = adapter.predict_batch(reg, X)
    assert batch.probabilities is None
    assert len(batch.predictions) == 6
    traces = adapter.trace_internal_flow(reg, X, [0], ["a", "b", "c"])
    tr = traces[0]
    np.testing.assert_allclose(
        tr.base_value + sum(tr.feature_contributions.values()),
        float(batch.predictions[0]),
        rtol=1e-4,
        atol=1e-4,
    )
    assert tr.extra_mechanics["is_regressor"] is True
    UniversalSchemaDetector.validate_is_fitted(reg)
    schema = UniversalSchemaDetector.extract_schema(reg)
    assert schema.n_features_in == 3
    assert schema.classes is None
    weights = adapter.extract_feature_weights(reg, 3)
    assert len(weights) == 3 and all(w is not None for w in weights)


def test_surface_carries_latent_manifold_overlays() -> None:
    deep = _build_mlp()
    adapter = ModelRegistry.resolve_adapter(deep, ".onnx")
    mesh = adapter.compute_3d_surface(deep, _spec(), 4)
    assert mesh.visual_type.value == "neural_morphing_manifold"
    assert mesh.resolution == 12
    assert len(mesh.z_values) == 144
    assert all(0.0 <= v <= 1.0 for v in mesh.z_values)
    ov = mesh.extra_overlays
    assert ov["model_family"] == "onnx_deep"
    assert ov["batch_mode"] == "dynamic"
    assert ov["provider"] == "CPUExecutionProvider"
    assert ov["penultimate_layer"] == "r1"
    assert len(ov["latent_manifold_sample"]) > 0
    assert all(len(p) == 3 for p in ov["latent_manifold_sample"])
    big = adapter.compute_3d_surface(deep, _spec(resolution=200), 4)
    assert big.resolution == 100

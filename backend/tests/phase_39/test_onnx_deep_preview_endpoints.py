"""Phase 39 — upload / trace-preview / surface-preview REST tests (.onnx deep)."""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

client = TestClient(create_app())


def _build_mlp(n_features: int = 4, hidden: tuple[int, ...] = (8, 6), n_outputs: int = 2, batch: object = None, seed: int = 39):  # type: ignore[no-untyped-def]
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


def _data(seed: int = 39, n: int = 24, width: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)


def _upload(proto: object, name: str, expected_key: str) -> str:  # type: ignore[no-untyped-def]
    payload = proto.SerializeToString()  # type: ignore[union-attr]
    res = client.post(
        "/api/v1/models/upload",
        files={"file": (name, payload, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == expected_key, body
    assert body["visual_3d_type"] == "neural_morphing_manifold", body
    assert body["n_features_in"] == 4 or expected_key == "onnx_deep_regressor", body
    return body["model_id"]


def test_upload_rejects_non_onnx_bytes() -> None:
    res = client.post(
        "/api/v1/models/upload",
        files={"file": ("evil.onnx", b"definitely not an onnx graph", "application/octet-stream")},
    )
    assert res.status_code in (400, 415), res.text


def test_trace_preview_deep_classifier_dual_probes() -> None:
    X = _data()
    model_id = _upload(_build_mlp(), "deep_mlp.onnx", "onnx_deep")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:2].tolist(), "row_indices": [0, 1]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 2
    for tr in traces:
        assert set(tr["feature_contributions"]) == {"feature_0", "feature_1", "feature_2", "feature_3"}
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "onnx_deep"
        assert mech["domain_badge"].startswith("ONNX Deep Net")
        assert mech["n_probed_layers"] >= 2
        assert len(mech["latent_coords_3d"]) == 3
        assert len(mech["layer_telemetry"]) == mech["n_probed_layers"]
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(np.asarray(mech["probabilities"])[0]),
            rtol=1e-4,
            atol=1e-4,
        )


def test_trace_preview_static_batch_model() -> None:
    X = _data(n=4)
    model_id = _upload(_build_mlp(batch=1, seed=43), "deep_static.onnx", "onnx_deep")
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X.tolist(), "row_indices": [0, 1, 2, 3]},
    )
    assert res.status_code == 200, res.text
    traces = res.json()["traces"]
    assert len(traces) == 4
    # Pure Gemm graphs accept the in-memory dim_param relaxation (§2.3
    # attempt 1 → dynamic); Reshape-pinned graphs fall to the micro-batch
    # unroller (§2.3 attempt 2 → static_unrolled). Either way predictions
    # must be valid probabilities with an intact waterfall identity.
    assert traces[0]["extra_mechanics"]["batch_mode"] in ("dynamic", "static_unrolled")
    for tr in traces:
        mech = tr["extra_mechanics"]
        assert mech["model_family"] == "onnx_deep"
        np.testing.assert_allclose(
            tr["base_value"] + sum(tr["feature_contributions"].values()),
            float(np.asarray(mech["probabilities"])[0]),
            rtol=1e-4,
            atol=1e-4,
        )


def test_surface_preview_deep_latent_overlays() -> None:
    model_id = _upload(_build_mlp(), "deep_surface.onnx", "onnx_deep")
    res = client.post(
        f"/api/v1/models/{model_id}/surface-preview",
        json={
            "axis_x_feature_index": 0,
            "axis_y_feature_index": 1,
            "x_min": -2.0,
            "x_max": 2.0,
            "y_min": -2.0,
            "y_max": 2.0,
            "resolution": 12,
            "fixed_feature_medians": {},
            "target_idx": 1,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["visual_type"] == "neural_morphing_manifold"
    assert body["resolution"] == 12
    assert len(body["z_values"]) == 144
    assert body["extra_overlays"]["model_family"] == "onnx_deep"
    assert body["extra_overlays"]["penultimate_layer"] == "r1"
    assert len(body["extra_overlays"]["latent_manifold_sample"]) > 0
    assert body["telemetry"]["grid_points"] == 144


def test_upload_deep_regressor_schema() -> None:
    reg = _build_mlp(n_features=3, hidden=(6, 4), n_outputs=1, seed=41)
    payload = reg.SerializeToString()
    res = client.post(
        "/api/v1/models/upload",
        files={"file": ("deep_reg.onnx", payload, "application/octet-stream")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["algorithm_key"] == "onnx_deep_regressor", body
    assert body["n_features_in"] == 3, body
    model_id = body["model_id"]
    X = _data(seed=41, n=2, width=3)
    res = client.post(
        f"/api/v1/models/{model_id}/trace-preview",
        json={"rows": X[:1].tolist(), "row_indices": [0]},
    )
    assert res.status_code == 200, res.text
    tr = res.json()["traces"][0]
    assert tr["extra_mechanics"]["is_regressor"] is True

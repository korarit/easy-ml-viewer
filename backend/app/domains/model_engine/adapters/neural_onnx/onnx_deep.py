# backend/app/domains/model_engine/adapters/neural_onnx/onnx_deep.py
"""Phase 39 — ONNX Deep Neural Network intermediate layer extractor.

Covers ``.onnx`` tabular deep-learning graphs exported from PyTorch /
TensorFlow / Keras / JAX (no ``ai.onnx.ml`` nodes) built from deep
operators (``Gemm`` / ``MatMul`` + ``Relu`` / ``Gelu`` / ``Silu`` /
``Sigmoid`` / ``Tanh`` / ``BatchNormalization`` / ``LayerNormalization``
/ ``Attention`` / ``Add`` / ``Softmax``).

Blueprint mapping (phase-39-onnx-deep-learning-intermediate.md):

- §2.1 in-memory ONNX graph output injection — clone the ``ModelProto``,
  run ``onnx.shape_inference`` (guarded ``try/except`` for custom ops),
  append ``make_empty_tensor_value_info`` probes for up to
  ``MAX_PROBED_LAYERS = 8`` hidden activations, then open a single
  ``ort.InferenceSession`` on the serialized bytes so one ``sess.run()``
  returns the final output plus ``{H^(1) .. H^(L-1)}`` in a single C++
  pass. The source ``.onnx`` file on disk is never modified.
- §2.2 3D latent-space PCA manifold — the penultimate-layer activation
  ``H in R^{N x D}`` is centered and projected with a truncated SVD to
  ``Z_latent in R^{N x 3}`` (float32). >2-D intermediate tensors are
  reduced with global-average-pooling (``mean`` over trailing axes) or
  flattened to ``(N, D)`` first (edge #3).
- §2.3 static ``batch_size=1`` guard — PyTorch exports without
  ``dynamic_axes`` lock ``input.shape`` to ``[1, F]``. The adapter first
  tries an in-memory ``dim_param = "batch_size"`` relaxation; when the
  graph still reports a fixed batch (e.g. a ``Reshape`` bakes the ``1``
  in), it falls back to a pre-allocated micro-batch loop of
  ``batch_size = dim_0`` rows and concatenates the results.
- §6 memory guards — probe cap (8), immediate Top-32 neuron downsampling
  per probed layer (float32) + ``del outputs`` right after each chunk,
  ``<= 2_500``-row inference chunks, ``ORT_SEQUENTIAL`` with
  ``intra_op_num_threads = max(1, cores - 1)`` and
  ``arena_extend_strategy = "kSameAsRequested"``.
- §7 edge #2 raw-logits auto-transform — classifier graphs ending in a
  bare ``Linear`` (``CrossEntropyLoss`` training) get an automatic
  ``softmax`` (``C > 1``) / ``expit`` (``C == 1``) wrap when the session
  output is not already a valid probability distribution.
- §3 attribution — finite-difference input sensitivity
  (``Input x Gradient`` via session-backed central differences) with the
  waterfall identity ``base + sum(contribs) == prediction`` by
  construction (``base = pred - sum(attr)``).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.domains.model_engine.base_adapter import BaseModelAdapter
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import (
    GridSpec3D,
    InternalFlowRowTrace,
    PredictionBatchResult,
    SurfaceMesh3DData,
    TaskType,
    Visual3DType,
)

PREDICT_CHUNK_SIZE = 2_500
MAX_SURFACE_RESOLUTION = 100
FINITE_DIFF_EPS = 1e-4

_DEEP_GEMM_OPS = frozenset({"Gemm", "MatMul"})
_DEEP_ACTIVATION_OPS = frozenset({
    "Relu", "Gelu", "Silu", "Sigmoid", "Tanh", "LeakyRelu", "Elu",
    "Selu", "Softplus", "Mish", "Swish", "HardSigmoid", "Shrink",
})
_DEEP_NORM_OPS = frozenset({
    "BatchNormalization", "LayerNormalization", "InstanceNormalization",
    "Dropout", "Attention", "Add", "Softmax",
})
_DEEP_OPS = _DEEP_GEMM_OPS | _DEEP_ACTIVATION_OPS | _DEEP_NORM_OPS


def is_onnx_model_proto(obj: Any) -> bool:
    """Duck-typed check for an ``onnx.ModelProto`` (no hard onnx import)."""
    graph = getattr(obj, "graph", None)
    return graph is not None and hasattr(graph, "node") and hasattr(graph, "input")


def _load_onnx_module() -> Any:
    try:
        import onnx  # noqa: PLC0415

        return onnx
    except ImportError as exc:
        raise ValueError(
            "ONNX support requires the 'onnx' package (pip install onnx onnxruntime)"
        ) from exc


def _load_ort_module() -> Any:
    try:
        import onnxruntime as ort  # noqa: PLC0415

        return ort
    except ImportError as exc:
        raise ValueError(
            "ONNX inference requires the 'onnxruntime' package (pip install onnxruntime)"
        ) from exc


def _graph_op_types(proto: Any) -> list[str]:
    return [str(n.op_type) for n in list(proto.graph.node)]


def _has_classical_ml_nodes(proto: Any) -> bool:
    for node in list(proto.graph.node):
        if str(getattr(node, "domain", "")) == "ai.onnx.ml":
            return True
    return False


def is_deep_learning_proto(proto: Any) -> bool:
    """True for deep tabular nets: no ``ai.onnx.ml``, has Gemm/MatMul + depth."""
    try:
        if not is_onnx_model_proto(proto):
            return False
        if _has_classical_ml_nodes(proto):
            return False
        ops = _graph_op_types(proto)
        if not any(o in _DEEP_GEMM_OPS for o in ops):
            return False
        deep_marks = [o for o in ops if o in _DEEP_OPS]
        return len(deep_marks) >= 2
    except (AttributeError, TypeError):
        return False


def _node_chain_summary(proto: Any) -> list[str]:
    chain: list[str] = []
    for node in list(proto.graph.node):
        domain = str(getattr(node, "domain", "") or "ai.onnx")
        chain.append(f"{domain}::{node.op_type}")
    return chain


def _input_signature(proto: Any) -> list[dict[str, Any]]:
    sig: list[dict[str, Any]] = []
    for inp in list(proto.graph.input):
        shape: list[Any] = []
        try:
            dims = inp.type.tensor_type.shape.dim
            for d in dims:
                param = str(getattr(d, "dim_param", "") or "")
                if param:
                    shape.append(None)
                    continue
                v = int(getattr(d, "dim_value", 0) or 0)
                shape.append(None if v == 0 else v)
        except (AttributeError, TypeError, ValueError):
            shape = []
        try:
            from onnx import TensorProto  # noqa: PLC0415

            elem = TensorProto.DataType.Name(int(inp.type.tensor_type.elem_type))
            dtype = f"tensor({elem.lower()})"
        except (ImportError, AttributeError, TypeError, ValueError):
            dtype = "tensor(float)"
        try:
            name = str(inp.name)
        except (AttributeError, TypeError):
            name = "input"
        sig.append({"name": name, "shape": shape, "dtype": dtype})
    return sig


def _output_signature(proto: Any) -> list[dict[str, Any]]:
    sig: list[dict[str, Any]] = []
    for out in list(proto.graph.output):
        try:
            which = out.type.WhichOneof("value") if hasattr(out.type, "WhichOneof") else None
        except (AttributeError, TypeError):
            which = None
        try:
            name = str(out.name)
        except (AttributeError, TypeError):
            name = "output"
        sig.append({"name": name, "kind": str(which or "unknown")})
    return sig


def _preferred_providers(ort: Any) -> list[str]:
    try:
        available = list(ort.get_available_providers())
    except (AttributeError, RuntimeError):
        available = ["CPUExecutionProvider"]
    ordered = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    picked = [p for p in ordered if p in available]
    return picked or ["CPUExecutionProvider"]


def _session_options(ort: Any) -> Any:
    opts = ort.SessionOptions()
    try:
        cores = int(os.cpu_count() or 2)
    except (TypeError, ValueError):
        cores = 2
    try:
        opts.intra_op_num_threads = max(1, cores - 1)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        opts.add_session_config_entry("arena_extend_strategy", "kSameAsRequested")
    except (AttributeError, TypeError, ValueError):
        pass
    return opts


def _stable_sigmoid(z: np.ndarray | float) -> Any:
    arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    e = np.exp(arr[~pos])
    out[~pos] = e / (1.0 + e)
    return out if out.shape != () else float(out)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)


def _looks_like_probas(mat: np.ndarray) -> bool:
    try:
        m = np.asarray(mat, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if m.size == 0:
        return False
    if bool((m < -1e-6).any()) or bool((m > 1.0 + 1e-6).any()):
        return False
    if m.ndim == 2 and m.shape[1] > 1:
        sums = np.sum(m, axis=1)
        return bool(np.all(np.abs(sums - 1.0) < 5e-3))
    if m.ndim == 2 and m.shape[1] == 1:
        return bool((m >= 0.0).all() and (m <= 1.0).all())
    return False


def apply_output_transform(raw: np.ndarray, is_classifier: bool) -> tuple[np.ndarray, str]:
    """Wrap raw logits into probabilities for classifiers (edge #2)."""
    mat = np.asarray(raw, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if mat.ndim > 2:
        mat = mat.reshape(int(mat.shape[0]), -1)
    if not is_classifier:
        return np.ascontiguousarray(mat, dtype=np.float32), "identity"
    if _looks_like_probas(mat):
        return np.ascontiguousarray(mat, dtype=np.float32), "identity"
    if mat.shape[1] == 1:
        p = np.asarray(_stable_sigmoid(mat.ravel()), dtype=np.float64).reshape(-1, 1)
        return np.ascontiguousarray(p, dtype=np.float32), "sigmoid"
    p = np.asarray(_stable_softmax(np.asarray(mat, dtype=np.float64)), dtype=np.float32)
    return np.ascontiguousarray(p, dtype=np.float32), "softmax"


def normalize_intermediate(tensor: Any, n_rows: int) -> np.ndarray:
    """Reduce an intermediate tensor to ``(N, D)`` float32 (edge #3)."""
    arr = np.asarray(tensor)
    if arr.shape == ():
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        # Either (N,) single-neuron rows or a stray scalar vector.
        if int(arr.shape[0]) == int(n_rows):
            arr = arr.reshape(n_rows, 1)
        else:
            arr = arr.reshape(1, -1)
    while arr.ndim > 2:
        # Global average pooling over the trailing (sequence/spatial) axis.
        with np.errstate(all="ignore"):
            arr = np.mean(arr, axis=-1)
    if arr.ndim != 2:
        arr = np.asarray(arr).reshape(int(n_rows), -1)
    if int(arr.shape[0]) != int(n_rows):
        # Batch-static graphs may return one row per micro-batch; tile/trim.
        flat = np.asarray(arr).reshape(-1)
        need = int(n_rows) * max(int(arr.shape[1]) if arr.ndim == 2 else 1, 1)
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size))
        arr = flat[:need].reshape(int(n_rows), -1)
    with np.errstate(all="ignore"):
        clean = np.where(np.isfinite(np.asarray(arr, dtype=np.float64)), arr, 0.0)
    return np.ascontiguousarray(np.asarray(clean, dtype=np.float32))


def topk_neuron_indices(layer: np.ndarray, k: int = 32) -> list[int]:
    """Top-K neuron columns by mean |activation| energy (§6 downsampling)."""
    mat = np.asarray(layer, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] <= k:
        return list(range(int(mat.shape[1]) if mat.ndim == 2 else 0))
    with np.errstate(all="ignore"):
        energy = np.mean(np.abs(np.where(np.isfinite(mat), mat, 0.0)), axis=0)
    order = np.argsort(energy, kind="stable")[::-1][:k]
    return sorted(int(v) for v in order.tolist())


def layer_telemetry(name: str, layer: np.ndarray, top_k: int = 32) -> dict[str, Any]:
    mat = np.asarray(layer, dtype=np.float64)
    if mat.ndim != 2:
        mat = np.asarray(layer).reshape(int(mat.shape[0]), -1)
    with np.errstate(all="ignore"):
        finite = np.where(np.isfinite(mat), mat, 0.0)
    d = int(mat.shape[1]) if mat.ndim == 2 else 0
    mean_act = float(np.mean(finite)) if finite.size else 0.0
    max_act = float(np.max(finite)) if finite.size else 0.0
    sparsity = float(np.mean(finite <= 0.0)) if finite.size else 0.0
    kept = topk_neuron_indices(np.asarray(finite, dtype=np.float32), top_k)
    kept_vals = finite[:, kept] if kept and finite.size else np.zeros((finite.shape[0], 0))
    return {
        "layer_name": str(name),
        "width": int(d),
        "rendered_neurons": int(len(kept)),
        "mean_activation": round(float(mean_act) if np.isfinite(mean_act) else 0.0, 5),
        "max_activation": round(float(max_act) if np.isfinite(max_act) else 0.0, 5),
        "sparsity": round(float(sparsity) if np.isfinite(sparsity) else 0.0, 5),
        "rendered_indices": [int(v) for v in kept],
        "rendered_means": [round(float(v), 5) for v in (np.mean(kept_vals, axis=0).tolist() if kept_vals.size else [])],
    }


def truncated_svd_latent(h: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, list[float], float]:
    """Centered truncated-SVD projection ``(N, D) -> (N, 3)`` (§2.2)."""
    mat = np.asarray(h, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if mat.ndim != 2:
        mat = np.asarray(mat).reshape(int(mat.shape[0]), -1)
    mat = np.where(np.isfinite(mat), mat, 0.0)
    n, d = int(mat.shape[0]), int(mat.shape[1])
    k = max(min(int(n_components), n, d, 3), 1)
    if n == 0 or d == 0:
        return np.zeros((n, 3), dtype=np.float32), [0.0, 0.0, 0.0], 0.0
    centered = mat - np.mean(mat, axis=0, keepdims=True)
    if not bool(np.any(centered)) or n < 2:
        # Degenerate (single row / constant) — fall back to raw padding.
        z = np.zeros((n, 3), dtype=np.float64)
        width = min(d, 3, n and d)
        z[:, :width] = centered[:, :width] if d >= 1 else 0.0
        return np.ascontiguousarray(z, dtype=np.float32), [0.0, 0.0, 0.0], 0.0
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        z = np.zeros((n, 3), dtype=np.float64)
        width = min(d, 3)
        z[:, :width] = centered[:, :width]
        return np.ascontiguousarray(z, dtype=np.float32), [0.0, 0.0, 0.0], 0.0
    comps = np.asarray(vt[:k], dtype=np.float64).T  # (D, k)
    z = np.asarray(centered @ comps, dtype=np.float64)
    if k < 3:
        z = np.pad(z, ((0, 0), (0, 3 - k)))
    total = float(np.sum(s * s))
    ratios = [(float(v * v / total) if total > 0 else 0.0) for v in s[:3]]
    while len(ratios) < 3:
        ratios.append(0.0)
    explained = float(sum(ratios[:3])) if total > 0 else 0.0
    return np.ascontiguousarray(np.asarray(z[:, :3], dtype=np.float32)), [float(v) for v in ratios], float(explained)


@dataclass
class ONNXDeepMetadata:
    """Decompiled deep-graph summary (JSON-serializable via to_dict)."""

    architecture: str = "tabular_dnn"
    is_classifier: bool = True
    is_regressor: bool = False
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    input_name: str = "input"
    input_static_batch: int | None = None
    is_static_batch: bool = False
    output_width: int = 1
    classes: list[Any] = field(default_factory=list)
    input_signature: list[dict[str, Any]] = field(default_factory=list)
    output_signature: list[dict[str, Any]] = field(default_factory=list)
    node_chain: list[str] = field(default_factory=list)
    probe_names: list[str] = field(default_factory=list)
    penultimate_name: str | None = None
    final_output_name: str | None = None
    n_hidden_blocks: int = 0
    opset_version: int = 0
    provider_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "is_classifier": bool(self.is_classifier),
            "is_regressor": bool(self.is_regressor),
            "n_features": int(self.n_features),
            "feature_names": list(self.feature_names),
            "input_name": self.input_name,
            "input_static_batch": self.input_static_batch,
            "is_static_batch": bool(self.is_static_batch),
            "output_width": int(self.output_width),
            "classes": list(self.classes),
            "input_signature": [dict(s) for s in self.input_signature],
            "output_signature": [dict(s) for s in self.output_signature],
            "node_chain": list(self.node_chain),
            "probe_names": list(self.probe_names),
            "penultimate_name": self.penultimate_name,
            "final_output_name": self.final_output_name,
            "n_hidden_blocks": int(self.n_hidden_blocks),
            "opset_version": int(self.opset_version),
        }


def _candidate_probe_names(proto: Any) -> list[str]:
    """Hidden-block endpoints in graph order (activations/norms, then Gemm)."""
    ordered: list[str] = []
    try:
        final_outs = {str(o.name) for o in list(proto.graph.output)}
    except (AttributeError, TypeError):
        final_outs = set()
    for node in list(proto.graph.node):
        try:
            op = str(node.op_type)
            outs = [str(o) for o in list(node.output) if str(o)]
        except (AttributeError, TypeError):
            continue
        if not outs:
            continue
        name = outs[0]
        if name in final_outs or name in ordered:
            continue
        # Graph order (Gemm -> Relu -> ... -> penultimate Relu) so the last
        # tap is always the penultimate representation before the output
        # linear layer. Gemm/MatMul taps expose pre-activation blocks;
        # activation/norm taps expose post-activation blocks.
        if op in (_DEEP_ACTIVATION_OPS | _DEEP_NORM_OPS | _DEEP_GEMM_OPS):
            ordered.append(name)
    return ordered


def _evenly_capped(names: list[str], cap: int) -> list[str]:
    if len(names) <= cap:
        return list(names)
    # Evenly spaced taps from input side to penultimate (always kept).
    idx = np.linspace(0, len(names) - 1, cap)
    picked: list[str] = []
    for v in idx:
        name = names[int(round(float(v)))]
        if name not in picked:
            picked.append(name)
    if names and names[-1] not in picked:
        picked[-1] = names[-1]
    return picked[:cap]


def inspect_onnx_deep(proto: Any, max_probed_layers: int = 8) -> ONNXDeepMetadata:
    """Scan a deep graph for probes, batch mode and output width (§2.1)."""
    meta = ONNXDeepMetadata()
    meta.input_signature = _input_signature(proto)
    meta.output_signature = _output_signature(proto)
    meta.node_chain = _node_chain_summary(proto)
    try:
        for op in list(proto.opset_import):
            if str(getattr(op, "domain", "") or "") in ("", "ai.onnx"):
                meta.opset_version = int(getattr(op, "version", 0) or 0)
                break
    except (AttributeError, TypeError, ValueError):
        pass
    if meta.input_signature:
        first = meta.input_signature[0]
        meta.input_name = str(first.get("name", "input"))
        shape = first.get("shape") or []
        width = 0
        if len(shape) >= 2 and isinstance(shape[1], int):
            width = int(shape[1])
        meta.n_features = width
        meta.feature_names = [f"feature_{i}" for i in range(max(width, 0))]
        if shape and isinstance(shape[0], int) and int(shape[0]) > 0:
            meta.input_static_batch = int(shape[0])
            meta.is_static_batch = True
    # Output width via shape inference when available, else value_info dim.
    width_out = 0
    try:
        for out in list(proto.graph.output):
            dims = out.type.tensor_type.shape.dim
            if len(dims) >= 2:
                v = int(getattr(dims[1], "dim_value", 0) or 0)
                if v > 0:
                    width_out = max(width_out, v)
    except (AttributeError, TypeError, ValueError):
        pass
    if width_out < 1:
        # Unknown until runtime — assume binary classifier (width 2) so the
        # session pass below can correct it from the real output shape.
        width_out = 2
    meta.output_width = int(width_out)
    meta.is_regressor = bool(width_out == 1)
    meta.is_classifier = not meta.is_regressor
    meta.classes = [] if meta.is_regressor else list(range(int(width_out)))
    try:
        meta.final_output_name = str(list(proto.graph.output)[-1].name)
    except (AttributeError, TypeError, IndexError):
        meta.final_output_name = None
    candidates = _candidate_probe_names(proto)
    meta.probe_names = _evenly_capped(candidates, max(1, int(max_probed_layers)))
    meta.penultimate_name = meta.probe_names[-1] if meta.probe_names else None
    meta.n_hidden_blocks = len(meta.probe_names)
    return meta


@dataclass
class _ResolvedDeepSession:
    session: Any
    input_name: str
    input_static_batch: int | None
    is_static_batch: bool
    output_names: list[str]
    final_output: str | None
    probe_names: list[str]
    penultimate_name: str | None
    providers_used: list[str]
    injection_active: bool
    output_width: int
    is_classifier: bool
    output_transform: str = "identity"


def _try_relax_static_batch(working: Any, input_name: str) -> None:
    """In-memory ``dim_param = "batch_size"`` relaxation (§2.3 attempt 1)."""
    try:
        for inp in list(working.graph.input):
            if str(inp.name) != str(input_name):
                continue
            dims = inp.type.tensor_type.shape.dim
            if len(dims) >= 1:
                try:
                    dims[0].dim_param = "batch_size"
                except (AttributeError, TypeError, ValueError):
                    pass
        for info in list(working.graph.value_info):
            if str(info.name) != str(input_name):
                continue
            try:
                dims = info.type.tensor_type.shape.dim
                if len(dims) >= 1:
                    dims[0].dim_param = "batch_size"
            except (AttributeError, TypeError, ValueError):
                pass
    except (AttributeError, TypeError):
        pass


def build_probed_session(proto: Any, max_probed_layers: int = 8) -> _ResolvedDeepSession:
    """Clone the graph in-memory, inject probes, open the ORT session."""
    onnx = _load_onnx_module()
    ort = _load_ort_module()
    try:
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(proto))
    except Exception:
        # Edge #4: custom ops break shape inference — probe with empty
        # value infos and let onnxruntime resolve shapes at run time.
        inferred = copy.deepcopy(proto)
    meta = inspect_onnx_deep(inferred, max_probed_layers)
    working = copy.deepcopy(inferred)
    try:
        existing = {str(o.name) for o in list(working.graph.output)}
    except (AttributeError, TypeError):
        existing = set()
    injection = False
    try:
        for name in meta.probe_names:
            if name in existing:
                continue
            working.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
            existing.add(name)
            injection = True
    except (AttributeError, TypeError, ValueError):
        injection = False
    if meta.is_static_batch:
        _try_relax_static_batch(working, meta.input_name)
    try:
        # Newer ``onnx`` releases stamp IR 14+ which older ORT builds
        # reject — the probed Gemm/Relu graphs run fine on IR 10.
        if int(getattr(working, "ir_version", 0) or 0) > 10:
            working.ir_version = 10
    except (AttributeError, TypeError, ValueError):
        pass
    sess = ort.InferenceSession(
        working.SerializeToString(),
        sess_options=_session_options(ort),
        providers=_preferred_providers(ort),
    )
    try:
        providers_used = [str(p) for p in sess.get_providers()]
    except (AttributeError, RuntimeError):
        providers_used = ["CPUExecutionProvider"]
    try:
        out_names = [str(o.name) for o in sess.get_outputs()]
    except (AttributeError, RuntimeError):
        out_names = []
    try:
        in_shape = list(sess.get_inputs())[0].shape if sess.get_inputs() else []
    except (AttributeError, RuntimeError, IndexError):
        in_shape = []
    static_batch: int | None = None
    if in_shape and isinstance(in_shape[0], int) and int(in_shape[0]) > 0:
        static_batch = int(in_shape[0])
    final_out = meta.final_output_name if meta.final_output_name in out_names else (out_names[0] if out_names else None)
    probes_live = [n for n in meta.probe_names if n in out_names and n != final_out]
    return _ResolvedDeepSession(
        session=sess,
        input_name=str(list(sess.get_inputs())[0].name) if sess.get_inputs() else meta.input_name,
        input_static_batch=static_batch,
        is_static_batch=bool(static_batch is not None),
        output_names=list(out_names),
        final_output=final_out,
        probe_names=probes_live,
        penultimate_name=(probes_live[-1] if probes_live else None),
        providers_used=providers_used,
        injection_active=bool(injection and probes_live),
        output_width=int(meta.output_width),
        is_classifier=bool(meta.is_classifier),
    )


def _feed_for_input(session: Any, input_name: str, X_chunk: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(X_chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    with np.errstate(all="ignore"):
        arr = np.where(np.isfinite(arr.astype(np.float64)), arr, 0.0).astype(np.float32)
    feed = {str(input_name): np.ascontiguousarray(arr)}
    # Multi-input deep graphs (rare): broadcast the same chunk positionally.
    try:
        inputs = list(session.get_inputs())
    except (AttributeError, RuntimeError):
        inputs = []
    if len(inputs) > 1:
        out: dict[str, np.ndarray] = {}
        for j, spec in enumerate(inputs):
            col = arr[:, j : j + 1] if j < arr.shape[1] else np.zeros((arr.shape[0], 1), dtype=np.float32)
            out[str(spec.name)] = np.ascontiguousarray(col.astype(np.float32))
        return out
    return feed


def run_session_outputs(resolved: _ResolvedDeepSession, X_chunk: np.ndarray) -> list[np.ndarray]:
    """Full-batch when dynamic, micro-batch loop when static (§2.3)."""
    sess = resolved.session
    arr = np.asarray(X_chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n = int(arr.shape[0])
    if n == 0:
        return []
    names = list(resolved.output_names)
    if not resolved.is_static_batch or (resolved.input_static_batch or n) >= n:
        # Dynamic batch — or a chunk that already fits the static window.
        if resolved.is_static_batch and n != int(resolved.input_static_batch or n):
            pass  # fall through to the micro-batch unroller below
        else:
            feed = _feed_for_input(sess, resolved.input_name, arr)
            return [np.asarray(o) for o in sess.run(names, feed)]
    step = int(resolved.input_static_batch or 1)
    parts: list[list[np.ndarray]] = []
    for start in range(0, n, step):
        sub = arr[start : start + step]
        if int(sub.shape[0]) != step:
            pad = np.zeros((step - int(sub.shape[0]), sub.shape[1]), dtype=np.float32)
            sub = np.concatenate([sub, pad], axis=0)
        feed = _feed_for_input(sess, resolved.input_name, sub)
        outs = [np.asarray(o) for o in sess.run(names, feed)]
        trimmed = [o[: int(min(step, n - start))] if int(o.shape[0]) == step else o for o in outs]
        parts.append(trimmed)
    combined: list[np.ndarray] = []
    for k in range(len(names)):
        combined.append(np.concatenate([p[k] for p in parts], axis=0)[:n])
    return combined


def _first_layer_feature_scores(proto: Any, width: int) -> list[float | None] | None:
    """Mean |W| of the first Gemm/MatMul weight initializer (MLP-style)."""
    try:
        inits = {str(i.name): np.asarray(i).view(np.float32) if False else i for i in list(proto.graph.initializer)}
    except (AttributeError, TypeError):
        return None
    _ = inits
    for node in list(proto.graph.node):
        try:
            if str(node.op_type) not in _DEEP_GEMM_OPS or len(list(node.input)) < 2:
                continue
            w_name = str(list(node.input)[1])
        except (AttributeError, TypeError, IndexError):
            continue
        w: np.ndarray | None = None
        for init in list(proto.graph.initializer):
            try:
                if str(init.name) == w_name:
                    from onnx import numpy_helper  # noqa: PLC0415

                    w = np.asarray(numpy_helper.to_array(init), dtype=np.float64)
                    break
            except (AttributeError, TypeError, ValueError):
                continue
        if w is None or w.ndim != 2:
            continue
        # Gemm B is (K, M): rows == input features. MatMul may be either
        # orientation — pick the axis matching the model width.
        if int(w.shape[0]) == int(width):
            scores = np.mean(np.abs(w), axis=1)
        elif int(w.shape[1]) == int(width):
            scores = np.mean(np.abs(w), axis=0)
        else:
            continue
        if scores.size != width:
            continue
        return [float(v) if np.isfinite(v) else 0.0 for v in scores.tolist()]
    return None


class _ONNXDeepBase(BaseModelAdapter):
    is_classifier: bool = True
    supports_nan_input: bool = False

    MAX_PROBED_LAYERS: int = 8
    MAX_NEURONS_PER_PROBED_LAYER: int = 32

    def __init__(
        self,
        model_path: Path | str | None = None,
        onnx_proto: Any = None,
        providers: list[str] | None = None,
    ) -> None:
        # Spec-style construction ``ONNXDeepLearningAdapter(model_path=...)``
        # binds the artifact; contract-style use leaves both None.
        self._bound_path = Path(model_path) if model_path is not None else None
        self._bound_proto = onnx_proto
        self._bound_providers = list(providers) if providers else None
        self._bound_session: _ResolvedDeepSession | None = None

    # -- resolution ------------------------------------------------------
    def _resolve_proto(self, model_obj: Any | None) -> Any:
        target = model_obj if model_obj is not None else self._bound_proto
        if target is None and self._bound_path is not None:
            onnx = _load_onnx_module()
            target = onnx.load(str(self._bound_path))
        if target is None:
            raise ValueError("ONNXDeepLearningAdapter requires an onnx.ModelProto (or model_path)")
        if not is_onnx_model_proto(target):
            raise ValueError(f"Estimator '{type(target).__name__}' is not an ONNX model")
        if not is_deep_learning_proto(target):
            raise ValueError("ONNX graph is not a deep neural network (no Gemm/MatMul depth)")
        return target

    def _inspect_deep_operators(self, proto: Any) -> ONNXDeepMetadata:
        """Scan ``graph.node`` for deep Gemm/activation/norm blocks (§2.1)."""
        return inspect_onnx_deep(proto, self.MAX_PROBED_LAYERS)

    def _inject_intermediate_outputs(self, proto: Any) -> tuple[Any, list[str]]:
        """Clone ModelProto in-memory and append hidden activation outputs."""
        meta = self._inspect_deep_operators(proto)
        working = copy.deepcopy(proto)
        try:
            existing = {str(o.name) for o in list(working.graph.output)}
        except (AttributeError, TypeError):
            existing = set()
        added: list[str] = []
        onnx = _load_onnx_module()
        for name in meta.probe_names:
            if name in existing:
                continue
            try:
                working.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
                existing.add(name)
                added.append(name)
            except (AttributeError, TypeError, ValueError):
                continue
        return working, added

    def _build_probed_session(self, proto: Any) -> _ResolvedDeepSession:
        """Probed session with provider auto-selection + batch guard (§2.3)."""
        resolved = build_probed_session(proto, self.MAX_PROBED_LAYERS)
        if self._bound_providers:
            # Spec accepts an explicit provider list — re-open only when the
            # requested providers are actually available.
            try:
                ort = _load_ort_module()
                available = set(ort.get_available_providers())
                wanted = [p for p in self._bound_providers if p in available]
                if wanted:
                    sess = ort.InferenceSession(
                        resolved.session._model_bytes  # type: ignore[attr-defined]
                        if hasattr(resolved.session, "_model_bytes")
                        else copy.deepcopy(proto).SerializeToString(),
                        sess_options=_session_options(ort),
                        providers=wanted,
                    )
                    resolved.providers_used = [str(p) for p in sess.get_providers()]
                    resolved.session = sess
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
        return resolved

    def _bound_resolved(self, X_chunk: np.ndarray | None = None) -> _ResolvedDeepSession:
        _ = X_chunk
        if self._bound_session is None:
            proto = self._resolve_proto(None)
            self._bound_session = self._build_probed_session(proto)
        return self._bound_session

    def _run_session_with_batch_guard(self, X_chunk: np.ndarray) -> list[np.ndarray]:
        """Execute sess.run() in full batch, or micro-batch loop if static."""
        resolved = self._bound_resolved(X_chunk)
        return run_session_outputs(resolved, np.asarray(X_chunk))

    def compute_latent_pca_manifold(self, penultimate_activations: np.ndarray) -> np.ndarray:
        """Project (N, D_latent) activations into (N, 3) float32 via SVD."""
        mat = normalize_intermediate(np.asarray(penultimate_activations), int(np.asarray(penultimate_activations).shape[0]))
        z, _, _ = truncated_svd_latent(mat, 3)
        return np.ascontiguousarray(np.asarray(z, dtype=np.float32))

    def _infer_chunks(
        self, proto: Any, X: Any, n_features: int
    ) -> tuple[np.ndarray, np.ndarray, _ResolvedDeepSession, list[np.ndarray], dict[str, Any]]:
        """Chunked inference → ``(value_matrix, latent3d, session, probes, info)``.

        ``value_matrix`` is ``(N, C)`` probabilities (classifiers, raw
        logits auto-wrapped) or ``(N, 1)`` raw values (regressors).
        Intermediate probes are Top-32 downsampled per chunk and released
        immediately (``del outputs``) per §6.
        """
        resolved = self._build_probed_session(proto)
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            arr = np.asarray(arr).reshape(int(arr.shape[0]), -1)
        n = int(arr.shape[0])
        width = max(int(n_features), int(arr.shape[1]) if arr.ndim == 2 else 0, 1)
        if int(arr.shape[1]) != width:
            fixed = np.zeros((n, width), dtype=np.float32)
            keep = min(int(arr.shape[1]), width)
            fixed[:, :keep] = np.asarray(arr[:, :keep], dtype=np.float32)
            arr = fixed
        out_w = max(int(resolved.output_width), 1)
        if n == 0:
            z0 = np.zeros((0, 3), dtype=np.float32)
            return np.zeros((0, out_w), dtype=np.float32), z0, resolved, [], {"transform": "identity", "penultimate": None}
        value_parts: list[np.ndarray] = []
        penultimate_parts: list[np.ndarray] = []
        probe_accum: dict[str, list[np.ndarray]] = {name: [] for name in resolved.probe_names}
        transform = "identity"
        for start in range(0, n, PREDICT_CHUNK_SIZE):
            chunk = np.ascontiguousarray(arr[start : start + PREDICT_CHUNK_SIZE], dtype=np.float32)
            m = int(chunk.shape[0])
            outputs = run_session_outputs(resolved, chunk)
            by_name = {nm: o for nm, o in zip(resolved.output_names, outputs)}
            raw_final = by_name.get(resolved.final_output or "", None)
            if raw_final is None and outputs:
                raw_final = outputs[0]
            mat_raw = np.asarray(raw_final, dtype=np.float32).reshape(m, -1)
            mat, transform = apply_output_transform(mat_raw, resolved.is_classifier)
            if mat.shape[1] != out_w:
                out_w = int(mat.shape[1])
            value_parts.append(np.ascontiguousarray(mat, dtype=np.float32))
            pen = None
            if resolved.penultimate_name is not None:
                pen = by_name.get(resolved.penultimate_name, None)
            if pen is None and len(outputs) > 1:
                pen = outputs[-1]
            penultimate_parts.append(normalize_intermediate(pen if pen is not None else mat, m))
            for name in resolved.probe_names:
                t = by_name.get(name, None)
                if t is None:
                    continue
                normed = normalize_intermediate(t, m)
                kept = topk_neuron_indices(normed, self.MAX_NEURONS_PER_PROBED_LAYER)
                probe_accum[name].append(np.ascontiguousarray(normed[:, kept] if kept else normed, dtype=np.float32))
            del outputs, by_name
        mat_all = (
            np.ascontiguousarray(np.concatenate(value_parts, axis=0).astype(np.float32))
            if value_parts
            else np.zeros((n, out_w), dtype=np.float32)
        )
        pen_all = (
            np.ascontiguousarray(np.concatenate(penultimate_parts, axis=0).astype(np.float32))
            if penultimate_parts
            else np.zeros((n, 1), dtype=np.float32)
        )
        probes = [np.ascontiguousarray(np.concatenate(v, axis=0)) if v else np.zeros((n, 0), dtype=np.float32) for v in probe_accum.values()]
        del value_parts, penultimate_parts, probe_accum
        latent, _, _ = truncated_svd_latent(pen_all, 3)
        info = {"transform": transform, "penultimate_width": int(pen_all.shape[1])}
        del pen_all
        return mat_all, np.ascontiguousarray(latent, dtype=np.float32), resolved, probes, info

    # -- BaseModelAdapter contract -------------------------------------
    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        width = max(int(n_features), 0)
        try:
            proto = self._resolve_proto(estimator)
            scores = _first_layer_feature_scores(proto, width)
            if scores is not None and len(scores) == width:
                return scores
        except (ValueError, TypeError, AttributeError):
            pass
        return [None for _ in range(width)]

    def predict_batch(self, model_obj: Any, X_float32: Any = None) -> PredictionBatchResult:  # type: ignore[override]
        # Spec-style ``predict_batch(df_chunk)``: single-arg call runs the
        # bound artifact; contract-style ``predict_batch(proto, X)`` is used
        # by the registry/orchestrators.
        if X_float32 is None:
            chunk = np.asarray(model_obj)
            proto = self._resolve_proto(None)
            mat, _, resolved, _, _ = self._infer_chunks(proto, chunk, int(chunk.shape[1]) if np.asarray(chunk).ndim == 2 else 0)
        else:
            proto = self._resolve_proto(model_obj)
            width = int(np.asarray(X_float32).shape[1]) if np.asarray(X_float32).ndim == 2 else 0
            mat, _, resolved, _, _ = self._infer_chunks(proto, X_float32, width)
        if resolved.is_classifier:
            n_classes = int(mat.shape[1])
            classes = list(range(n_classes))
            if mat.shape[1] == 1:
                p = np.asarray(mat, dtype=np.float64).ravel()
                py_preds: list[float | int | str] = [int(v >= 0.5) for v in p.tolist()]
                probabilities = [[float(1.0 - v), float(v)] for v in p.tolist()]
                confidences = [float(max(r)) for r in probabilities]
            else:
                idx = np.argmax(np.asarray(mat, dtype=np.float64), axis=1)
                py_preds = [int(v) for v in idx.tolist()]
                probabilities = [
                    [float(v) if np.isfinite(v) else 0.0 for v in row]
                    for row in np.asarray(mat, dtype=np.float64).tolist()
                ]
                confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            _ = classes
            return PredictionBatchResult(predictions=py_preds, probabilities=probabilities, confidences=confidences)
        vals = np.asarray(mat, dtype=np.float64).ravel()
        return PredictionBatchResult(
            predictions=[float(v) if np.isfinite(v) else 0.0 for v in vals.tolist()],
            probabilities=None,
            confidences=None,
        )

    def _finite_diff_row_gradient(
        self, proto: Any, resolved: _ResolvedDeepSession, width: int, cells: list[Any], resolved_idx: int
    ) -> tuple[np.ndarray, float]:
        """Central differences for one row (session-backed, §3 waterfall)."""
        while len(cells) < width:
            cells.append(0.0)
        cells = cells[:width]
        x_num = np.zeros((width,), dtype=np.float64)
        for j in range(width):
            try:
                x_num[j] = float(cells[j])
            except (TypeError, ValueError):
                x_num[j] = 0.0
        x_num = np.where(np.isfinite(x_num), x_num, 0.0)

        def scalar_at(row_cells: list[Any]) -> float:
            mat_in = np.empty((1, width), dtype=np.float32)
            for j, c in enumerate(row_cells):
                try:
                    mat_in[0, j] = float(c)
                except (TypeError, ValueError):
                    mat_in[0, j] = 0.0
            mat_in = np.where(np.isfinite(mat_in.astype(np.float64)), mat_in, 0.0).astype(np.float32)
            outs = run_session_outputs(resolved, mat_in)
            by_name = {nm: o for nm, o in zip(resolved.output_names, outs)}
            raw = by_name.get(resolved.final_output or "", outs[0] if outs else np.zeros((1, 1)))
            m, _ = apply_output_transform(np.asarray(raw), resolved.is_classifier)
            del outs, by_name
            if resolved.is_classifier:
                k = min(int(resolved_idx), int(m.shape[1]) - 1)
                return float(m[0, k])
            return float(m[0, 0])

        grad = np.zeros((width,), dtype=np.float64)
        scale = float(np.mean(np.abs(x_num)) + 1.0)
        eps = max(FINITE_DIFF_EPS * scale, 1e-6)
        for j in range(width):
            rp = list(cells)
            rm = list(cells)
            try:
                center = float(cells[j])
            except (TypeError, ValueError):
                continue
            rp[j] = center + eps
            rm[j] = center - eps
            grad[j] = (scalar_at(rp) - scalar_at(rm)) / (2.0 * eps)
        grad = np.where(np.isfinite(grad), grad, 0.0)
        base_cells = list(cells)
        final_pred = scalar_at(base_cells)
        attr = grad * x_num
        attr = np.where(np.isfinite(attr), attr, 0.0)
        base = float(final_pred - float(np.sum(attr)))
        return np.ascontiguousarray(attr, dtype=np.float32), (base if np.isfinite(base) else 0.0)

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        proto = self._resolve_proto(model_obj)
        meta = self._inspect_deep_operators(proto)
        width = int(np.asarray(X_float32).shape[1]) if np.asarray(X_float32).ndim == 2 else len(feature_names)
        names = list(feature_names) if len(feature_names) == width else [f"feature_{i}" for i in range(width)]
        n_classes = max(int(meta.output_width), 1) if meta.is_classifier else 1
        resolved_idx = min(max(int(target_class_idx if target_class_idx is not None else target_idx), 0), max(n_classes - 1, 0))
        X_all = np.asarray(X_float32)
        if X_all.ndim == 1:
            X_all = X_all.reshape(1, -1)
        mat_all, latent_all, resolved, _, info = self._infer_chunks(proto, X_all, width)
        latent_all = np.asarray(latent_all, dtype=np.float64)
        n_classes_live = int(mat_all.shape[1]) if mat_all.ndim == 2 else 1
        resolved_idx = min(resolved_idx, max(n_classes_live - 1, 0))
        # Chunk-level penultimate telemetry (recomputed per requested row
        # below would cost a session per row — instead reuse one probe pass).
        probe_telemetry: list[dict[str, Any]] = []
        try:
            probe_outs = run_session_outputs(resolved, np.ascontiguousarray(X_all[: min(int(X_all.shape[0]), 64)], dtype=np.float32))
            by_name_probe = {nm: o for nm, o in zip(resolved.output_names, probe_outs)}
            for name in resolved.probe_names:
                t = by_name_probe.get(name, None)
                if t is None:
                    continue
                probe_telemetry.append(layer_telemetry(name, normalize_intermediate(t, int(min(int(X_all.shape[0]), 64))), self.MAX_NEURONS_PER_PROBED_LAYER))
            del probe_outs, by_name_probe
        except (AttributeError, TypeError, ValueError, RuntimeError):
            probe_telemetry = []
        traces: list[InternalFlowRowTrace] = []
        for row_id in list(row_indices):
            if not (0 <= int(row_id) < int(X_all.shape[0])):
                continue
            rid = int(row_id)
            try:
                cells = list(np.asarray(X_all[rid], dtype=object).ravel().tolist())
            except (TypeError, ValueError):
                cells = [0.0] * width
            while len(cells) < width:
                cells.append(0.0)
            cells = cells[:width]
            x_num = np.zeros((width,), dtype=np.float64)
            for j in range(width):
                try:
                    x_num[j] = float(cells[j])
                except (TypeError, ValueError):
                    x_num[j] = 0.0
            x_num = np.where(np.isfinite(x_num), x_num, 0.0)
            attr, base = self._finite_diff_row_gradient(proto, resolved, width, list(cells), resolved_idx)
            if meta.is_classifier:
                k = min(int(resolved_idx), int(mat_all.shape[1]) - 1) if mat_all.size else 0
                final_pred = float(mat_all[rid, k]) if mat_all.size else 0.0
                proba_row = np.asarray(mat_all[rid], dtype=np.float64).ravel().tolist() if mat_all.size else []
                final_label: Any = int(k) if mat_all.shape[1] != 1 else int(final_pred >= 0.5)
            else:
                final_pred = float(mat_all[rid, 0]) if mat_all.size else 0.0
                proba_row = []
                final_label = float(final_pred)
            if not np.isfinite(final_pred):
                final_pred = 0.0
            latent_row = latent_all[rid].tolist() if rid < latent_all.shape[0] else [0.0, 0.0, 0.0]
            extra: dict[str, Any] = {
                "target_idx": int(resolved_idx),
                "target_class_idx": int(resolved_idx),
                "is_classifier": bool(meta.is_classifier),
                "is_regressor": bool(meta.is_regressor),
                "algorithm": "ONNXDeep",
                "model_family": "onnx_deep",
                "operator_family": "deep_mlp",
                "detected_operator": "Gemm/MatMul DeepNet",
                "detected_operator_family": "deep",
                "domain_badge": "ONNX Deep Net | In-Memory Probe Injection",
                "node_chain": list(meta.node_chain),
                "input_signature": [dict(s) for s in meta.input_signature],
                "opset_version": int(meta.opset_version),
                "attribution_method": "finite_difference_input_sensitivity",
                "output_transform": str(info.get("transform", "identity")),
                "provider": str(resolved.providers_used[0]) if resolved.providers_used else "CPUExecutionProvider",
                "providers_available": list(resolved.providers_used),
                "batch_mode": ("static_unrolled" if resolved.is_static_batch else "dynamic"),
                "static_batch_size": resolved.input_static_batch,
                "probed_layers": list(resolved.probe_names),
                "n_probed_layers": int(len(resolved.probe_names)),
                "penultimate_layer": resolved.penultimate_name,
                "penultimate_width": int(info.get("penultimate_width", 0)),
                "injection_active": bool(resolved.injection_active),
                "layer_telemetry": probe_telemetry,
                "latent_coords_3d": [float(v) for v in latent_row],
                "classes": list(range(int(mat_all.shape[1]))) if meta.is_classifier and mat_all.ndim == 2 else [],
                "class_labels": list(range(int(mat_all.shape[1]))) if meta.is_classifier and mat_all.ndim == 2 else [],
            }
            if meta.is_classifier:
                extra.update({
                    "probabilities": [float(v) for v in proba_row],
                    "predicted_class": final_label if isinstance(final_label, (int, float, str)) else str(final_label),
                    "resolved_class_idx": int(resolved_idx),
                    "winning_class_idx": int(resolved_idx),
                })
            else:
                extra.update({"predicted_value": float(final_pred)})
            order = np.argsort(np.abs(np.asarray(attr, dtype=np.float64)))[::-1][: min(8, width)] if width else np.asarray([], dtype=int)
            extra["top_shap_features"] = [{
                "rank": int(r + 1),
                "feature_idx": int(f),
                "feature_name": names[int(f)] if int(f) < len(names) else f"feature_{int(f)}",
                "shap_value": round(float(np.asarray(attr, dtype=np.float64)[int(f)]), 5),
            } for r, f in enumerate(np.asarray(order).tolist())]
            extra["base_raw_score"] = float(base)
            extra["final_raw_score"] = float(final_pred)
            try:
                has_nan = bool(np.isnan(np.asarray(X_all[rid], dtype=np.float64)).any())
            except (TypeError, ValueError):
                has_nan = False
            extra["has_missing_values"] = has_nan
            traces.append(InternalFlowRowTrace(
                row_index=rid,
                base_value=float(base),
                final_prediction=final_label,  # type: ignore[arg-type]
                feature_contributions={
                    names[i] if i < len(names) else f"feature_{i}": float(np.asarray(attr, dtype=np.float64)[i])
                    for i in range(int(width))
                },
                extra_mechanics=extra,
            ))
        return traces

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        proto = self._resolve_proto(model_obj)
        meta = self._inspect_deep_operators(proto)
        width = max(int(n_features), int(meta.n_features or 0), 1)
        names = [f"feature_{i}" for i in range(width)]
        n_classes = max(int(meta.output_width), 1) if meta.is_classifier else 1
        resolved = min(max(int(target_class_idx if target_class_idx is not None else target_idx), 0), max(n_classes - 1, 0))
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res).astype(np.float64)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res).astype(np.float64)
        medians = np.zeros((width,), dtype=np.float64)
        for feat_idx, med_val in (grid_spec.fixed_feature_medians or {}).items():
            try:
                j = int(feat_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= j < width:
                try:
                    medians[j] = float(med_val)
                except (TypeError, ValueError):
                    medians[j] = 0.0
        ax, ay = int(grid_spec.axis_x_feature_index), int(grid_spec.axis_y_feature_index)
        grid = np.tile(medians.reshape(1, -1), (res * res, 1))
        if 0 <= ax < width:
            grid[:, ax] = np.tile(x_lin, res)
        if 0 <= ay < width:
            grid[:, ay] = np.repeat(y_lin, res)
        grid = np.where(np.isfinite(grid), grid, 0.0).astype(np.float32)
        mat, latent, session, _, info = self._infer_chunks(proto, grid, width)
        k = min(int(resolved), int(mat.shape[1]) - 1) if mat.ndim == 2 else 0
        if meta.is_classifier:
            z_flat = np.asarray(mat[:, k], dtype=np.float64).ravel() if mat.size else np.zeros((res * res,))
        else:
            z_flat = np.asarray(mat[:, 0], dtype=np.float64).ravel() if mat.size else np.zeros((res * res,))
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        z_surface = z_flat.reshape(res, res)
        if meta.is_classifier:
            z_min, z_max = 0.0, 1.0
        else:
            z_min, z_max = (float(np.min(z_surface)), float(np.max(z_surface))) if z_surface.size else (0.0, 0.0)
        latent_sample = np.asarray(latent, dtype=np.float64).reshape(-1, 3)
        # Downsample the latent cloud for the payload (≤ 400 pts, §6).
        step = max(1, int(len(latent_sample) // 400))
        latent_down = latent_sample[::step][:400]
        overlays: dict[str, Any] = {
            "model_family": "onnx_deep",
            "operator_family": "deep_mlp",
            "detected_operator": "Gemm/MatMul DeepNet",
            "detected_operator_family": "deep",
            "domain_badge": "ONNX Deep Net | In-Memory Probe Injection",
            "node_chain": list(meta.node_chain),
            "input_signature": [dict(s) for s in meta.input_signature],
            "opset_version": int(meta.opset_version),
            "target_idx": int(resolved),
            "resolved_target_class_idx": int(resolved),
            "classes": list(range(int(mat.shape[1]))) if meta.is_classifier and mat.ndim == 2 else [],
            "is_classifier": bool(meta.is_classifier),
            "is_regressor": bool(meta.is_regressor),
            "is_binary": bool(meta.is_classifier and mat.shape[1] == 2),
            "num_class": int(mat.shape[1]) if meta.is_classifier and mat.ndim == 2 else 0,
            "regression_z_range": [float(z_min), float(z_max)],
            "boundary_threshold": 0.5 if meta.is_classifier else None,
            "output_transform": str(info.get("transform", "identity")),
            "provider": str(session.providers_used[0]) if session.providers_used else "CPUExecutionProvider",
            "batch_mode": ("static_unrolled" if session.is_static_batch else "dynamic"),
            "static_batch_size": session.input_static_batch,
            "probed_layers": list(session.probe_names),
            "n_probed_layers": int(len(session.probe_names)),
            "penultimate_layer": session.penultimate_name,
            "injection_active": bool(session.injection_active),
            "latent_manifold_sample": [[float(v) for v in row] for row in latent_down.tolist()],
            "latent_sample_stride": int(step),
        }
        return SurfaceMesh3DData(
            visual_type=Visual3DType.NEURAL_MORPHING_MANIFOLD,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in np.asarray(z_surface, dtype=np.float64).ravel()],
            extra_overlays=overlays,
        )


@ModelRegistry.register("onnx_deep")
class ONNXDeepClassifierAdapter(_ONNXDeepBase):
    display_name = "ONNX Deep Learning (Intermediate Probe Injector)"
    family_name = "ONNX Deep Learning"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            if not is_onnx_model_proto(estimator):
                return False
            if not is_deep_learning_proto(estimator):
                return False
            meta = inspect_onnx_deep(estimator)
            return bool(meta.is_classifier)
        except (ValueError, TypeError, AttributeError):
            return False

    def trace_internal_flow(  # type: ignore[override]
        self, model_obj: Any, X_float32: np.ndarray, row_indices: list[int],
        feature_names: list[str], target_idx: int = 0, target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return _ONNXDeepBase.trace_internal_flow(
            self, model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(  # type: ignore[override]
        self, model_obj: Any, grid_spec: GridSpec3D, n_features: int,
        target_class_idx: int | None = None, target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return super().compute_3d_surface(model_obj, grid_spec, n_features, target_class_idx, target_idx)


class ONNXDeepLearningAdapter(_ONNXDeepBase):
    """Spec facade: ``ONNXDeepLearningAdapter(model_path=..., onnx_proto=...)``.

    Delegates to the classifier / regressor subclass by inspected output
    width, mirroring the ``ONNXClassicalAdapter`` facade from Phase 38.
    """

    display_name = "ONNX Deep Learning (Intermediate Probe Injector)"
    family_name = "ONNX Deep Learning"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            return bool(is_deep_learning_proto(estimator))
        except (ValueError, TypeError, AttributeError):
            return False

    def _delegate_for(self, model_obj: Any | None) -> _ONNXDeepBase:
        target = model_obj if model_obj is not None else self._bound_proto
        try:
            meta = inspect_onnx_deep(target) if target is not None else None
        except (ValueError, TypeError, AttributeError):
            meta = None
        if meta is not None and meta.is_regressor:
            delegate = ONNXDeepRegressorAdapter.__new__(ONNXDeepRegressorAdapter)
            delegate._bound_path = self._bound_path
            delegate._bound_proto = self._bound_proto
            delegate._bound_providers = self._bound_providers
            delegate._bound_session = None
            return delegate
        delegate = ONNXDeepClassifierAdapter.__new__(ONNXDeepClassifierAdapter)
        delegate._bound_path = self._bound_path
        delegate._bound_proto = self._bound_proto
        delegate._bound_providers = self._bound_providers
        delegate._bound_session = None
        return delegate

    def trace_internal_flow(  # type: ignore[override]
        self, model_obj: Any, X_float32: np.ndarray, row_indices: list[int],
        feature_names: list[str], target_idx: int = 0, target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        target = model_obj if model_obj is not None else self._bound_proto
        return self._delegate_for(target).trace_internal_flow(
            target, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(  # type: ignore[override]
        self, model_obj: Any, grid_spec: GridSpec3D, n_features: int,
        target_class_idx: int | None = None, target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        target = model_obj if model_obj is not None else self._bound_proto
        return self._delegate_for(target).compute_3d_surface(
            target, grid_spec, n_features, target_idx, target_idx
        )


@ModelRegistry.register("onnx_deep_regressor")
class ONNXDeepRegressorAdapter(_ONNXDeepBase):
    display_name = "ONNX Deep Learning Regressor (Intermediate Probe Injector)"
    family_name = "ONNX Deep Learning"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            if not is_onnx_model_proto(estimator):
                return False
            if not is_deep_learning_proto(estimator):
                return False
            meta = inspect_onnx_deep(estimator)
            return bool(meta.is_regressor)
        except (ValueError, TypeError, AttributeError):
            return False

    def trace_internal_flow(  # type: ignore[override]
        self, model_obj: Any, X_float32: np.ndarray, row_indices: list[int],
        feature_names: list[str], target_idx: int = 0, target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return _ONNXDeepBase.trace_internal_flow(
            self, model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(  # type: ignore[override]
        self, model_obj: Any, grid_spec: GridSpec3D, n_features: int,
        target_class_idx: int | None = None, target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return super().compute_3d_surface(model_obj, grid_spec, n_features, target_class_idx, target_idx)

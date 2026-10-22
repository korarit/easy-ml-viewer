# backend/app/domains/model_engine/adapters/neural_onnx/onnx_classical.py
"""Phase 38 — ONNX Runtime Classical ML inspector (``ai.onnx.ml`` operators).

Covers ``.onnx`` artifacts exported from classical sklearn-style estimators
(via ``skl2onnx`` / ``onnxmltools``) whose graphs run on ``ai.onnx.ml``
operators: ``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor`` (OpSet <= 3
and unified ``TreeEnsemble`` OpSet 5), ``LinearClassifier`` /
``LinearRegressor``, and ``SVMClassifier`` / ``SVMRegressor``.

Blueprint mapping (phase-38-onnx-classical-ml-engine.md):

- §2.1 ``ai.onnx.ml`` attribute decompilation — ``TreeEnsemble*`` flattened
  parallel arrays (``nodes_treeids`` / ``nodes_featureids`` / ``nodes_values``
  / ``nodes_modes`` / ``nodes_truenodeids`` / ``nodes_falsenodeids`` plus
  ``class_*`` or ``target_*`` leaf weights), ``Linear*`` (``coefficients`` /
  ``intercepts`` reshaped to ``W in R^{C x F}``), ``SVM*``
  (``support_vectors`` / ``coefficients`` / ``kernel_type`` /
  ``kernel_params``).
- §2.2 single 2D tensor input (``float_input: [None, F]``) vs multi-column
  named inputs (``[None, 1]`` per column, e.g. DataFrame pipelines).
- §2.3 ``ZipMap`` (``seq(map(int64/string, float))``) unpacking — the adapter
  rewires ``graph.output`` in memory to expose the raw ``float32[N, C]``
  probability tensor before ``ZipMap`` (~15x faster, no Python-object
  blowup), with a ``list[dict]`` → ``ndarray`` fallback unpacker.
- §6 memory guards — ``np.ascontiguousarray`` before every ``run()`` call,
  ``<= 2_500``-row inference chunks, ``ORT_SEQUENTIAL`` with
  ``intra_op_num_threads = max(1, cores - 1)``.

Attribution methods (all satisfy ``base + sum(contribs) == prediction``):

- ``LinearRegressor``: exact ``phi_j = w_j * x_j``, ``base = intercept``.
- ``LinearClassifier``: exact analytic Input x Gradient
  (``g = p(1-p) w`` binary logistic, softmax Jacobian multiclass).
- ``TreeEnsemble*``: NumPy tree walk gives the true ``decision_path``;
  the residual ``(final - global_leaf_prior)`` is allocated across split
  features on the path proportional to split frequency (``tree_path_share``).
- ``SVM*``: analytic kernel gradient (RBF / Poly / Linear / Sigmoid,
  binary) or session finite differences (multiclass), Input x Gradient.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
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

ONNX_TYPE_TO_NP: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
    "tensor(string)": np.object_,
}

_TREE_OPS = frozenset({"TreeEnsembleClassifier", "TreeEnsembleRegressor", "TreeEnsemble"})
_LINEAR_OPS = frozenset({"LinearClassifier", "LinearRegressor"})
_SVM_OPS = frozenset({"SVMClassifier", "SVMRegressor"})
_CLASSICAL_OPS = _TREE_OPS | _LINEAR_OPS | _SVM_OPS


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


# ---------------------------------------------------------------------------
# Attribute + graph inspection (pure python, no onnxruntime needed)
# ---------------------------------------------------------------------------


def _attr_to_python(attr: Any) -> Any:
    """Convert an ``onnx.AttributeProto`` to plain python scalars / lists."""
    kind = int(getattr(attr, "type", 0))
    if kind == 1:  # FLOAT
        return float(getattr(attr, "f", 0.0))
    if kind == 2:  # INT
        return int(getattr(attr, "i", 0))
    if kind == 3:  # STRING
        raw = getattr(attr, "s", b"")
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    if kind == 6:  # FLOATS
        return [float(v) for v in list(getattr(attr, "floats", []))]
    if kind == 7:  # INTS
        return [int(v) for v in list(getattr(attr, "ints", []))]
    if kind == 8:  # STRINGS
        out = []
        for v in list(getattr(attr, "strings", [])):
            out.append(v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v))
        return out
    return None


def _node_attrs(node: Any) -> dict[str, Any]:
    return {str(a.name): _attr_to_python(a) for a in list(getattr(node, "attribute", []))}


def _classical_nodes(proto: Any) -> list[Any]:
    """All ``ai.onnx.ml`` classical estimator nodes in graph order."""
    out = []
    for node in list(proto.graph.node):
        if str(getattr(node, "domain", "")) == "ai.onnx.ml" and str(node.op_type) in _CLASSICAL_OPS:
            out.append(node)
    return out


def _zipmap_nodes(proto: Any) -> list[Any]:
    return [
        n
        for n in list(proto.graph.node)
        if str(getattr(n, "domain", "")) == "ai.onnx.ml" and str(n.op_type) == "ZipMap"
    ]


def _node_chain_summary(proto: Any) -> list[str]:
    """Compact operator chain, e.g. ``ai.onnx.ml::Scaler -> TreeEnsemble*``."""
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
        sig.append({"name": str(inp.name), "shape": shape, "dtype": dtype})
    return sig


def _output_signature(proto: Any) -> list[dict[str, Any]]:
    sig: list[dict[str, Any]] = []
    for out in list(proto.graph.output):
        which = out.type.WhichOneof("value") if hasattr(out.type, "WhichOneof") else None
        sig.append({"name": str(out.name), "kind": str(which or "unknown")})
    return sig


def _class_labels_from_attrs(attrs: dict[str, Any]) -> list[Any] | None:
    for key in ("classlabels_int64s", "classlabels_ints", "class_ids"):
        vals = attrs.get(key)
        if isinstance(vals, list) and vals:
            return [int(v) for v in vals]
    for key in ("classlabels_strings", "classlabels_strs"):
        vals = attrs.get(key)
        if isinstance(vals, list) and vals:
            return [str(v) for v in vals]
    return None


@dataclass
class ONNXClassicalMetadata:
    """Decompiled ``ai.onnx.ml`` graph summary (JSON-serializable via to_dict)."""

    detected_operator: str = "unknown"
    operator_family: str = "unknown"  # tree_ensemble | linear | svm | unknown
    is_classifier: bool = False
    is_regressor: bool = False
    classes: list[Any] = field(default_factory=list)
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    input_signature: list[dict[str, Any]] = field(default_factory=list)
    output_signature: list[dict[str, Any]] = field(default_factory=list)
    node_chain: list[str] = field(default_factory=list)
    has_zipmap: bool = False
    zipmap_input_tensor: str | None = None
    opset_version: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_operator": self.detected_operator,
            "operator_family": self.operator_family,
            "is_classifier": bool(self.is_classifier),
            "is_regressor": bool(self.is_regressor),
            "classes": list(self.classes),
            "n_features": int(self.n_features),
            "feature_names": list(self.feature_names),
            "input_signature": [dict(s) for s in self.input_signature],
            "output_signature": [dict(s) for s in self.output_signature],
            "node_chain": list(self.node_chain),
            "has_zipmap": bool(self.has_zipmap),
            "zipmap_input_tensor": self.zipmap_input_tensor,
            "opset_version": int(self.opset_version),
        }


def inspect_onnx_classical(proto: Any) -> ONNXClassicalMetadata:
    """Scan ``graph.node`` for the primary ``ai.onnx.ml`` estimator (§2.1)."""
    meta = ONNXClassicalMetadata()
    meta.input_signature = _input_signature(proto)
    meta.output_signature = _output_signature(proto)
    meta.node_chain = _node_chain_summary(proto)
    try:
        for op in list(proto.opset_import):
            if str(getattr(op, "domain", "")) == "ai.onnx.ml":
                meta.opset_version = int(getattr(op, "version", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        pass

    classical = _classical_nodes(proto)
    if not classical:
        return meta
    node = classical[0]
    op = str(node.op_type)
    meta.detected_operator = op
    attrs = _node_attrs(node)
    # Keep only JSON-safe summary scalars (arrays stay in the adapter).
    meta.attrs = {
        k: (v if isinstance(v, (str, int, float, bool)) else f"<{type(v).__name__}:{len(v) if hasattr(v, '__len__') else '?'} items>")
        for k, v in attrs.items()
    }
    if op in _TREE_OPS:
        meta.operator_family = "tree_ensemble"
    elif op in _LINEAR_OPS:
        meta.operator_family = "linear"
    elif op in _SVM_OPS:
        meta.operator_family = "svm"
    meta.is_classifier = op.endswith("Classifier") or (
        op == "TreeEnsemble" and _class_labels_from_attrs(attrs) is not None
    )
    meta.is_regressor = not meta.is_classifier and op != "unknown"

    labels = _class_labels_from_attrs(attrs)
    if labels is None:
        for z in _zipmap_nodes(proto):
            labels = _class_labels_from_attrs(_node_attrs(z))
            if labels is not None:
                break
    meta.classes = list(labels) if labels is not None else []

    sig = meta.input_signature
    if len(sig) == 1:
        shape = sig[0].get("shape") or []
        width = 0
        if len(shape) >= 2 and isinstance(shape[1], int):
            width = int(shape[1])
        meta.n_features = width
        meta.feature_names = [f"feature_{i}" for i in range(max(width, 0))]
    elif len(sig) > 1:
        meta.n_features = len(sig)
        meta.feature_names = [str(s.get("name", f"feature_{i}")) for i, s in enumerate(sig)]

    zips = _zipmap_nodes(proto)
    meta.has_zipmap = len(zips) > 0
    if zips:
        try:
            meta.zipmap_input_tensor = str(list(zips[0].input)[0])
        except (IndexError, TypeError):
            meta.zipmap_input_tensor = None
    return meta


# ---------------------------------------------------------------------------
# Session construction with in-memory ZipMap bypass (§6.1)
# ---------------------------------------------------------------------------


@dataclass
class _ResolvedSession:
    session: Any
    label_output: str | None
    proba_output: str | None
    bypass_active: bool
    classes: list[Any]
    is_classifier: bool


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
    return opts


def build_optimized_session(proto: Any) -> _ResolvedSession:
    """Create an ``InferenceSession`` exposing raw ``float32[N, C]`` probas.

    When the graph ends in ``ZipMap``, ``graph.output`` is rewired in a deep
    copy to point at the ``ZipMap`` input tensor (the pre-``ZipMap``
    probabilities) while keeping the label output, so ``sess.run`` returns
    dense ``ndarray``s instead of ``list[dict]`` (§6.1).
    """
    onnx = _load_onnx_module()
    ort = _load_ort_module()
    meta = inspect_onnx_classical(proto)
    classes = list(meta.classes)

    working = copy.deepcopy(proto)
    bypass = False
    zips = _zipmap_nodes(working)
    if zips and meta.zipmap_input_tensor:
        prob_name = str(meta.zipmap_input_tensor)
        n_classes = max(len(classes), 1)
        try:
            from onnx import TensorProto, helper  # noqa: PLC0415

            # Keep every non-ZipMap graph output (the label tensor) and append
            # the raw probability tensor output.
            keep = [o for o in list(working.graph.output) if str(o.name) != str(list(zips[0].output)[0])]
            del working.graph.output[:]
            for o in keep:
                working.graph.output.append(o)
            working.graph.output.append(
                helper.make_tensor_value_info(prob_name, TensorProto.FLOAT, [None, n_classes])
            )
            bypass = True
        except (AttributeError, TypeError, ValueError):
            bypass = False

    sess = ort.InferenceSession(
        working.SerializeToString(),
        sess_options=_session_options(ort),
        providers=_preferred_providers(ort),
    )
    out_names = [str(o.name) for o in sess.get_outputs()]
    label_output: str | None = None
    proba_output: str | None = None
    if meta.is_classifier:
        # Label output is the int64 tensor; probas the float tensor (or the
        # rewired ZipMap input).
        float_outs = [n for n in out_names if n == (meta.zipmap_input_tensor or "")] or out_names[1:]
        label_output = out_names[0] if out_names else None
        proba_output = float_outs[0] if float_outs else None
        if not classes:
            classes = list(range(2 if proba_output is None else 2))
    else:
        proba_output = out_names[0] if out_names else None
    return _ResolvedSession(
        session=sess,
        label_output=label_output,
        proba_output=proba_output,
        bypass_active=bool(bypass),
        classes=classes,
        is_classifier=bool(meta.is_classifier),
    )


def unpack_zipmap_sequence(raw: Any, class_labels: list[Any]) -> np.ndarray:
    """Fallback: ``list[dict[int|str, float]]`` → ``float32 (N, C)`` (§2.3)."""
    rows = list(raw) if not isinstance(raw, list) else raw
    n = len(rows)
    c = max(len(class_labels), 1)
    mat = np.zeros((n, c), dtype=np.float32)
    index_of: dict[str, int] = {}
    for i, lab in enumerate(class_labels):
        index_of[str(lab)] = i
        try:
            index_of[int(lab)] = i  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    for r, mapping in enumerate(rows):
        try:
            items = dict(mapping).items()
        except (TypeError, ValueError, AttributeError):
            continue
        for k, v in items:
            j = index_of.get(k, index_of.get(str(k), None))
            if j is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            mat[r, j] = f if np.isfinite(f) else 0.0
    return np.ascontiguousarray(mat, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feed-dict construction: single 2D tensor vs multi named inputs (§2.2)
# ---------------------------------------------------------------------------


def _numpy_dtype_for_input(dtype_str: str) -> Any:
    return ONNX_TYPE_TO_NP.get(str(dtype_str), np.float32)


def build_feed_dict(session: Any, X_chunk: Any, feature_names: list[str]) -> dict[str, np.ndarray]:
    """Map a chunk to ``{input_name: contiguous array}`` (§2.2)."""
    inputs = list(session.get_inputs())
    if len(inputs) == 1 and len(feature_names) >= 1:
        spec = inputs[0]
        target = _numpy_dtype_for_input(getattr(spec, "type", "tensor(float)"))
        arr = np.asarray(X_chunk)
        if target is np.object_:
            feed = np.asarray(arr, dtype=object).reshape(int(arr.shape[0]), -1)
        else:
            with np.errstate(all="ignore"):
                feed = np.asarray(arr, dtype=target)
            if feed.ndim == 1:
                feed = feed.reshape(1, -1)
            feed = np.where(np.isfinite(feed.astype(np.float64)), feed, 0.0).astype(target, copy=False)
        return {str(spec.name): np.ascontiguousarray(feed)}
    # Multi-column named inputs: positional mapping in schema order.
    arr = np.asarray(X_chunk, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_rows = int(arr.shape[0])
    feed: dict[str, np.ndarray] = {}
    for j, spec in enumerate(inputs):
        target = _numpy_dtype_for_input(getattr(spec, "type", "tensor(float)"))
        col = arr[:, j] if j < arr.shape[1] else np.zeros((n_rows,), dtype=object)
        col = np.asarray(col).reshape(n_rows, 1)
        if target is np.object_:
            feed[str(spec.name)] = np.ascontiguousarray(col.astype(object))
        elif target in (np.int64, np.int32):
            with np.errstate(all="ignore"):
                num = np.asarray(col, dtype=np.float64).ravel()
            num = np.where(np.isfinite(num), num, 0.0)
            feed[str(spec.name)] = np.ascontiguousarray(num.astype(target).reshape(n_rows, 1))
        else:
            with np.errstate(all="ignore"):
                num = np.asarray(col, dtype=np.float64).ravel()
            num = np.where(np.isfinite(num), num, 0.0).astype(np.float32)
            feed[str(spec.name)] = np.ascontiguousarray(num.astype(target).reshape(n_rows, 1))
    _ = feature_names
    return feed


# ---------------------------------------------------------------------------
# Decompiled parameter accessors (§2.1)
# ---------------------------------------------------------------------------


def _primary_classical_node(proto: Any) -> tuple[Any, dict[str, Any]]:
    nodes = _classical_nodes(proto)
    if not nodes:
        raise ValueError("ONNX graph holds no ai.onnx.ml TreeEnsemble/Linear/SVM node")
    node = nodes[0]
    return node, _node_attrs(node)


def decompile_linear(attrs: dict[str, Any], n_features: int) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return ``(W[C, F], b[C], multi_class, post_transform)``."""
    coefs = [float(v) for v in list(attrs.get("coefficients", []) or [])]
    intercepts = [float(v) for v in list(attrs.get("intercepts", []) or [])]
    post = str(attrs.get("post_transform", "NONE") or "NONE").upper()
    multi = str(attrs.get("multi_class", "") or "")
    n_feat = max(int(n_features), 1)
    if not coefs:
        raise ValueError("Linear* node carries no coefficients")
    if intercepts and len(coefs) % max(len(intercepts), 1) == 0 and len(coefs) != n_feat:
        n_cls = max(len(intercepts), 1)
        w = np.asarray(coefs, dtype=np.float64).reshape(n_cls, len(coefs) // n_cls)
        b = np.asarray(intercepts, dtype=np.float64).ravel()
    elif len(coefs) == n_feat:
        w = np.asarray(coefs, dtype=np.float64).reshape(1, n_feat)
        b = np.asarray(intercepts if intercepts else [0.0], dtype=np.float64).ravel()
        if b.size == 0:
            b = np.zeros((1,), dtype=np.float64)
    else:
        # Best-effort reshape when the exporter flattened C*F.
        n_cls = max(len(intercepts) if intercepts else 1, 1)
        per = len(coefs) // n_cls
        w = np.asarray(coefs[: n_cls * per], dtype=np.float64).reshape(n_cls, per)
        b = np.asarray(intercepts if intercepts else [0.0] * n_cls, dtype=np.float64).ravel()
        if b.size < n_cls:
            b = np.pad(b, (0, n_cls - b.size))
    return w, b, multi, post


def _as_int_list(vals: Any) -> list[int]:
    if isinstance(vals, list):
        return [int(v) for v in vals]
    return []


def _as_float_list(vals: Any) -> list[float]:
    if isinstance(vals, list):
        return [float(v) for v in vals]
    return []


@dataclass
class DecompiledTreeEnsemble:
    n_trees: int = 0
    n_nodes: int = 0
    modes: list[str] = field(default_factory=list)
    tree_ids: list[int] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    feature_ids: list[int] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)
    true_ids: list[int] = field(default_factory=list)
    false_ids: list[int] = field(default_factory=list)
    # (tree_id, node_id) -> {class_or_target_id: weight}
    leaf_weights: dict[tuple[int, int], dict[Any, float]] = field(default_factory=dict)
    leaf_class_order: list[Any] = field(default_factory=list)
    post_transform: str = "NONE"


def decompile_tree_ensemble(attrs: dict[str, Any]) -> DecompiledTreeEnsemble:
    """Flattened parallel arrays → lookup tables (OpSet ≤3 and OpSet 5)."""
    tree = DecompiledTreeEnsemble()
    tree.tree_ids = _as_int_list(attrs.get("nodes_treeids", attrs.get("nodes_trees", [])))
    tree.node_ids = _as_int_list(attrs.get("nodes_nodeids", attrs.get("nodes_ids", [])))
    tree.feature_ids = _as_int_list(attrs.get("nodes_featureids", attrs.get("nodes_features", [])))
    tree.thresholds = _as_float_list(attrs.get("nodes_values", attrs.get("nodes_splits", [])))
    raw_modes = attrs.get("nodes_modes", [])
    tree.modes = [str(m) for m in list(raw_modes)] if isinstance(raw_modes, list) else []
    tree.true_ids = _as_int_list(attrs.get("nodes_truenodeids", attrs.get("nodes_true_ids", [])))
    tree.false_ids = _as_int_list(attrs.get("nodes_falsenodeids", attrs.get("nodes_false_ids", [])))
    tree.n_nodes = len(tree.node_ids)
    tree.n_trees = len(set(tree.tree_ids)) if tree.tree_ids else 0
    tree.post_transform = str(attrs.get("post_transform", "NONE") or "NONE").upper()
    # Leaf weights: classifier (class_*) or regressor (target_*) spellings.
    if "class_treeids" in attrs or "class_nodeids" in attrs:
        t_ids = _as_int_list(attrs.get("class_treeids", []))
        n_ids = _as_int_list(attrs.get("class_nodeids", []))
        c_ids = attrs.get("class_ids", [])
        weights = _as_float_list(attrs.get("class_weights", []))
        order: list[Any] = []
        for t, n, c, w in zip(t_ids, n_ids, list(c_ids), weights):
            try:
                key_c: Any = int(c)
            except (TypeError, ValueError):
                key_c = c
            if key_c not in order:
                order.append(key_c)
            tree.leaf_weights.setdefault((int(t), int(n)), {})[key_c] = float(w)
        tree.leaf_class_order = order
    else:
        t_ids = _as_int_list(attrs.get("target_treeids", []))
        n_ids = _as_int_list(attrs.get("target_nodeids", []))
        c_ids = attrs.get("target_ids", [0] * len(t_ids))
        weights = _as_float_list(attrs.get("target_weights", []))
        for t, n, c, w in zip(t_ids, n_ids, list(c_ids), weights):
            try:
                key_c = int(c)
            except (TypeError, ValueError):
                key_c = c
            tree.leaf_weights.setdefault((int(t), int(n)), {})[key_c] = float(w)
        tree.leaf_class_order = sorted({k for m in tree.leaf_weights.values() for k in m})
    return tree


def _walk_tree(
    tree: DecompiledTreeEnsemble,
    tree_id: int,
    x_row: np.ndarray,
) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
    """Walk one tree; return ``(path_steps, leaf_key)``."""
    children: dict[int, dict[str, Any]] = {}
    for pos, tid in enumerate(tree.tree_ids):
        if int(tid) != int(tree_id):
            continue
        nid = int(tree.node_ids[pos]) if pos < len(tree.node_ids) else -1
        children[nid] = {
            "mode": str(tree.modes[pos]) if pos < len(tree.modes) else "LEAF",
            "feat": int(tree.feature_ids[pos]) if pos < len(tree.feature_ids) else -1,
            "thr": float(tree.thresholds[pos]) if pos < len(tree.thresholds) else 0.0,
            "true": int(tree.true_ids[pos]) if pos < len(tree.true_ids) else -1,
            "false": int(tree.false_ids[pos]) if pos < len(tree.false_ids) else -1,
        }
    if 0 not in children:
        return [], None
    path: list[dict[str, Any]] = []
    cur = 0
    guard = 0
    while guard < max(len(children) + 2, 8):
        guard += 1
        info = children.get(cur)
        if info is None:
            return path, None
        if info["mode"] == "LEAF":
            return path, (int(tree_id), int(cur))
        f = int(info["feat"])
        thr = float(info["thr"])
        try:
            xv = float(x_row[f]) if 0 <= f < len(x_row) else 0.0
        except (TypeError, ValueError, IndexError):
            xv = 0.0
        if not np.isfinite(xv):
            xv = 0.0
        mode = str(info["mode"])
        if mode == "BRANCH_LEQ":
            go_true = bool(xv <= thr)
        elif mode == "BRANCH_LT":
            go_true = bool(xv < thr)
        elif mode == "BRANCH_GTE":
            go_true = bool(xv >= thr)
        elif mode == "BRANCH_GT":
            go_true = bool(xv > thr)
        elif mode == "BRANCH_EQ":
            go_true = bool(xv == thr)
        elif mode == "BRANCH_NEQ":
            go_true = bool(xv != thr)
        else:  # Unknown split mode — default to LEQ semantics.
            go_true = bool(xv <= thr)
        path.append({
            "tree_id": int(tree_id),
            "node_id": int(cur),
            "feature_idx": f,
            "threshold": thr,
            "mode": mode,
            "went_true": bool(go_true),
        })
        nxt = int(info["true"]) if go_true else int(info["false"])
        if nxt == cur or nxt < 0:
            return path, None
        cur = nxt
    return path, None


@dataclass
class DecompiledSVM:
    kernel_type: str = "LINEAR"
    kernel_params: list[float] = field(default_factory=list)  # [gamma, coef0, degree]
    support_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    coefficients: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    vectors_per_class: list[int] = field(default_factory=list)
    prob_a: list[float] = field(default_factory=list)
    prob_b: list[float] = field(default_factory=list)
    n_support: int = 0


def decompile_svm(attrs: dict[str, Any], n_features: int) -> DecompiledSVM:
    svm = DecompiledSVM()
    svm.kernel_type = str(attrs.get("kernel_type", "LINEAR") or "LINEAR").upper()
    svm.kernel_params = _as_float_list(attrs.get("kernel_params", []))
    while len(svm.kernel_params) < 3:
        svm.kernel_params.append(0.0)
    flat = _as_float_list(attrs.get("support_vectors", []))
    n_feat = max(int(n_features), 1)
    if flat and n_feat:
        rows = len(flat) // n_feat
        svm.support_vectors = np.asarray(flat[: rows * n_feat], dtype=np.float64).reshape(rows, n_feat)
        svm.n_support = rows
    svm.coefficients = _as_float_list(attrs.get("coefficients", []))
    svm.rho = _as_float_list(attrs.get("rho", []))
    svm.vectors_per_class = _as_int_list(attrs.get("vectors_per_class", []))
    svm.prob_a = _as_float_list(attrs.get("prob_a", []))
    svm.prob_b = _as_float_list(attrs.get("prob_b", []))
    return svm


def _svm_decision_binary(x_row: np.ndarray, svm: DecompiledSVM) -> float:
    """Raw ``f(x)`` for a binary SVM (coefficients already carry the sign)."""
    gamma = float(svm.kernel_params[0]) if len(svm.kernel_params) > 0 else 0.0
    coef0 = float(svm.kernel_params[1]) if len(svm.kernel_params) > 1 else 0.0
    degree = float(svm.kernel_params[2]) if len(svm.kernel_params) > 2 else 3.0
    rho = float(svm.rho[0]) if svm.rho else 0.0
    total = 0.0
    for i in range(svm.n_support):
        sv = svm.support_vectors[i]
        alpha = float(svm.coefficients[i]) if i < len(svm.coefficients) else 0.0
        if svm.kernel_type == "RBF":
            diff = sv - x_row
            total += alpha * float(np.exp(-gamma * float(np.dot(diff, diff))))
        elif svm.kernel_type == "POLY":
            total += alpha * float((gamma * float(np.dot(sv, x_row)) + coef0) ** degree)
        elif svm.kernel_type == "SIGMOID":
            total += alpha * float(np.tanh(gamma * float(np.dot(sv, x_row)) + coef0))
        else:  # LINEAR
            total += alpha * float(np.dot(sv, x_row))
    return float(total + rho)


def _svm_gradient_binary(x_row: np.ndarray, svm: DecompiledSVM) -> np.ndarray:
    """Analytic ``df/dx`` for a binary SVM (§2.1 kernel saddle)."""
    gamma = float(svm.kernel_params[0]) if len(svm.kernel_params) > 0 else 0.0
    coef0 = float(svm.kernel_params[1]) if len(svm.kernel_params) > 1 else 0.0
    degree = float(svm.kernel_params[2]) if len(svm.kernel_params) > 2 else 3.0
    grad = np.zeros_like(x_row, dtype=np.float64)
    for i in range(svm.n_support):
        sv = svm.support_vectors[i]
        alpha = float(svm.coefficients[i]) if i < len(svm.coefficients) else 0.0
        if svm.kernel_type == "RBF":
            diff = sv - x_row
            k = float(np.exp(-gamma * float(np.dot(diff, diff))))
            grad += alpha * (2.0 * gamma * k) * diff
        elif svm.kernel_type == "POLY":
            base = gamma * float(np.dot(sv, x_row)) + coef0
            grad += alpha * degree * gamma * (base ** (degree - 1.0)) * sv
        elif svm.kernel_type == "SIGMOID":
            t = float(np.tanh(gamma * float(np.dot(sv, x_row)) + coef0))
            grad += alpha * gamma * (1.0 - t * t) * sv
        else:  # LINEAR
            grad += alpha * sv
    return np.where(np.isfinite(grad), grad, 0.0)


def _stable_sigmoid(z: float) -> float:
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = float(np.exp(z))
    return float(e / (1.0 + e))


# ---------------------------------------------------------------------------
# Adapter base
# ---------------------------------------------------------------------------


class _ONNXClassicalBase(BaseModelAdapter):
    is_classifier: bool = True
    supports_nan_input: bool = False

    def __init__(self, model_path: Any = None, onnx_proto: Any = None) -> None:
        # Spec-style construction ``ONNXClassicalAdapter(model_path=...)``
        # binds the artifact; contract-style use leaves it None.
        self._bound_path = model_path
        self._bound_proto = onnx_proto

    # -- resolution ------------------------------------------------------
    def _resolve_proto(self, model_obj: Any | None) -> Any:
        target = model_obj if model_obj is not None else self._bound_proto
        if target is None and self._bound_path is not None:
            onnx = _load_onnx_module()
            target = onnx.load(str(self._bound_path))
        if target is None:
            raise ValueError("ONNXClassicalAdapter requires an onnx.ModelProto (or model_path)")
        if not is_onnx_model_proto(target):
            raise ValueError(f"Estimator '{type(target).__name__}' is not an ONNX model")
        if not _classical_nodes(target):
            raise ValueError("ONNX graph holds no ai.onnx.ml TreeEnsemble/Linear/SVM node")
        return target

    def _inspect_ml_operators(self, proto: Any) -> ONNXClassicalMetadata:
        """Scan ``graph.node`` for ``TreeEnsemble*`` / ``Linear*`` / ``SVM*``."""
        return inspect_onnx_classical(proto)

    def _build_optimized_session(self, proto: Any) -> _ResolvedSession:
        """Session with in-memory ``ZipMap`` removal when present (§6.1)."""
        return build_optimized_session(proto)

    def _build_feed_dict(self, session: Any, df_chunk: Any, feature_names: list[str] | None = None) -> dict[str, np.ndarray]:
        """Single 2D tensor vs multi-column named tensor inputs (§2.2)."""
        return build_feed_dict(session, df_chunk, list(feature_names or []))

    def decompile_tree_ensemble_splits(
        self, proto: Any, feat_x_idx: int, feat_y_idx: int
    ) -> dict[str, list[float]]:
        """Split thresholds on the X/Y axes for 3D split-plane overlays."""
        _, attrs = _primary_classical_node(proto)
        tree = decompile_tree_ensemble(attrs)
        out = {"axis_x_thresholds": [], "axis_y_thresholds": []}
        for pos, mode in enumerate(tree.modes):
            if str(mode) == "LEAF" or pos >= len(tree.feature_ids):
                continue
            f = int(tree.feature_ids[pos])
            thr = float(tree.thresholds[pos]) if pos < len(tree.thresholds) else 0.0
            if not np.isfinite(thr):
                continue
            if f == int(feat_x_idx):
                out["axis_x_thresholds"].append(thr)
            if f == int(feat_y_idx):
                out["axis_y_thresholds"].append(thr)
        out["axis_x_thresholds"] = sorted(set(out["axis_x_thresholds"]))
        out["axis_y_thresholds"] = sorted(set(out["axis_y_thresholds"]))
        return out

    # -- BaseModelAdapter contract -------------------------------------
    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        width = max(int(n_features), 0)
        try:
            proto = self._resolve_proto(estimator)
            meta = self._inspect_ml_operators(proto)
            _, attrs = _primary_classical_node(proto)
            if meta.operator_family == "linear":
                w, _, _, _ = decompile_linear(attrs, meta.n_features or width)
                if int(w.shape[1]) != width:
                    # Preprocessing prefix changes width — raw-space weights
                    # are not directly readable from W.
                    return [None for _ in range(width)]
                scores = np.mean(np.abs(w), axis=0)
                return [float(v) if np.isfinite(v) else 0.0 for v in scores.tolist()][:width]
            if meta.operator_family == "tree_ensemble":
                tree = decompile_tree_ensemble(attrs)
                counts = np.zeros((width,), dtype=np.float64)
                for pos, mode in enumerate(tree.modes):
                    if str(mode) == "LEAF" or pos >= len(tree.feature_ids):
                        continue
                    f = int(tree.feature_ids[pos])
                    if 0 <= f < width:
                        counts[f] += 1.0
                total = float(np.sum(counts))
                if total <= 0:
                    return [None for _ in range(width)]
                return [float(v / total) for v in counts.tolist()]
        except (ValueError, TypeError, AttributeError):
            pass
        return [None for _ in range(width)]

    def _infer_chunks(
        self,
        proto: Any,
        X: Any,
        feature_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, _ResolvedSession]:
        """Single-session chunked inference → ``(labels, value_matrix)``.

        ``value_matrix`` is ``(N, C)`` probabilities for classifiers
        (``ZipMap`` bypassed in-memory, ``list[dict]`` fallback otherwise)
        or ``(N, 1)`` raw values for regressors. Chunks are ``<= 2_500``
        rows (§6) and every feed array is C-contiguous.
        """
        resolved = self._build_optimized_session(proto)
        sess = resolved.session
        arr = np.asarray(X, dtype=object) if np.asanyarray(X).dtype == object else np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_rows = int(arr.shape[0])
        width = max(len(resolved.classes), 1) if resolved.is_classifier else 1
        if n_rows == 0:
            return np.zeros((0,), dtype=object), np.zeros((0, width), dtype=np.float32), resolved
        label_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for start in range(0, n_rows, PREDICT_CHUNK_SIZE):
            chunk = arr[start:start + PREDICT_CHUNK_SIZE]
            m = int(np.asarray(chunk).shape[0])
            feed = self._build_feed_dict(sess, chunk, feature_names)
            out_names = [o.name for o in sess.get_outputs()]
            raw = sess.run(out_names, feed)
            by_name = {n: r for n, r in zip(out_names, raw)}
            if resolved.is_classifier:
                lab = by_name.get(resolved.label_output or "", None)
                label_parts.append(
                    np.asarray(lab).ravel() if lab is not None else np.zeros((m,), dtype=object)
                )
                prob = by_name.get(resolved.proba_output or "", None)
                if isinstance(prob, list):
                    prob = unpack_zipmap_sequence(prob, resolved.classes)
                value_parts.append(np.asarray(prob, dtype=np.float32).reshape(m, -1))
            else:
                val = by_name.get(resolved.proba_output or out_names[0], None)
                value_parts.append(np.asarray(val, dtype=np.float32).reshape(m, -1))
                label_parts.append(np.zeros((m,), dtype=object))
        labels = np.concatenate(label_parts).ravel() if label_parts else np.zeros((n_rows,), dtype=object)
        mat = (
            np.ascontiguousarray(np.asarray(np.concatenate(value_parts), dtype=np.float32))
            if value_parts
            else np.zeros((n_rows, width), dtype=np.float32)
        )
        return labels, mat, resolved

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        proto = self._resolve_proto(model_obj)
        meta = self._inspect_ml_operators(proto)
        width_in = int(np.asarray(X_float32).shape[1]) if np.asarray(X_float32).ndim == 2 else 0
        names = list(meta.feature_names) or [f"feature_{i}" for i in range(width_in)]
        labels, mat, resolved = self._infer_chunks(proto, X_float32, names)
        if meta.is_classifier:
            class_list = list(resolved.classes) if resolved.classes else list(range(int(mat.shape[1])))
            py_preds: list[float | int | str] = []
            for v in labels.tolist():
                match: Any = v
                if isinstance(v, (bool, np.bool_)):
                    match = int(v)
                elif isinstance(v, (np.integer, int)):
                    match = int(v)
                elif isinstance(v, (np.floating, float)):
                    f = float(v)
                    match = int(f) if float(f).is_integer() else f
                elif isinstance(v, bytes):
                    match = v.decode("utf-8", errors="replace")
                elif isinstance(v, np.ndarray):
                    match = v.item() if v.size == 1 else str(v.tolist())
                py_preds.append(match)
            probabilities = [
                [float(v) if np.isfinite(v) else 0.0 for v in row]
                for row in np.asarray(mat, dtype=np.float64).tolist()
            ]
            confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            _ = class_list
            return PredictionBatchResult(
                predictions=py_preds, probabilities=probabilities, confidences=confidences
            )
        vals = np.asarray(mat, dtype=np.float64).ravel()
        py_vals: list[float | int | str] = [float(v) if np.isfinite(v) else 0.0 for v in vals.tolist()]
        return PredictionBatchResult(predictions=py_vals, probabilities=None, confidences=None)

    # -- internal-flow tracing -----------------------------------------
    def _target_resolution(
        self, meta: ONNXClassicalMetadata, target_idx: int, target_class_idx: int | None
    ) -> int:
        n_classes = len(meta.classes) if meta.classes else 2
        if target_class_idx is not None:
            return min(max(int(target_class_idx), 0), max(n_classes - 1, 0))
        return min(max(int(target_idx), 0), max(n_classes - 1, 0))

    def _linear_attribution(
        self,
        w: np.ndarray,
        b: np.ndarray,
        post: str,
        X: np.ndarray,
        resolved_idx: int,
        proba_row: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """Exact Input x Gradient for ``Linear*`` (identity by construction)."""
        x = np.asarray(X, dtype=np.float64).ravel()
        width = int(x.shape[0])
        if w.shape[0] > 1:
            k = min(int(resolved_idx), w.shape[0] - 1)
            wk = np.asarray(w[k], dtype=np.float64)
            bk = float(b[k]) if k < b.size else 0.0
        else:
            wk = np.asarray(w[0], dtype=np.float64)
            bk = float(b[0]) if b.size else 0.0
        if wk.size < width:
            wk = np.pad(wk, (0, width - wk.size))
        wk = wk[:width]
        if post in ("LOGISTIC", "SIGMOID"):
            p = float(proba_row[min(resolved_idx, len(proba_row) - 1)]) if proba_row is not None and len(proba_row) else _stable_sigmoid(float(np.dot(wk, x) + bk))
            grad = p * (1.0 - p) * wk
            pred = p
        elif post in ("SOFTMAX", "SOFTMAX_ZERO"):
            p = np.asarray(proba_row, dtype=np.float64) if proba_row is not None else None
            if p is None or p.size == 0:
                z = w @ x + (b if b.size == w.shape[0] else 0.0)
                z = z - np.max(z)
                e = np.exp(np.clip(z, -60, 60))
                p = e / max(np.sum(e), 1e-12)
            k = min(int(resolved_idx), len(p) - 1)
            pk = float(p[k])
            # Softmax Jacobian row k: J[k, c] = p_k (δ_kc − p_c).
            grad = np.zeros((width,), dtype=np.float64)
            for c in range(w.shape[0]):
                wc = np.asarray(w[c], dtype=np.float64)
                if wc.size < width:
                    wc = np.pad(wc, (0, width - wc.size))
                coef = pk * ((1.0 - pk) if c == k else -float(p[c]))
                grad += coef * wc[:width]
            pred = pk
        else:
            grad = wk.copy()
            pred = float(np.dot(wk, x) + bk)
        attr = grad * x
        attr = np.where(np.isfinite(attr), attr, 0.0)
        base = float(pred - float(np.sum(attr)))
        return np.ascontiguousarray(attr, dtype=np.float32), (base if np.isfinite(base) else 0.0)

    def _tree_attribution(
        self,
        tree: DecompiledTreeEnsemble,
        tree_ids: list[int],
        x_row: np.ndarray,
        final_pred: float,
        resolved_label: Any,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[tuple[int, int]], float]:
        """Walk every tree; allocate ``(final − leaf_prior)`` over path splits."""
        width = int(x_row.shape[0])
        path_all: list[dict[str, Any]] = []
        leaf_keys: list[tuple[int, int]] = []
        leaf_vals: list[float] = []
        split_counts = np.zeros((width,), dtype=np.float64)
        for tid in tree_ids:
            path, leaf_key = _walk_tree(tree, int(tid), np.asarray(x_row, dtype=np.float64))
            if leaf_key is not None:
                leaf_keys.append(leaf_key)
            path_all.extend(path)
            for step in path:
                f = int(step["feature_idx"])
                if 0 <= f < width:
                    split_counts[f] += 1.0
            if leaf_key is not None and leaf_key in tree.leaf_weights:
                weights = tree.leaf_weights[leaf_key]
                if resolved_label in weights:
                    leaf_vals.append(float(weights[resolved_label]))
                elif weights:
                    leaf_vals.append(float(np.mean(list(weights.values()))))
        # Global leaf prior for the resolved class over the whole forest.
        prior_pool: list[float] = []
        for weights in tree.leaf_weights.values():
            if resolved_label in weights:
                prior_pool.append(float(weights[resolved_label]))
            elif weights and not isinstance(resolved_label, (int, float, str)):
                prior_pool.append(float(np.mean(list(weights.values()))))
        base = float(np.mean(prior_pool)) if prior_pool else 0.0
        residual = float(final_pred - base)
        total = float(np.sum(split_counts))
        if total > 0 and np.isfinite(residual):
            attr = (split_counts / total) * residual
        else:
            attr = np.zeros((width,), dtype=np.float64)
        attr = np.where(np.isfinite(attr), attr, 0.0)
        _ = leaf_vals
        return np.ascontiguousarray(attr, dtype=np.float32), path_all, leaf_keys, (base if np.isfinite(base) else 0.0)

    def _svm_attribution(
        self,
        svm: DecompiledSVM,
        x_row: np.ndarray,
        final_pred: float,
        grad_fallback: Any | None,
    ) -> tuple[np.ndarray, float]:
        """Analytic kernel gradient (binary) or session finite diff (multi)."""
        x = np.asarray(x_row, dtype=np.float64).ravel()
        if grad_fallback is None:
            grad = _svm_gradient_binary(x, svm)
            a = float(svm.prob_a[0]) if svm.prob_a else 0.0
            b = float(svm.prob_b[0]) if svm.prob_b else 0.0
            if svm.prob_a or svm.prob_b:
                f = _svm_decision_binary(x, svm)
                p = _stable_sigmoid(a * f + b)
                grad = p * (1.0 - p) * a * grad
        else:
            grad = np.asarray(grad_fallback, dtype=np.float64).ravel()
            if grad.size < x.size:
                grad = np.pad(grad, (0, x.size - grad.size))
            grad = grad[: x.size]
        attr = grad * x
        attr = np.where(np.isfinite(attr), attr, 0.0)
        base = float(final_pred - float(np.sum(attr)))
        return np.ascontiguousarray(attr, dtype=np.float32), (base if np.isfinite(base) else 0.0)

    def _finite_diff_row_gradient(
        self,
        proto: Any,
        feature_names: list[str],
        raw_row: Any,
        numeric_mask: list[bool],
        resolved_idx: int,
        is_classifier: bool,
    ) -> np.ndarray:
        """Session-backed central differences for one row (SVM multiclass,
        preprocessing-prefixed graphs). String columns keep their original
        values and receive zero gradient (recorded in ``non_numeric``)."""
        width = len(feature_names)
        try:
            base_cells = list(np.asarray(raw_row, dtype=object).ravel().tolist())
        except (TypeError, ValueError):
            base_cells = [0.0] * width
        while len(base_cells) < width:
            base_cells.append(0.0)
        base_cells = base_cells[:width]
        x_num = np.zeros((width,), dtype=np.float64)
        for j in range(width):
            try:
                x_num[j] = float(base_cells[j])
            except (TypeError, ValueError):
                x_num[j] = 0.0

        def scalar_at(cells: list[Any]) -> float:
            mat_input = np.empty((1, width), dtype=object)
            mat_input[0, :] = cells
            _, mat, _ = self._infer_chunks(proto, mat_input, feature_names)
            if is_classifier:
                k = min(int(resolved_idx), int(mat.shape[1]) - 1)
                return float(mat[0, k])
            return float(mat[0, 0])

        grad = np.zeros((width,), dtype=np.float64)
        scale = float(np.mean(np.abs(x_num)) + 1.0)
        eps = max(FINITE_DIFF_EPS * scale, 1e-6)
        for j in range(width):
            if not bool(numeric_mask[j]):
                continue
            rp = list(base_cells)
            rm = list(base_cells)
            try:
                center = float(base_cells[j])
            except (TypeError, ValueError):
                continue
            rp[j] = center + eps
            rm[j] = center - eps
            grad[j] = (scalar_at(rp) - scalar_at(rm)) / (2.0 * eps)
        return np.where(np.isfinite(grad), grad, 0.0)

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
        meta = self._inspect_ml_operators(proto)
        width = int(np.asarray(X_float32).shape[1]) if np.asarray(X_float32).ndim == 2 else len(feature_names)
        names = list(feature_names) if len(feature_names) == width else list(meta.feature_names or [f"feature_{i}" for i in range(width)])
        resolved = self._target_resolution(meta, target_idx, target_class_idx)
        _, attrs = _primary_classical_node(proto)

        X_all = np.asarray(X_float32, dtype=object) if np.asanyarray(X_float32).dtype == object else np.asarray(X_float32)
        if X_all.ndim == 1:
            X_all = X_all.reshape(1, -1)
        labels_all, proba_all, _ = self._infer_chunks(proto, X_all, names)

        try:
            label_list = list(np.asarray(labels_all).ravel().tolist())
        except (TypeError, ValueError, AttributeError):
            label_list = []

        if meta.operator_family == "linear":
            w, b, _, post = decompile_linear(attrs, meta.n_features or width)
            _linear_same_space = bool(w.shape[1] == width)
        elif meta.operator_family == "tree_ensemble":
            tree = decompile_tree_ensemble(attrs)
            tree_ids = sorted(set(int(t) for t in tree.tree_ids))
        elif meta.operator_family == "svm":
            svm = decompile_svm(attrs, meta.n_features or width)
        else:  # pragma: no cover - guarded by _resolve_proto
            raise ValueError("Unsupported ONNX classical operator family")

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(list(row_indices)):
            if not (0 <= int(row_id) < int(X_all.shape[0])):
                continue
            rid = int(row_id)
            raw_row = X_all[rid]
            try:
                cells = list(np.asarray(raw_row, dtype=object).ravel().tolist())
            except (TypeError, ValueError):
                cells = [0.0] * width
            while len(cells) < width:
                cells.append(0.0)
            cells = cells[:width]
            numeric_mask: list[bool] = []
            x_num = np.zeros((width,), dtype=np.float64)
            for j in range(width):
                try:
                    x_num[j] = float(cells[j])
                    numeric_mask.append(True)
                except (TypeError, ValueError):
                    x_num[j] = 0.0
                    numeric_mask.append(False)
            x_num = np.where(np.isfinite(x_num), x_num, 0.0)
            non_numeric = [names[j] for j in range(width) if not numeric_mask[j]]

            if meta.is_classifier:
                k = min(int(resolved), int(proba_all.shape[1]) - 1) if proba_all.ndim == 2 else 0
                final_pred = float(proba_all[rid, k]) if proba_all.size else 0.0
                try:
                    final_label: Any = label_list[rid] if rid < len(label_list) else (meta.classes[k] if k < len(meta.classes) else k)
                except (TypeError, ValueError, IndexError):
                    final_label = k
                proba_row = np.asarray(proba_all[rid], dtype=np.float64).ravel() if proba_all.size else None
            else:
                final_pred = float(proba_all[rid, 0]) if proba_all.size else 0.0
                final_label = final_pred
                proba_row = None
            if not np.isfinite(final_pred):
                final_pred = 0.0

            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(meta.is_classifier),
                "is_regressor": bool(meta.is_regressor),
                "algorithm": str(meta.detected_operator),
                "model_family": "onnx_classical",
                "operator_family": str(meta.operator_family),
                "detected_operator": str(meta.detected_operator),
                "detected_operator_family": str(meta.operator_family),
                "domain_badge": f"ONNX ai.onnx.ml | {meta.detected_operator}",
                "node_chain": list(meta.node_chain),
                "input_signature": [dict(s) for s in meta.input_signature],
                "has_zipmap": bool(meta.has_zipmap),
                "zipmap_bypass_active": True,
                "opset_version": int(meta.opset_version),
                "classes": [str(c) for c in meta.classes] if meta.classes else [],
                "class_labels": list(meta.classes) if meta.classes else [],
            }
            if meta.is_classifier:
                extra.update({
                    "probabilities": [float(v) for v in (proba_row.tolist() if proba_row is not None else [])],
                    "predicted_class": final_label if isinstance(final_label, (int, float, str)) else str(final_label),
                    "resolved_class_idx": int(k),
                    "winning_class_idx": int(k),
                })
            else:
                extra.update({"predicted_value": float(final_pred)})

            if meta.operator_family == "linear":
                if _linear_same_space:
                    attr, base = self._linear_attribution(w, b, post, x_num, resolved, proba_row)
                    method = "linear_exact" if post == "NONE" else "input_x_gradient"
                else:
                    # Preprocessing prefix (Scaler/OneHotEncoder) between raw
                    # inputs and Linear*: attribute end-to-end via session
                    # finite differences so base + sum == prediction holds in
                    # raw units (§2.2 multi-input graphs).
                    grad = self._finite_diff_row_gradient(proto, names, cells, numeric_mask, resolved, meta.is_classifier)
                    attr_raw = grad * x_num
                    attr_raw = np.where(np.isfinite(attr_raw), attr_raw, 0.0)
                    attr = np.ascontiguousarray(attr_raw, dtype=np.float32)
                    base = float(final_pred - float(np.sum(attr_raw)))
                    base = base if np.isfinite(base) else 0.0
                    method = "input_x_gradient_session"
                extra.update({
                    "attribution_method": method,
                    "post_transform": post,
                    "linear_same_space": bool(_linear_same_space),
                    "coefficients": [round(float(v), 5) for v in (np.asarray(w[min(resolved, w.shape[0] - 1)]).tolist() if w.shape[0] > 1 else np.asarray(w[0]).tolist())][:16],
                    "intercept": round(float(b[min(resolved, b.size - 1)] if b.size else 0.0), 5),
                })
            elif meta.operator_family == "tree_ensemble":
                _tree_same_space = True
                try:
                    _max_fid = max((int(f) for pos, f in enumerate(tree.feature_ids) if str(tree.modes[pos]) != "LEAF"), default=-1)
                    _tree_same_space = bool(_max_fid < width)
                except (TypeError, ValueError, IndexError):
                    _tree_same_space = False
                if not _tree_same_space or any(not m for m in numeric_mask):
                    grad = self._finite_diff_row_gradient(proto, names, cells, numeric_mask, resolved, meta.is_classifier)
                    attr_raw = grad * x_num
                    attr_raw = np.where(np.isfinite(attr_raw), attr_raw, 0.0)
                    attr = np.ascontiguousarray(attr_raw, dtype=np.float32)
                    base = float(final_pred - float(np.sum(attr_raw)))
                    base = base if np.isfinite(base) else 0.0
                    extra.update({
                        "attribution_method": "input_x_gradient_session",
                        "n_trees": int(tree.n_trees),
                        "n_nodes": int(tree.n_nodes),
                        "post_transform": str(tree.post_transform),
                        "tree_same_space": False,
                        "non_numeric_columns": list(non_numeric),
                    })
                if _tree_same_space and all(numeric_mask):
                    if meta.is_classifier:
                        try:
                            res_label = meta.classes[k] if k < len(meta.classes) else k
                        except (TypeError, IndexError):
                            res_label = k
                        # ZipMap int64 labels arrive as python ints from the session.
                        try:
                            lab_raw = label_list[rid] if rid < len(label_list) else res_label
                            for cand in meta.classes:
                                if str(cand) == str(lab_raw):
                                    res_label = cand
                                    break
                        except (TypeError, ValueError, IndexError):
                            pass
                    else:
                        res_label = 0
                    attr, path_all, leaf_keys, base = self._tree_attribution(tree, tree_ids, x_num, final_pred, res_label)
                    shown = path_all[:64]
                    extra.update({
                        "attribution_method": "tree_path_share",
                        "n_trees": int(tree.n_trees),
                        "n_nodes": int(tree.n_nodes),
                        "post_transform": str(tree.post_transform),
                        "tree_same_space": True,
                        "decision_path": shown,
                        "decision_path_truncated": bool(len(path_all) > len(shown)),
                        "n_path_steps": int(len(path_all)),
                        "active_leaf_count": int(len(leaf_keys)),
                        "leaf_trace": [
                            {"tree_id": int(t), "node_id": int(n), "weight": round(float(tree.leaf_weights.get((int(t), int(n)), {}).get(res_label, 0.0)), 5)}
                            for (t, n) in leaf_keys[:64]
                        ],
                    })
            else:  # svm
                multiclass = bool(meta.is_classifier and len(meta.classes) > 2)
                same_space = bool(svm.support_vectors.shape[1] == width) if svm.n_support else True
                grad_fb = None
                if multiclass or not same_space or any(not m for m in numeric_mask):
                    grad_fb = self._finite_diff_row_gradient(proto, names, cells, numeric_mask, resolved, True)
                attr, base = self._svm_attribution(svm, x_num, final_pred, grad_fb)
                extra.update({
                    "attribution_method": "svm_kernel_gradient" if not multiclass else "svm_finite_difference",
                    "kernel_type": str(svm.kernel_type),
                    "kernel_params": [round(float(v), 5) for v in svm.kernel_params],
                    "n_support_vectors": int(svm.n_support),
                    "support_vector_preview": [[round(float(v), 4) for v in row[:8]] for row in svm.support_vectors[:4].tolist()] if svm.n_support else [],
                })

            order = np.argsort(np.abs(np.asarray(attr, dtype=np.float64)))[::-1][: min(8, width)] if width else np.asarray([], dtype=int)
            stages = [{
                "rank": int(r + 1),
                "feature_idx": int(f),
                "feature_name": names[int(f)] if int(f) < len(names) else f"feature_{int(f)}",
                "shap_value": round(float(np.asarray(attr, dtype=np.float64)[int(f)]), 5),
            } for r, f in enumerate(np.asarray(order).tolist())]
            extra["top_shap_features"] = stages
            extra.setdefault("non_numeric_columns", list(non_numeric))
            extra["base_raw_score"] = float(base)
            extra["final_raw_score"] = float(final_pred)
            try:
                has_nan = bool(np.isnan(np.asarray(X_float32[rid], dtype=np.float64)).any())
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
            _ = pos
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
        meta = self._inspect_ml_operators(proto)
        width = max(int(n_features), meta.n_features or 0, 1)
        names = list(meta.feature_names) if len(meta.feature_names) == width else [f"feature_{i}" for i in range(width)]
        resolved = self._target_resolution(meta, target_idx, target_class_idx)
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
        grid = np.where(np.isfinite(grid), grid, 0.0)

        resolved_session = self._build_optimized_session(proto)
        sess = resolved_session.session
        z_flat = np.zeros((res * res,), dtype=np.float64)
        for start in range(0, res * res, PREDICT_CHUNK_SIZE):
            chunk = grid[start:start + PREDICT_CHUNK_SIZE]
            feed = self._build_feed_dict(sess, chunk, names)
            out_names = [o.name for o in sess.get_outputs()]
            raw = sess.run(out_names, feed)
            by_name = {n: r for n, r in zip(out_names, raw)}
            if meta.is_classifier:
                prob = by_name.get(resolved_session.proba_output or "", None)
                if isinstance(prob, list):
                    prob = unpack_zipmap_sequence(prob, resolved_session.classes)
                mat = np.asarray(prob, dtype=np.float64)
                if mat.ndim == 1:
                    mat = mat.reshape(-1, 1)
                k = min(int(resolved), int(mat.shape[1]) - 1)
                z_flat[start:start + chunk.shape[0]] = np.asarray(mat[:, k], dtype=np.float64).ravel()
            else:
                val = by_name.get(resolved_session.proba_output or out_names[0], None)
                z_flat[start:start + chunk.shape[0]] = np.asarray(val, dtype=np.float64).ravel()
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        z_surface = z_flat.reshape(res, res)
        if meta.is_classifier:
            z_min, z_max = 0.0, 1.0
        else:
            z_min, z_max = (float(np.min(z_surface)), float(np.max(z_surface))) if z_surface.size else (0.0, 0.0)

        if meta.operator_family == "tree_ensemble":
            visual = Visual3DType.STEPPED_AXIS_BLOCKS
        elif meta.operator_family == "svm":
            visual = Visual3DType.SVM_MARGIN_TUBE
        elif meta.is_classifier:
            visual = Visual3DType.PROBABILITY_SIGMOID_SURFACE
        else:
            visual = Visual3DType.FLAT_HYPERPLANE

        overlays: dict[str, Any] = {
            "model_family": "onnx_classical",
            "operator_family": str(meta.operator_family),
            "detected_operator": str(meta.detected_operator),
            "detected_operator_family": str(meta.operator_family),
            "domain_badge": f"ONNX ai.onnx.ml | {meta.detected_operator}",
            "node_chain": list(meta.node_chain),
            "input_signature": [dict(s) for s in meta.input_signature],
            "has_zipmap": bool(meta.has_zipmap),
            "zipmap_bypass_active": bool(resolved_session.bypass_active),
            "opset_version": int(meta.opset_version),
            "target_idx": int(resolved),
            "resolved_target_class_idx": int(resolved),
            "classes": [str(c) for c in meta.classes] if meta.classes else [],
            "is_classifier": bool(meta.is_classifier),
            "is_regressor": bool(meta.is_regressor),
            "is_binary": bool(meta.is_classifier and len(meta.classes) == 2),
            "num_class": int(len(meta.classes)) if meta.is_classifier else 0,
            "regression_z_range": [float(z_min), float(z_max)],
            "boundary_threshold": 0.5 if meta.is_classifier else None,
        }
        try:
            _, attrs = _primary_classical_node(proto)
            if meta.operator_family == "tree_ensemble":
                tree = decompile_tree_ensemble(attrs)
                overlays.update({
                    "n_trees": int(tree.n_trees),
                    "n_nodes": int(tree.n_nodes),
                    "post_transform": str(tree.post_transform),
                    "split_thresholds": self.decompile_tree_ensemble_splits(proto, ax, ay),
                })
            elif meta.operator_family == "linear":
                w, b, multi, post = decompile_linear(attrs, meta.n_features or width)
                overlays.update({
                    "weight_shape": [int(v) for v in w.shape],
                    "coefficients_preview": [round(float(v), 5) for v in np.asarray(w[0]).tolist()[:16]],
                    "intercept": round(float(b[0]) if b.size else 0.0, 5),
                    "post_transform": post,
                    "multi_class": multi,
                })
            elif meta.operator_family == "svm":
                svm = decompile_svm(attrs, meta.n_features or width)
                overlays.update({
                    "kernel_type": str(svm.kernel_type),
                    "kernel_params": [round(float(v), 5) for v in svm.kernel_params],
                    "n_support_vectors": int(svm.n_support),
                })
        except (ValueError, TypeError, AttributeError):
            pass
        return SurfaceMesh3DData(
            visual_type=visual,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in np.asarray(z_surface, dtype=np.float64).ravel()],
            extra_overlays=overlays,
        )


@ModelRegistry.register("onnx_classical")
class ONNXClassicalClassifierAdapter(_ONNXClassicalBase):
    display_name = "ONNX Classical ML (ai.onnx.ml Inspector)"
    family_name = "ONNX Classical ML"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            if not is_onnx_model_proto(estimator):
                return False
            meta = inspect_onnx_classical(estimator)
            return bool(meta.operator_family in ("tree_ensemble", "linear", "svm") and meta.is_classifier)
        except (ValueError, TypeError, AttributeError):
            return False

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return self._trace_common(model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx)

    def _trace_common(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return _ONNXClassicalBase.trace_internal_flow(
            self, model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return super().compute_3d_surface(model_obj, grid_spec, n_features, target_class_idx, target_idx)


class ONNXClassicalAdapter(_ONNXClassicalBase):
    """Spec facade: ``ONNXClassicalAdapter(model_path=..., onnx_proto=...)``.

    Delegates to the classifier / regressor subclass by inspected operator,
    mirroring the ``MLPAdapter`` facade from Phase 36.
    """

    display_name = "ONNX Classical ML (ai.onnx.ml Inspector)"
    family_name = "ONNX Classical ML"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            if not is_onnx_model_proto(estimator):
                return False
            meta = inspect_onnx_classical(estimator)
            return bool(meta.operator_family in ("tree_ensemble", "linear", "svm"))
        except (ValueError, TypeError, AttributeError):
            return False

    def _delegate_for(self, model_obj: Any | None) -> _ONNXClassicalBase:
        target = model_obj if model_obj is not None else self._bound_proto
        try:
            meta = inspect_onnx_classical(target) if target is not None else None
        except (ValueError, TypeError, AttributeError):
            meta = None
        if meta is not None and meta.is_regressor:
            delegate = ONNXClassicalRegressorAdapter.__new__(ONNXClassicalRegressorAdapter)
            delegate._bound_path = self._bound_path
            delegate._bound_proto = self._bound_proto
            return delegate
        delegate = ONNXClassicalClassifierAdapter.__new__(ONNXClassicalClassifierAdapter)
        delegate._bound_path = self._bound_path
        delegate._bound_proto = self._bound_proto
        return delegate

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        target = model_obj if model_obj is not None else self._bound_proto
        return self._delegate_for(target).trace_internal_flow(
            target, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        target = model_obj if model_obj is not None else self._bound_proto
        return self._delegate_for(target).compute_3d_surface(
            target, grid_spec, n_features, target_class_idx, target_idx
        )


@ModelRegistry.register("onnx_classical_regressor")
class ONNXClassicalRegressorAdapter(_ONNXClassicalBase):
    display_name = "ONNX Classical ML Regressor (ai.onnx.ml Inspector)"
    family_name = "ONNX Classical ML"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            if not is_onnx_model_proto(estimator):
                return False
            meta = inspect_onnx_classical(estimator)
            return bool(meta.operator_family in ("tree_ensemble", "linear", "svm") and meta.is_regressor)
        except (ValueError, TypeError, AttributeError):
            return False

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        return _ONNXClassicalBase.trace_internal_flow(
            self, model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return super().compute_3d_surface(model_obj, grid_spec, n_features, target_class_idx, target_idx)

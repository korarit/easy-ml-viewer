# backend/app/domains/model_engine/adapters/neural_onnx/onnx_tabular.py
"""Generic tabular ONNX adapter — KNN / NaiveBayes / preprocessing chains.

Covers ``.onnx`` graphs that ``skl2onnx`` emits for estimators with NO
dedicated ``ai.onnx.ml`` estimator op:

- ``KNeighborsClassifier/Regressor`` → pure ``ai.onnx`` distance graph
  (``Sub/Mul/ReduceSum/Sqrt/TopK/Gather`` + ``ArrayFeatureExtractor`` + ``ZipMap``).
- ``BernoulliNB/MultinomialNB/...`` → ``MatMul/ReduceLogSumExp/ArgMax`` graph.
- Preprocessing-only chains (``Scaler/Normalizer/Imputer/OneHotEncoder`` …
  plus a downstream head the classical scanner does not recognise).

Image / sequence / transformer graphs (``Conv/Pool/LSTM/Attention`` …,
non-2D inputs) are explicitly NOT claimed — they stay
``UNSUPPORTED_MODEL_TYPE`` instead of being misreported as tabular.

Inference is exact (native ORT session with the shared ZipMap bypass);
attribution is session finite-difference Input x Gradient, flagged
``generic_fallback: True``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.domains.model_engine.adapters.neural_onnx.onnx_classical import (
    build_feed_dict,
    build_optimized_session,
    is_onnx_model_proto,
    unpack_zipmap_sequence,
)
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

TABULAR_FAMILIES = frozenset({"knn", "naive_bayes", "tabular_preprocessing", "tabular_generic"})

# Ops that prove the graph is NOT tabular (vision / sequence / generative).
_NON_TABULAR_OPS = frozenset(
    {
        "Conv",
        "ConvTranspose",
        "QLinearConv",
        "MaxPool",
        "AveragePool",
        "GlobalMaxPool",
        "GlobalAveragePool",
        "LSTM",
        "GRU",
        "RNN",
        "Attention",
        "MultiHeadAttention",
        "EmbedLayerNormalization",
        "SkipLayerNormalization",
        "BeamSearch",
        "GreedySearch",
        "Sampling",
        "Whisper",
        "ImageScaler",
        "Crop",
        "Resize",
    }
)

# skl2onnx preprocessing ops (ai.onnx.ml domain) tolerated as a prefix.
_PRE_OPS = frozenset(
    {
        "Scaler",
        "Normalizer",
        "Imputer",
        "OneHotEncoder",
        "LabelEncoder",
        "Binarizer",
        "FeatureVectorizer",
        "ArrayFeatureExtractor",
    }
)


@dataclass
class ONNXTabularMetadata:
    """Graph summary for generic tabular ONNX (JSON-safe via to_dict)."""

    detected_operator: str = "unknown"
    operator_family: str = "unknown"  # knn | naive_bayes | tabular_preprocessing | tabular_generic | unknown
    is_classifier: bool = False
    is_regressor: bool = False
    classes: list[Any] = field(default_factory=list)
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    node_chain: list[str] = field(default_factory=list)
    has_zipmap: bool = False
    opset_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_operator": self.detected_operator,
            "operator_family": self.operator_family,
            "is_classifier": bool(self.is_classifier),
            "is_regressor": bool(self.is_regressor),
            "classes": list(self.classes),
            "n_features": int(self.n_features),
            "feature_names": list(self.feature_names),
            "node_chain": list(self.node_chain),
            "has_zipmap": bool(self.has_zipmap),
            "opset_version": int(self.opset_version),
        }


def _ops(proto: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for node in list(getattr(getattr(proto, "graph", None), "node", [])):
        out.append((str(getattr(node, "domain", "") or "ai.onnx"), str(getattr(node, "op_type", ""))))
    return out


def _zipmap_labels(proto: Any) -> list[Any] | None:
    for node in list(getattr(getattr(proto, "graph", None), "node", [])):
        if str(getattr(node, "domain", "")) == "ai.onnx.ml" and str(node.op_type) == "ZipMap":
            for attr in list(getattr(node, "attribute", [])):
                name = str(getattr(attr, "name", ""))
                if name in ("classlabels_int64s", "classlabels_ints", "class_ids"):
                    try:
                        return [int(v) for v in list(getattr(attr, "ints", []))]
                    except (TypeError, ValueError):
                        pass
                if name in ("classlabels_strings", "classlabels_strs"):
                    try:
                        vals = []
                        for v in list(getattr(attr, "strings", [])):
                            vals.append(v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v))
                        return vals
                    except (TypeError, ValueError):
                        pass
    return None


def _input_width(proto: Any) -> tuple[int, list[str]]:
    """Return ``(n_features, feature_names)`` for 2D tabular inputs, else ``(0, [])``."""
    try:
        inputs = list(proto.graph.input)
    except AttributeError:
        return 0, []
    if len(inputs) == 1:
        try:
            dims = inputs[0].type.tensor_type.shape.dim
            shape = [int(getattr(d, "dim_value", 0) or 0) or None for d in dims]
        except (AttributeError, TypeError, ValueError):
            return 0, []
        if len(shape) == 2 and isinstance(shape[1], int) and shape[1] >= 1:
            try:
                from onnx import TensorProto  # noqa: PLC0415

                elem = int(inputs[0].type.tensor_type.elem_type)
                if elem not in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT64, TensorProto.INT32):
                    return 0, []
            except ImportError:
                pass
            width = int(shape[1])
            return width, [f"feature_{i}" for i in range(width)]
        return 0, []
    if len(inputs) > 1:
        # Multi-column named inputs (DataFrame pipelines: one [N,1] per column).
        names = [str(i.name) for i in inputs]
        return len(inputs), names
    return 0, []


def inspect_onnx_tabular(proto: Any) -> ONNXTabularMetadata:
    """Classify a non-classical, non-deep ONNX graph as tabular or unknown."""
    meta = ONNXTabularMetadata()
    if not is_onnx_model_proto(proto):
        return meta
    ops = _ops(proto)
    op_types = [op for _, op in ops]
    meta.node_chain = [f"{d}::{o}" for d, o in ops]
    try:
        for imp in list(getattr(proto, "opset_import", [])):
            if str(getattr(imp, "domain", "")) in ("", "ai.onnx"):
                meta.opset_version = int(getattr(imp, "version", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        pass

    # Classical estimator graphs belong to onnx_classical.
    for domain, op in ops:
        if domain == "ai.onnx.ml" and op in (
            "TreeEnsembleClassifier",
            "TreeEnsembleRegressor",
            "TreeEnsemble",
            "LinearClassifier",
            "LinearRegressor",
            "SVMClassifier",
            "SVMRegressor",
        ):
            return meta
    # Vision / sequence / generative graphs are out of scope.
    if any(op in _NON_TABULAR_OPS for op in op_types):
        return meta
    # Deep Gemm/MatMul nets belong to onnx_deep.
    try:
        from app.domains.model_engine.adapters.neural_onnx.onnx_deep import (  # noqa: PLC0415
            is_deep_learning_proto,
        )

        if bool(is_deep_learning_proto(proto)):
            return meta
    except (ImportError, ValueError, TypeError, AttributeError):
        pass

    width, names = _input_width(proto)
    if width < 1:
        return meta

    labels = _zipmap_labels(proto)
    # Output-kind heuristic: sequence outputs / (label + proba) pairs are classifiers.
    n_outputs = len(list(getattr(getattr(proto, "graph", None), "output", [])))
    is_classifier = labels is not None or n_outputs >= 2
    # Single-output graphs with an ArgMax/TopK head are classifiers even without ZipMap.
    if not is_classifier and any(op in ("ArgMax", "TopK") for op in op_types):
        is_classifier = True
        n_outputs = max(n_outputs, 2)

    if any(op == "TopK" for op in op_types):
        meta.operator_family = "knn"
        meta.detected_operator = "KNN (TopK distance graph)"
    elif any(op in ("ReduceLogSumExp", "ReduceSumSquare") for op in op_types) and any(
        op in ("ArgMax", "MatMul", "Gemm") for op in op_types
    ):
        meta.operator_family = "naive_bayes"
        meta.detected_operator = "NaiveBayes (log-likelihood graph)"
    elif any(domain == "ai.onnx.ml" and op in _PRE_OPS for domain, op in ops):
        meta.operator_family = "tabular_preprocessing"
        head = next((op for _, op in ops if op not in _PRE_OPS and op not in ("ZipMap", "Identity", "Cast")), "Preprocessing")
        meta.detected_operator = f"Tabular chain (-> {head})"
    elif labels is not None or is_classifier:
        meta.operator_family = "tabular_generic"
        head = next((op for _, op in ops if op not in ("ZipMap", "Identity", "Cast")), "unknown")
        meta.detected_operator = f"TabularGeneric ({head})"
    else:
        # Single float output, no classifier markers → generic regressor only
        # when the graph is small/pure-tabular (no exotic control flow).
        meta.operator_family = "tabular_generic"
        head = next((op for _, op in ops if op not in ("Identity", "Cast")), "unknown")
        meta.detected_operator = f"TabularGeneric ({head})"

    meta.n_features = int(width)
    meta.feature_names = list(names)
    meta.has_zipmap = labels is not None or any(op == "ZipMap" for op in op_types)
    meta.is_classifier = bool(is_classifier)
    meta.is_regressor = bool(not is_classifier)
    meta.classes = list(labels) if labels is not None else []
    return meta


def _run_session_proba(proto: Any, X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray | None, list[Any], bool]:
    """Run ORT session → ``(labels, proba_or_None, classes, is_classifier)``."""
    resolved = build_optimized_session(proto)
    sess = resolved.session
    classes = list(resolved.classes)
    X_c = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    labels_all: list[Any] = []
    proba_all: list[Any] = []
    for start in range(0, X_c.shape[0], PREDICT_CHUNK_SIZE):
        chunk = np.ascontiguousarray(X_c[start : start + PREDICT_CHUNK_SIZE])
        feed = build_feed_dict(sess, chunk, feature_names)
        try:
            outs = sess.run(None, feed)
        except Exception:
            outs = sess.run([o.name for o in sess.get_outputs()], feed)
        if resolved.is_classifier and resolved.label_output is not None:
            out_names = [str(o.name) for o in sess.get_outputs()]
            try:
                li = out_names.index(str(resolved.label_output))
                labels_all.append(np.asarray(outs[li]).ravel())
            except (ValueError, IndexError):
                labels_all.append(np.asarray(outs[0]).ravel())
            if resolved.proba_output is not None:
                try:
                    pi = out_names.index(str(resolved.proba_output))
                    raw = outs[pi]
                except (ValueError, IndexError):
                    raw = outs[1] if len(outs) > 1 else outs[0]
                if isinstance(raw, list):
                    raw = unpack_zipmap_sequence(raw, classes)
                proba_all.append(np.asarray(raw, dtype=np.float64))
        else:
            # Regressor (or unlabeled head): first float output.
            raw = outs[0]
            if isinstance(raw, list):
                raw = unpack_zipmap_sequence(raw, classes)
            proba_all.append(np.asarray(raw, dtype=np.float64))
    labels = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((X_c.shape[0],), dtype=np.int64)
    proba = None
    if proba_all:
        stacked = [np.asarray(a) for a in proba_all]
        try:
            proba = np.concatenate([a.reshape(a.shape[0], -1) for a in stacked], axis=0)
        except ValueError:
            proba = None
    if proba is not None:
        proba = np.where(np.isfinite(proba), proba, 0.0)
    return labels, proba, classes, bool(resolved.is_classifier)


class _ONNXTabularBase(BaseModelAdapter):
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        return [None for _ in range(max(int(n_features), 0))]

    def _score_vector(self, proto: Any, X: np.ndarray, feature_names: list[str], class_idx: int) -> np.ndarray:
        """Scalar surface score: winning-class proba (classifier) or raw output."""
        _, proba, _, is_clf = _run_session_proba(proto, X, feature_names)
        if proba is None:
            return np.zeros((X.shape[0],), dtype=np.float64)
        mat = np.asarray(proba, dtype=np.float64)
        if mat.ndim == 1:
            return mat
        j = int(min(max(class_idx, 0), mat.shape[1] - 1))
        return mat[:, j]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        X = np.ascontiguousarray(np.asarray(X_float32, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        meta = inspect_onnx_tabular(model_obj)
        names = list(meta.feature_names) or [f"feature_{i}" for i in range(X.shape[1])]
        labels, proba, classes, is_clf = _run_session_proba(model_obj, X, names)
        preds: list[float | int | str] = []
        for v in list(np.asarray(labels).ravel().tolist()):
            if isinstance(v, bytes):
                preds.append(v.decode("utf-8", errors="replace"))
            elif isinstance(v, (bool, np.bool_)):
                preds.append(int(v))
            elif isinstance(v, (np.integer, int)):
                preds.append(int(v))
            elif isinstance(v, (np.floating, float)):
                f = float(v)
                preds.append(int(f) if f.is_integer() else f)
            else:
                preds.append(v)
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if proba is not None and np.asarray(proba).size:
            mat = np.asarray(proba, dtype=np.float64)
            if mat.ndim == 1:
                mat = mat.reshape(-1, 1)
            probabilities = [[float(v) for v in row] for row in mat.tolist()]
            if is_clf:
                confidences = [float(max(row)) if row else 0.0 for row in probabilities]
        if not is_clf and proba is not None:
            # Regressor: predictions are the raw numeric outputs.
            mat = np.asarray(proba, dtype=np.float64)
            if mat.ndim == 2 and mat.shape[1] == 1:
                preds = [float(v) for v in mat.ravel().tolist()]
        return PredictionBatchResult(predictions=preds, probabilities=probabilities, confidences=confidences)

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        from app.domains.model_engine.adapters.neural_onnx.onnx_classical import (  # noqa: PLC0415
            inspect_onnx_classical,
        )

        X = np.ascontiguousarray(np.asarray(X_float32, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        meta = inspect_onnx_tabular(model_obj)
        names = feature_names or list(meta.feature_names) or [f"feature_{i}" for i in range(X.shape[1])]
        n_features = X.shape[1]
        try:
            classical = inspect_onnx_classical(model_obj)
            classical_chain = list(classical.node_chain)
        except (ValueError, TypeError, AttributeError):
            classical_chain = []
        resolved = int(target_class_idx) if target_class_idx is not None else int(target_idx)
        try:
            labels, proba, classes, is_clf = _run_session_proba(model_obj, X, names)
            base_scores = self._score_vector(model_obj, X, names, resolved)
        except Exception:
            labels, proba, classes, is_clf, base_scores = (
                np.zeros((X.shape[0],)),
                None,
                list(meta.classes),
                bool(meta.is_classifier),
                np.zeros((X.shape[0],)),
            )
        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < X.shape[0]):
                continue
            x_row = X[row_id].astype(np.float64)
            try:
                pred_raw = list(np.asarray(labels).ravel().tolist())[row_id]
            except (IndexError, TypeError):
                pred_raw = 0
            if isinstance(pred_raw, bytes):
                pred: float | int | str = pred_raw.decode("utf-8", errors="replace")
            elif isinstance(pred_raw, (bool, np.bool_)):
                pred = int(pred_raw)
            elif isinstance(pred_raw, (np.integer, int)):
                pred = int(pred_raw)
            elif isinstance(pred_raw, (np.floating, float)):
                f = float(pred_raw)
                pred = int(f) if f.is_integer() else f
            else:
                pred = pred_raw
            score = float(base_scores[row_id]) if row_id < len(base_scores) else 0.0
            # Finite-difference Input x Gradient around the row itself.
            contribs: dict[str, float] = {}
            ok = False
            try:
                grads = np.zeros((n_features,), dtype=np.float64)
                for j in range(n_features):
                    scale = max(abs(float(x_row[j])), 1.0)
                    eps = float(FINITE_DIFF_EPS * scale)
                    x_up = np.ascontiguousarray(X[row_id : row_id + 1].astype(np.float64))
                    x_up[0, j] += eps
                    x_dn = np.ascontiguousarray(X[row_id : row_id + 1].astype(np.float64))
                    x_dn[0, j] -= eps
                    s_up = float(self._score_vector(model_obj, x_up.astype(np.float32), names, resolved)[0])
                    s_dn = float(self._score_vector(model_obj, x_dn.astype(np.float32), names, resolved)[0])
                    grads[j] = (s_up - s_dn) / (2.0 * eps)
                raw = grads * x_row[:n_features]
                raw = np.where(np.isfinite(raw), raw, 0.0)
                base = float(score - raw.sum())
                contribs = {names[i]: float(raw[i]) for i in range(min(len(names), n_features))}
                ok = True
            except Exception:
                ok = False
            if not ok or not contribs:
                base = 0.0
                share = float(score) / max(n_features, 1) if n_features else 0.0
                contribs = {name: share for name in names} if names else {"feature_0": 0.0}
            prob_row: list[float] = []
            try:
                if proba is not None:
                    mat = np.asarray(proba)
                    prob_row = [float(v) for v in list(mat[row_id].ravel().tolist())] if mat.ndim == 2 else [float(mat[row_id])]
            except (IndexError, TypeError, ValueError):
                prob_row = []
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base),
                    final_prediction=pred,  # type: ignore[arg-type]
                    feature_contributions={k: float(v) for k, v in contribs.items()},
                    extra_mechanics={
                        "target_idx": int(resolved),
                        "target_class_idx": int(resolved),
                        "is_classifier": bool(is_clf),
                        "classes": [str(c) for c in (classes or meta.classes)],
                        "class_labels": list(classes or meta.classes),
                        "probabilities": prob_row,
                        "has_native_proba": bool(proba is not None),
                        "is_pseudo_proba": False,
                        "detected_operator": meta.detected_operator,
                        "operator_family": meta.operator_family,
                        "node_chain": list(meta.node_chain) or list(classical_chain),
                        "generic_fallback": True,
                    },
                )
            )
        return traces

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        meta = inspect_onnx_tabular(model_obj)
        names = list(meta.feature_names) or [f"feature_{i}" for i in range(max(int(n_features), 1))]
        resolved = int(target_class_idx) if target_class_idx is not None else int(target_idx)
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        grid = np.zeros((res * res, max(int(n_features), 1)), dtype=np.float32)
        for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
            if 0 <= feat_idx < grid.shape[1] and np.isfinite(med_val):
                grid[:, feat_idx] = np.float32(med_val)
        ax, ay = grid_spec.axis_x_feature_index, grid_spec.axis_y_feature_index
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
        if 0 <= ax < grid.shape[1]:
            grid[:, ax] = xx.ravel()
        if int(n_features) > 1 and ay != ax and 0 <= ay < grid.shape[1]:
            grid[:, ay] = yy.ravel()
        try:
            z = np.asarray(self._score_vector(model_obj, grid, names, resolved), dtype=np.float64).ravel()
        except Exception:
            z = np.zeros((grid.shape[0],), dtype=np.float64)
        z = np.where(np.isfinite(z), z, 0.0)
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in z.tolist()],
            extra_overlays={
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "detected_operator": meta.detected_operator,
                "operator_family": meta.operator_family,
                "generic_fallback": True,
                "boundary_threshold": 0.5,
                "is_classifier": bool(meta.is_classifier),
            },
        )


@ModelRegistry.register("onnx_tabular_classifier")
class ONNXTabularClassifierAdapter(_ONNXTabularBase):
    display_name = "ONNX Tabular ML (Generic Inspector)"
    family_name = "ONNX Tabular ML"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            if not is_onnx_model_proto(estimator):
                return False
            meta = inspect_onnx_tabular(estimator)
            return bool(meta.operator_family in TABULAR_FAMILIES and meta.is_classifier)
        except (ValueError, TypeError, AttributeError):
            return False


@ModelRegistry.register("onnx_tabular_regressor")
class ONNXTabularRegressorAdapter(_ONNXTabularBase):
    display_name = "ONNX Tabular ML Regressor (Generic Inspector)"
    family_name = "ONNX Tabular ML"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".onnx") -> bool:
        try:
            if not is_onnx_model_proto(estimator):
                return False
            meta = inspect_onnx_tabular(estimator)
            return bool(meta.operator_family in TABULAR_FAMILIES and meta.is_regressor)
        except (ValueError, TypeError, AttributeError):
            return False


def _resolve_working_copy(proto: Any) -> Any:
    """Deep copy used by session builders that rewire graph outputs."""
    return copy.deepcopy(proto)

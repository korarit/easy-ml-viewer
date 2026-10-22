# backend/app/domains/model_engine/adapters/neural_onnx/mlp_adapter.py
"""Phase 36 — Multi-Layer Perceptron synapse-brain adapters (exact forward pass).

Covers ``MLPClassifier`` / ``MLPRegressor`` (``.pkl`` / ``.joblib``) with
``Pipeline`` / ``GridSearchCV``-style wrapper unwrapping — inference runs
through the full container while staged tracing unwraps to the fitted MLP.

Blueprint mapping (phase-36-model-mlp-neural-network.md):

- Exact layer-by-layer forward pass ``Z^(l) = A^(l-1) W^(l) + b^(l)`` and
  ``A^(l) = sigma_l(Z^(l))`` over ``coefs_`` / ``intercepts_`` lists (§2.1);
  hidden ``sigma`` follows ``model.activation`` (``relu``/``tanh``/
  ``logistic``/``identity``), the output layer follows ``out_activation_``
  (``softmax``/``logistic``/``identity``), overflow-guarded via stable
  ``expit`` / log-sum-exp ``softmax`` (edge #2).
- Dynamic Top-32 neuron downsampling per hidden layer by energy
  ``E_j = mean|A_j| * max_k |W_next[j,k]|`` (§2.2) capped at
  ``MAX_RENDERED_NEURONS_PER_LAYER = 32``; synapse flux
  ``F = mean(A_prev_u) * W[u,v]`` restricted to rendered pairs and capped
  at ``MAX_SYNAPSES_PER_LAYER_PAIR = 128`` (edge #1).
- Dead ReLU detection: ``DeadRatio_j = mean(A_j <= 0)`` over the chunk;
  ``== 1.0`` marks a dead neuron, exposed as full-width boolean masks
  plus per-layer telemetry (§2.3).
- Analytical Input x Gradient attribution via chain-rule Jacobian
  backpropagation in ``<= 512``-row sub-chunks (§6.3); waterfall identity
  ``base + sum(contribs) == prediction`` holds by construction
  (``base = pred - sum(attrib)``).
"""

from __future__ import annotations

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

try:  # Phase 36: sklearn is required, but keep registry importable.
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    _MLP_AVAILABLE = True
    _CLASSIFIER_TYPES = (MLPClassifier,)
    _REGRESSOR_TYPES = (MLPRegressor,)
except ImportError:  # pragma: no cover - exercised only without sklearn
    MLPClassifier = None  # type: ignore[assignment,misc]
    MLPRegressor = None  # type: ignore[assignment,misc]
    _MLP_AVAILABLE = False
    _CLASSIFIER_TYPES = ()
    _REGRESSOR_TYPES = ()

MAX_RENDERED_NEURONS_PER_LAYER: int = 32
MAX_SYNAPSES_PER_LAYER_PAIR: int = 128
MAX_SURFACE_RESOLUTION: int = 100
GRAD_CHUNK_ROWS: int = 512


@dataclass
class NeuralForwardTrace:
    """Exact forward-pass snapshot for one chunk (NumPy-backed).

    ``layer_activations`` holds downsampled ``(N, <=32)`` float32 arrays for
    hidden layers (input/output layers kept full-width); ``dead_neuron_masks``
    are full-width boolean arrays so caller indices stay in original space.
    """

    layer_dimensions: list[int] = field(default_factory=list)
    rendered_node_indices: list[list[int]] = field(default_factory=list)
    layer_activations: list[np.ndarray] = field(default_factory=list)
    full_layer_activations: list[np.ndarray] = field(default_factory=list)
    output_activations: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    dead_neuron_masks: list[np.ndarray] = field(default_factory=list)
    dead_ratios: list[np.ndarray] = field(default_factory=list)
    synapse_edges_per_layer: list[list[dict[str, Any]]] = field(default_factory=list)
    layer_telemetry: list[dict[str, Any]] = field(default_factory=list)
    activation: str = "relu"
    out_activation: str = "logistic"


@dataclass
class LayerTelemetry:
    layer_index: int = 0
    total_neurons: int = 0
    rendered_neurons: int = 0
    active_neurons: int = 0
    dead_neurons: int = 0
    max_activation: float = 0.0


def _unwrap_mlp_container(model_obj: Any) -> tuple[Any, Any]:
    """Return ``(predict_obj, final_estimator)`` following wrapper chains."""
    predict_obj = model_obj
    seen = 0
    while seen < 8:
        seen += 1
        best = getattr(model_obj, "best_estimator_", None)
        if best is not None:
            model_obj = best
            predict_obj = best
            continue
        steps = getattr(model_obj, "steps", None)
        if isinstance(steps, list) and steps:
            last = steps[-1]
            model_obj = last[1] if isinstance(last, (tuple, list)) and len(last) == 2 else last
            continue
        reg = getattr(model_obj, "regressor_", None)
        if reg is not None:
            model_obj = reg
            continue
        break
    return predict_obj, model_obj


def _final_mlp_estimator(model_obj: Any) -> Any:
    return _unwrap_mlp_container(model_obj)[1]


def _is_mlp_obj(obj: Any) -> bool:
    if _MLP_AVAILABLE and isinstance(obj, _CLASSIFIER_TYPES + _REGRESSOR_TYPES):
        return True
    return type(obj).__name__ in ("MLPClassifier", "MLPRegressor")


def _is_regressor_obj(est: Any) -> bool:
    if _MLP_AVAILABLE and isinstance(est, _REGRESSOR_TYPES):
        return True
    if _MLP_AVAILABLE and isinstance(est, _CLASSIFIER_TYPES):
        return False
    return type(est).__name__ == "MLPRegressor"


def _feature_names(est: Any, n: int) -> list[str]:
    for attr in ("feature_names_in_", "feature_names_"):
        try:
            raw = getattr(est, attr, None)
            if raw is not None and len(list(raw)) == n:
                return [str(v) for v in list(raw)]
        except (AttributeError, TypeError, ValueError):
            continue
    return [f"feature_{i}" for i in range(max(n, 0))]


def _classes_of(est: Any) -> list[Any]:
    raw = getattr(est, "classes_", None)
    if raw is None:
        return [0, 1]
    try:
        return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]
    except (TypeError, ValueError):
        return [0, 1]


def _hidden_activation_name(est: Any) -> str:
    raw = str(getattr(est, "activation", "relu") or "relu").lower()
    return raw if raw in ("relu", "tanh", "logistic", "identity") else "relu"


def _out_activation_name(est: Any, is_classifier: bool) -> str:
    raw = getattr(est, "out_activation_", None)
    if raw is not None:
        name = str(raw).lower()
        if name in ("softmax", "logistic", "identity", "tanh", "relu"):
            return name
    # Unfitted fallback mirrors sklearn defaults.
    if not is_classifier:
        return "identity"
    try:
        n_out = int(getattr(est, "n_outputs_", 1) or 1)
    except (TypeError, ValueError):
        n_out = 1
    classes = _classes_of(est) if getattr(est, "classes_", None) is not None else [0, 1]
    if n_out > 1 or len(classes) > 2:
        return "softmax"
    return "logistic"


def _stable_expit(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic sigmoid (edge #2: no RuntimeWarning on ±1e4)."""
    with np.errstate(over="ignore", under="ignore"):
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
    return np.where(np.isfinite(out), out, np.where(z >= 0, 1.0, 0.0))


def _stable_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with log-sum-exp stabilization (edge #2)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    shifted = z - np.max(z, axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore"):
        exp_m = np.exp(shifted)
    denom = np.sum(exp_m, axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    out = exp_m / denom
    return np.where(np.isfinite(out), out, 0.0)


def _apply_hidden_activation(z: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        with np.errstate(over="ignore", under="ignore"):
            return np.tanh(z)
    if name == "logistic":
        return _stable_expit(z)
    return z.copy()  # identity


def _apply_output_activation(z: np.ndarray, name: str) -> np.ndarray:
    if name == "softmax":
        return _stable_softmax(z)
    if name == "logistic":
        return _stable_expit(z)
    if name == "tanh":
        with np.errstate(over="ignore", under="ignore"):
            return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    return z.copy()  # identity


def _hidden_derivative(z: np.ndarray, a: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return (z > 0).astype(np.float64)
    if name == "tanh":
        return np.maximum(1.0 - a * a, 0.0)
    if name == "logistic":
        return np.maximum(a * (1.0 - a), 0.0)
    return np.ones_like(z, dtype=np.float64)  # identity


def _coefs_intercepts(est: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    raw_coefs = list(getattr(est, "coefs_", []))
    raw_intercepts = list(getattr(est, "intercepts_", []))
    coefs = [np.asarray(c, dtype=np.float64) for c in raw_coefs]
    intercepts = [np.asarray(b, dtype=np.float64).ravel() for b in raw_intercepts]
    return coefs, intercepts


def _forward_full(
    X: np.ndarray,
    coefs: list[np.ndarray],
    intercepts: list[np.ndarray],
    hidden_name: str,
    out_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(activations, pre_activations)`` including the input layer."""
    a = np.asarray(X, dtype=np.float64)
    activations = [np.ascontiguousarray(a, dtype=np.float64)]
    pre_activations: list[np.ndarray] = []
    n_transitions = len(coefs)
    for li in range(n_transitions):
        w = np.asarray(coefs[li], dtype=np.float64)
        b = np.asarray(intercepts[li], dtype=np.float64).ravel()
        z = activations[-1] @ w + b.reshape(1, -1)
        z = np.where(np.isfinite(z), z, 0.0)
        pre_activations.append(np.ascontiguousarray(z, dtype=np.float64))
        if li < n_transitions - 1:
            a_next = _apply_hidden_activation(z, hidden_name)
        else:
            a_next = _apply_output_activation(z, out_name)
        a_next = np.where(np.isfinite(a_next), a_next, 0.0)
        activations.append(np.ascontiguousarray(a_next, dtype=np.float64))
    return activations, pre_activations


def _layer_dimensions(n_in: int, coefs: list[np.ndarray]) -> list[int]:
    dims = [int(n_in)]
    for w in coefs:
        dims.append(int(np.asarray(w).shape[1]))
    return dims


def _check_fitted_mlp(est: Any) -> None:
    if not hasattr(est, "coefs_") or not hasattr(est, "intercepts_"):
        raise ValueError(f"Estimator '{type(est).__name__}' is not a fitted MLP model")
    coefs, intercepts = _coefs_intercepts(est)
    if not coefs or not intercepts or len(coefs) != len(intercepts):
        raise ValueError(f"Estimator '{type(est).__name__}' is not a fitted MLP model")


class _MLPBase(BaseModelAdapter):
    is_classifier: bool = True
    supports_nan_input: bool = False

    def __init__(self, model: Any | None = None) -> None:
        # Spec-style construction ``MLPAdapter(model=mlp)`` binds the model;
        # contract-style use (``Adapter()`` + ``model_obj`` args) leaves it None.
        self._bound_model: Any | None = model

    # -- spec-level engines (also used by the BaseModelAdapter contract) ----
    def compute_forward_trace(
        self,
        X_chunk: np.ndarray,
        model_obj: Any | None = None,
        estimator: Any | None = None,
    ) -> NeuralForwardTrace:
        est = estimator if estimator is not None else model_obj
        if est is None:
            est = getattr(self, "_bound_model", None)
        if est is None:
            raise ValueError("compute_forward_trace requires a fitted MLP estimator")
        _, est = _unwrap_mlp_container(est)
        if not _is_mlp_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not an MLP model")
        _check_fitted_mlp(est)
        X = np.asarray(X_chunk, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {X.shape}")
        X = np.where(np.isfinite(X), X, 0.0)
        n_rows = int(X.shape[0])

        is_clf = not _is_regressor_obj(est)
        hidden_name = _hidden_activation_name(est)
        out_name = _out_activation_name(est, is_clf)
        coefs, intercepts = _coefs_intercepts(est)
        activations, _ = _forward_full(X, coefs, intercepts, hidden_name, out_name)
        dims = _layer_dimensions(int(X.shape[1]), coefs)
        n_layers = len(dims)

        rendered: list[list[int]] = []
        down_activations: list[np.ndarray] = []
        for li, dim in enumerate(dims):
            if 0 < li < n_layers - 1 and dim > MAX_RENDERED_NEURONS_PER_LAYER:
                full_a = activations[li]
                mean_abs = np.mean(np.abs(full_a), axis=0) if n_rows else np.zeros((dim,))
                next_w = np.asarray(coefs[li], dtype=np.float64)  # W^(li+1): (dim, d_next)
                col_strength = np.max(np.abs(next_w), axis=1) if next_w.size else np.zeros((dim,))
                energy = mean_abs * col_strength
                order = np.argsort(-energy, kind="stable")[:MAX_RENDERED_NEURONS_PER_LAYER]
                sel = sorted(int(v) for v in order.tolist())
                rendered.append(sel)
                down_activations.append(
                    np.ascontiguousarray(full_a[:, sel], dtype=np.float32)
                )
            else:
                rendered.append(list(range(dim)))
                down_activations.append(
                    np.ascontiguousarray(activations[li], dtype=np.float32)
                )

        # Dead ReLU masks stay full-width so original indices are preserved.
        dead_masks: list[np.ndarray] = []
        dead_ratios: list[np.ndarray] = []
        for li, dim in enumerate(dims):
            if 0 < li < n_layers - 1 and hidden_name == "relu":
                full_a = activations[li]
                if n_rows:
                    ratio = np.mean(full_a <= 0, axis=0).astype(np.float64)
                else:
                    ratio = np.zeros((dim,), dtype=np.float64)
                dead_masks.append(np.ascontiguousarray(ratio >= 1.0))
                dead_ratios.append(np.ascontiguousarray(ratio, dtype=np.float64))
            else:
                dead_masks.append(np.zeros((dim,), dtype=bool))
                dead_ratios.append(np.zeros((dim,), dtype=np.float64))

        # Synapse flux on rendered pairs: F = mean(A_prev_u) * W[u, v].
        edges_per_layer: list[list[dict[str, Any]]] = []
        for li, w in enumerate(coefs):
            w = np.asarray(w, dtype=np.float64)
            prev_sel = rendered[li]
            curr_sel = rendered[li + 1]
            if not prev_sel or not curr_sel:
                edges_per_layer.append([])
                continue
            prev_mean = (
                np.mean(activations[li], axis=0) if n_rows else np.zeros((dims[li],))
            )
            candidates: list[dict[str, Any]] = []
            for u in prev_sel:
                au = float(prev_mean[u]) if 0 <= u < prev_mean.shape[0] else 0.0
                for v in curr_sel:
                    if not (0 <= u < w.shape[0] and 0 <= v < w.shape[1]):
                        continue
                    flux = float(au * float(w[u, v]))
                    if not np.isfinite(flux):
                        flux = 0.0
                    candidates.append(
                        {"fromLayer": int(li), "u": int(u), "v": int(v),
                         "flux": round(flux, 5)}
                    )
            candidates.sort(key=lambda e: abs(float(e["flux"])), reverse=True)
            edges_per_layer.append(candidates[:MAX_SYNAPSES_PER_LAYER_PAIR])

        telemetry: list[dict[str, Any]] = []
        for li, dim in enumerate(dims):
            full_a = activations[li]
            if 0 < li < n_layers - 1:
                active = int(np.sum(np.any(full_a > 0, axis=0))) if n_rows else 0
                dead = int(np.sum(dead_masks[li]))
                peak = float(np.max(np.abs(full_a))) if full_a.size and n_rows else 0.0
            elif li == 0:
                active, dead = int(dim), 0
                peak = float(np.max(np.abs(full_a))) if full_a.size and n_rows else 0.0
            else:
                active, dead = int(dim), 0
                peak = float(np.max(np.abs(full_a))) if full_a.size and n_rows else 0.0
            telemetry.append({
                "layer_index": int(li),
                "total_neurons": int(dim),
                "rendered_neurons": int(len(rendered[li])),
                "active_neurons": int(active),
                "dead_neurons": int(dead),
                "dead_ratio_mean": float(np.mean(dead_ratios[li])) if dim else 0.0,
                "max_activation": round(float(peak), 5) if np.isfinite(peak) else 0.0,
            })

        output = np.ascontiguousarray(activations[-1], dtype=np.float32)
        if is_clf and out_name == "logistic" and output.shape[1] == 1:
            # sklearn predict_proba convention: binary logistic head stores a
            # single P(class_1) column — expand to [P0, P1] for parity checks.
            p1 = np.clip(output.ravel().astype(np.float64), 0.0, 1.0)
            output = np.ascontiguousarray(
                np.column_stack([1.0 - p1, p1]), dtype=np.float32
            )
        return NeuralForwardTrace(
            layer_dimensions=dims,
            rendered_node_indices=rendered,
            layer_activations=down_activations,
            full_layer_activations=[
                np.ascontiguousarray(a, dtype=np.float32) for a in activations
            ],
            output_activations=output,
            dead_neuron_masks=dead_masks,
            dead_ratios=dead_ratios,
            synapse_edges_per_layer=edges_per_layer,
            layer_telemetry=telemetry,
            activation=hidden_name,
            out_activation=out_name,
        )

    def compute_input_gradient_attribution(
        self,
        X_chunk: np.ndarray,
        model_obj: Any | None = None,
        estimator: Any | None = None,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> np.ndarray:
        est = estimator if estimator is not None else model_obj
        if est is None:
            est = getattr(self, "_bound_model", None)
        if est is None:
            raise ValueError("compute_input_gradient_attribution requires a fitted MLP estimator")
        _, est = _unwrap_mlp_container(est)
        if not _is_mlp_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not an MLP model")
        _check_fitted_mlp(est)
        X = np.asarray(X_chunk, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = np.where(np.isfinite(X), X, 0.0)
        n_rows = int(X.shape[0])
        if n_rows == 0:
            return np.zeros((0, int(X.shape[1]) if X.ndim == 2 else 0), dtype=np.float32)

        is_clf = not _is_regressor_obj(est)
        hidden_name = _hidden_activation_name(est)
        out_name = _out_activation_name(est, is_clf)
        coefs, intercepts = _coefs_intercepts(est)
        n_transitions = len(coefs)
        classes = _classes_of(est)
        n_classes = len(classes) if is_clf else 0
        if is_clf and n_classes > 2:
            resolved = int(target_class_idx if target_class_idx is not None else target_idx)
            resolved = min(max(resolved, 0), n_classes - 1)
        elif is_clf:
            resolved = 1 if target_class_idx is None else min(max(int(target_class_idx), 0), 1)
            if n_classes == 2 and target_class_idx is None and target_idx == 0:
                resolved = 1
        else:
            resolved = 0

        out = np.zeros((n_rows, int(X.shape[1])), dtype=np.float64)
        for start in range(0, n_rows, GRAD_CHUNK_ROWS):
            xb = X[start:start + GRAD_CHUNK_ROWS]
            m = int(xb.shape[0])
            activations, pre = _forward_full(xb, coefs, intercepts, hidden_name, out_name)
            last_a = activations[-1]
            # d out / d Z_last for the resolved scalar head.
            if not is_clf:
                width = int(last_a.shape[1]) if last_a.ndim == 2 else 1
                k = min(resolved, width - 1)
                delta = np.zeros_like(last_a)
                delta[:, k] = 1.0
            elif out_name == "logistic":
                p = np.asarray(last_a, dtype=np.float64)
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                width = int(p.shape[1])
                k = min(resolved, width - 1)
                d = np.zeros_like(p)
                col = p[:, k]
                d[:, k] = col * (1.0 - col)
                delta = d
            elif out_name == "softmax":
                p = np.asarray(last_a, dtype=np.float64)
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                width = int(p.shape[1])
                k = min(resolved, width - 1)
                pk = p[:, k:k + 1]
                delta = -pk * p
                delta[:, k] = pk.ravel() * (1.0 - p[:, k])
            else:  # linear-ish output head (identity/tanh/relu)
                width = int(last_a.shape[1]) if last_a.ndim == 2 else 1
                k = min(resolved, width - 1)
                delta = np.zeros_like(np.asarray(last_a, dtype=np.float64))
                if out_name == "tanh":
                    a = np.asarray(last_a, dtype=np.float64)
                    deriv = np.maximum(1.0 - a * a, 0.0)
                    delta[:, k] = deriv[:, k]
                elif out_name == "relu":
                    z = pre[-1]
                    delta[:, k] = (np.asarray(z)[:, k] > 0).astype(np.float64)
                else:
                    delta[:, k] = 1.0
            # Backpropagate through hidden layers (skip output activation).
            for li in range(n_transitions - 1, 0, -1):
                w = np.asarray(coefs[li], dtype=np.float64)
                z_prev = pre[li - 1]
                delta = (delta @ w.T) * _hidden_derivative(
                    z_prev, _apply_hidden_activation(z_prev, hidden_name), hidden_name
                )
            # Final hop: d out / d X = (d out / d Z^(1)) @ W^(1).T.
            grad = delta @ np.asarray(coefs[0], dtype=np.float64).T
            out[start:start + m] = np.where(np.isfinite(grad), grad, 0.0)
        attribution = out * np.asarray(X, dtype=np.float64)
        attribution = np.where(np.isfinite(attribution), attribution, 0.0)
        return np.ascontiguousarray(attribution, dtype=np.float32)

    # -- BaseModelAdapter contract -------------------------------------
    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        width = max(int(n_features), 0)
        try:
            _, est = _unwrap_mlp_container(estimator)
            coefs, _ = _coefs_intercepts(est)
            if not coefs:
                return [None for _ in range(width)]
            w0 = np.asarray(coefs[0], dtype=np.float64)
            if w0.shape[0] != width:
                return [None for _ in range(width)]
            scores = np.mean(np.abs(w0), axis=1)
            return [float(v) if np.isfinite(v) else 0.0 for v in scores.tolist()]
        except Exception:
            return [None for _ in range(width)]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, est = _unwrap_mlp_container(model_obj)
        if not _is_mlp_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not an MLP model")
        X = np.asarray(X_float32, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xf = np.ascontiguousarray(X, dtype=np.float32)
        is_clf = not _is_regressor_obj(est)
        if is_clf:
            try:
                probas = np.asarray(predict_obj.predict_proba(Xf), dtype=np.float64)
            except (AttributeError, TypeError, ValueError, NotImplementedError):
                probas = np.asarray(est.predict_proba(Xf), dtype=np.float64)
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
            if probas.ndim == 2 and probas.shape[1] == 1:
                p = np.clip(np.where(np.isfinite(probas.ravel()), probas.ravel(), 0.0), 0.0, 1.0)
                probas = np.column_stack([1.0 - p, p])
            probas = np.where(np.isfinite(probas), probas, 0.0)
            try:
                preds = np.asarray(predict_obj.predict(Xf)).ravel()
            except (AttributeError, TypeError, ValueError):
                preds = np.asarray(est.predict(Xf)).ravel()
            py_preds: list[float | int | str] = []
            for v in preds.tolist():
                if isinstance(v, (bool, np.bool_)):
                    py_preds.append(int(v))
                elif isinstance(v, (np.integer, int)):
                    py_preds.append(int(v))
                elif isinstance(v, (np.floating, float)):
                    f = float(v)
                    py_preds.append(int(f) if float(f).is_integer() else f)
                else:
                    py_preds.append(v)
            probabilities = [
                [float(v) if np.isfinite(v) else 0.0 for v in row]
                for row in np.asarray(probas).tolist()
            ]
            confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            return PredictionBatchResult(
                predictions=py_preds, probabilities=probabilities, confidences=confidences
            )
        try:
            vals = np.asarray(predict_obj.predict(Xf), dtype=np.float64).ravel()
        except (AttributeError, TypeError, ValueError):
            vals = np.asarray(est.predict(Xf), dtype=np.float64).ravel()
        py_vals: list[float | int | str] = [
            float(v) if np.isfinite(v) else 0.0 for v in vals.tolist()
        ]
        return PredictionBatchResult(predictions=py_vals, probabilities=None, confidences=None)

    def _trace_common(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        predict_obj, est = _unwrap_mlp_container(model_obj)
        if not _is_mlp_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not an MLP model")
        _check_fitted_mlp(est)
        X = np.asarray(X_float32, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {X.shape}")
        X = np.where(np.isfinite(X), X, 0.0)
        n_rows = int(X.shape[0])
        width = int(X.shape[1]) if X.ndim == 2 else 0
        names = feature_names if len(feature_names) == width else _feature_names(est, width)
        is_clf = not _is_regressor_obj(est)
        classes: list[Any] = _classes_of(est) if is_clf else []
        n_classes = len(classes) if is_clf else 0
        if is_clf and n_classes > 2:
            resolved = int(target_class_idx if target_class_idx is not None else target_idx)
            resolved = min(max(resolved, 0), n_classes - 1)
        elif is_clf:
            if target_class_idx is not None:
                resolved = min(max(int(target_class_idx), 0), 1)
            else:
                resolved = 1
        else:
            resolved = 0

        trace = self.compute_forward_trace(X, estimator=est)
        attrib = self.compute_input_gradient_attribution(
            X, estimator=est, target_class_idx=resolved if is_clf else None,
            target_idx=resolved,
        )
        batch = self.predict_batch(predict_obj if predict_obj is not est else est, X)
        probas = batch.probabilities
        preds = batch.predictions

        hidden_name = trace.activation
        try:
            coefs, _ = _coefs_intercepts(est)
            depth_repr = [int(np.asarray(w).shape[0]) for w in coefs]
        except Exception:
            depth_repr = []
        _ = depth_repr

        traces: list[InternalFlowRowTrace] = []
        for pos, row_id in enumerate(row_indices):
            if not (0 <= row_id < n_rows):
                continue
            row_attr = np.asarray(attrib[row_id], dtype=np.float64).ravel()
            if row_attr.size < width:
                row_attr = np.pad(row_attr, (0, width - row_attr.size))
            row_attr = row_attr[:width]
            try:
                final_pred: Any = preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[resolved] if is_clf and resolved < len(classes) else 0
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                if not np.isfinite(f):
                    f = 0.0
                final_pred = f
            if is_clf and probas is not None and row_id < len(probas):
                link_val = float(probas[row_id][min(resolved, len(probas[row_id]) - 1)])
                pred_scalar = float(link_val)
            elif not is_clf:
                try:
                    pred_scalar = float(np.asarray(preds[row_id], dtype=np.float64))
                except (TypeError, ValueError, IndexError):
                    pred_scalar = 0.0
                link_val = pred_scalar
            else:
                link_val = 0.0
                pred_scalar = 0.0
            if not np.isfinite(pred_scalar):
                pred_scalar = 0.0
            contrib_sum = float(np.sum(row_attr)) if row_attr.size else 0.0
            base_raw = float(pred_scalar - contrib_sum)
            if not np.isfinite(base_raw):
                base_raw = 0.0

            order = np.argsort(np.abs(row_attr))[::-1][: min(8, width)] if width else np.asarray([], dtype=int)
            stages = [{
                "rank": int(f_pos + 1),
                "feature_idx": int(f_idx),
                "feature_name": names[int(f_idx)] if int(f_idx) < len(names) else f"feature_{int(f_idx)}",
                "shap_value": round(float(row_attr[int(f_idx)]), 5),
            } for f_pos, f_idx in enumerate(np.asarray(order).tolist())]

            # Per-row downsampled activations + row-specific synapse flux.
            row_layer_acts: list[list[float]] = []
            for li, arr in enumerate(trace.layer_activations):
                row_layer_acts.append([float(v) for v in np.asarray(arr[row_id]).tolist()])
            row_edges: list[dict[str, Any]] = []
            for li, edges in enumerate(trace.synapse_edges_per_layer):
                w = np.asarray(coefs[li], dtype=np.float64)
                a_prev_full = np.asarray(trace.full_layer_activations[li][row_id], dtype=np.float64)
                for e in edges:
                    u, v = int(e["u"]), int(e["v"])
                    flux = 0.0
                    if 0 <= u < a_prev_full.shape[0] and 0 <= u < w.shape[0] and 0 <= v < w.shape[1]:
                        flux = float(a_prev_full[u] * float(w[u, v]))
                    if not np.isfinite(flux):
                        flux = 0.0
                    row_edges.append({"fromLayer": int(li), "u": u, "v": v, "flux": round(flux, 5)})
            row_edges.sort(key=lambda e: abs(float(e["flux"])), reverse=True)
            row_edges = row_edges[: MAX_SYNAPSES_PER_LAYER_PAIR * max(len(coefs), 1)]

            dead_counts = [int(np.sum(m)) for m in trace.dead_neuron_masks]
            total_dead = int(sum(dead_counts))
            n_hidden = max(len(trace.layer_dimensions) - 2, 0)
            try:
                if X_float32 is None:
                    has_nan = False
                else:
                    has_nan = bool(np.isnan(np.asarray(X_float32[row_id], dtype=np.float64)).any())
            except (TypeError, ValueError):
                has_nan = False
            try:
                hidden_sizes = tuple(int(d) for d in trace.layer_dimensions[1:-1])
            except Exception:
                hidden_sizes = ()
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(is_clf),
                "is_regressor": bool(not is_clf),
                "is_binary": bool(is_clf and n_classes == 2),
                "algorithm": type(est).__name__,
                "activation": hidden_name,
                "out_activation": trace.out_activation,
                "hidden_layer_sizes": list(hidden_sizes),
                "layer_dimensions": list(trace.layer_dimensions),
                "n_layers": int(len(trace.layer_dimensions)),
                "n_hidden_layers": int(n_hidden),
                "classes": [str(c) for c in classes] if is_clf else [],
                "class_labels": classes if is_clf else [],
                "num_class": int(n_classes) if is_clf else 0,
                "model_family": "mlp",
                "rendered_node_indices": [list(map(int, r)) for r in trace.rendered_node_indices],
                "layer_activations_sample": row_layer_acts,
                "dead_neuron_masks": [[bool(v) for v in np.asarray(m).tolist()] for m in trace.dead_neuron_masks],
                "dead_ratios": [[float(v) for v in np.asarray(r).tolist()] for r in trace.dead_ratios],
                "dead_counts_per_layer": dead_counts,
                "total_dead_neurons": total_dead,
                "layer_telemetry": trace.layer_telemetry,
                "synapse_edges": row_edges,
                "synapse_edges_per_layer_counts": [len(e) for e in trace.synapse_edges_per_layer],
                "max_synapses_per_layer_pair": int(MAX_SYNAPSES_PER_LAYER_PAIR),
                "max_rendered_neurons_per_layer": int(MAX_RENDERED_NEURONS_PER_LAYER),
                "probabilities": ([float(v) for v in probas[row_id]] if probas is not None and row_id < len(probas) else []),
                "predicted_class": final_pred,
                "linked_prediction": round(float(link_val), 5) if np.isfinite(link_val) else 0.0,
                "winning_class_idx": int(resolved) if is_clf else 0,
                "resolved_class_idx": int(resolved),
                "base_raw_score": float(base_raw),
                "final_raw_score": float(pred_scalar),
                "has_missing_values": bool(has_nan),
                "top_shap_features": stages,
                "attribution_method": "input_x_gradient",
            }
            traces.append(InternalFlowRowTrace(
                row_index=int(row_id),
                base_value=float(base_raw),
                final_prediction=final_pred,  # type: ignore[arg-type]
                feature_contributions={
                    names[i] if i < len(names) else f"feature_{i}": float(row_attr[i])
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
        predict_obj, est = _unwrap_mlp_container(model_obj)
        if not _is_mlp_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not an MLP model")
        _check_fitted_mlp(est)
        is_clf = not _is_regressor_obj(est)
        classes = _classes_of(est) if is_clf else []
        n_classes = len(classes) if is_clf else 0
        if is_clf and n_classes > 2:
            resolved = int(target_class_idx if target_class_idx is not None else target_idx)
            resolved = min(max(resolved, 0), n_classes - 1)
        elif is_clf:
            resolved = 1 if target_class_idx is None else min(max(int(target_class_idx), 0), 1)
            if target_class_idx is None and target_idx == 0 and n_classes != 2:
                resolved = min(max(int(target_idx), 0), 1)
        else:
            resolved = 0
        hidden_name = _hidden_activation_name(est)
        out_name = _out_activation_name(est, is_clf)
        coefs, intercepts = _coefs_intercepts(est)

        width = max(int(n_features), 1)
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res).astype(np.float64)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res).astype(np.float64)
        medians = np.zeros((width,), dtype=np.float64)
        for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
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

        hidden_sizes = [int(np.asarray(w).shape[1]) for w in coefs[:-1]] if len(coefs) > 1 else []
        z_flat = np.zeros((res * res,), dtype=np.float64)
        for start in range(0, res * res, GRAD_CHUNK_ROWS * 4):
            chunk = grid[start:start + GRAD_CHUNK_ROWS * 4]
            activations, _ = _forward_full(chunk, coefs, intercepts, hidden_name, out_name)
            last = np.asarray(activations[-1], dtype=np.float64)
            if is_clf:
                if last.ndim == 1:
                    last = last.reshape(-1, 1)
                k = min(resolved, last.shape[1] - 1)
                z_flat[start:start + chunk.shape[0]] = np.asarray(last[:, k], dtype=np.float64).ravel()
            else:
                if last.ndim == 2:
                    k = min(resolved, last.shape[1] - 1)
                    z_flat[start:start + chunk.shape[0]] = np.asarray(last[:, k], dtype=np.float64).ravel()
                else:
                    z_flat[start:start + chunk.shape[0]] = np.asarray(last, dtype=np.float64).ravel()
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        z_surface = z_flat.reshape(res, res)
        if not is_clf and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0

        dims = _layer_dimensions(width, coefs)
        try:
            n_iter = int(getattr(est, "n_iter_", 0) or 0)
        except (TypeError, ValueError):
            n_iter = 0
        try:
            loss_v = float(getattr(est, "loss_", 0.0) or 0.0)
        except (TypeError, ValueError):
            loss_v = 0.0
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in np.asarray(z_surface, dtype=np.float64).ravel()],
            extra_overlays={
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes] if is_clf else [],
                "algorithm": type(est).__name__,
                "activation": hidden_name,
                "out_activation": out_name,
                "hidden_layer_sizes": hidden_sizes,
                "layer_dimensions": dims,
                "n_layers": int(len(dims)),
                "loss": getattr(est, "loss", "squared_error"),
                "final_loss": float(loss_v) if np.isfinite(loss_v) else 0.0,
                "n_iter": int(n_iter),
                "model_family": "mlp",
                "is_classifier": bool(is_clf),
                "is_regressor": bool(not is_clf),
                "is_binary": bool(is_clf and n_classes == 2),
                "num_class": int(n_classes) if is_clf else 0,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not is_clf else 0.5,
            },
        )


@ModelRegistry.register("mlp")
class MLPClassifierAdapter(_MLPBase):
    display_name = "MLP Classifier (Floating Synapse Brain)"
    family_name = "Neural Network"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        _, est = _unwrap_mlp_container(estimator)
        if _MLP_AVAILABLE and isinstance(est, _CLASSIFIER_TYPES):
            return True
        if type(est).__name__ == "MLPClassifier":
            return True
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
        return self._trace_common(
            model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return super().compute_3d_surface(
            model_obj, grid_spec, n_features, target_class_idx, target_idx
        )


class MLPAdapter(_MLPBase):
    """Spec фасад: ``MLPAdapter(model=mlp)`` — delegates by fitted class.

    Keeps the dev/test-plan import path working while the registry uses the
    ``mlp`` / ``mlp_regressor`` subclasses for upload resolution.
    """

    display_name = "MLP (Floating Synapse Brain)"
    family_name = "Neural Network"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        _, est = _unwrap_mlp_container(estimator)
        return _is_mlp_obj(est)

    def _delegate_for(self, model_obj: Any | None) -> _MLPBase:
        target = model_obj if model_obj is not None else getattr(self, "_bound_model", None)
        _, est = _unwrap_mlp_container(target)
        if _is_regressor_obj(est):
            delegate = MLPRegressorAdapter.__new__(MLPRegressorAdapter)
            delegate._bound_model = target
            return delegate
        delegate = MLPClassifierAdapter.__new__(MLPClassifierAdapter)
        delegate._bound_model = target
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
        target = model_obj if model_obj is not None else getattr(self, "_bound_model", None)
        return self._delegate_for(target)._trace_common(
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
        target = model_obj if model_obj is not None else getattr(self, "_bound_model", None)
        return self._delegate_for(target).compute_3d_surface(
            target, grid_spec, n_features, target_class_idx, target_idx
        )


@ModelRegistry.register("mlp_regressor")
class MLPRegressorAdapter(_MLPBase):
    display_name = "MLP Regressor (Floating Synapse Brain)"
    family_name = "Neural Network"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.NEURAL_MORPHING_MANIFOLD
    is_classifier = False

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        _, est = _unwrap_mlp_container(estimator)
        if _MLP_AVAILABLE and isinstance(est, _REGRESSOR_TYPES):
            return True
        if type(est).__name__ == "MLPRegressor":
            return True
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
        return self._trace_common(
            model_obj, X_float32, row_indices, feature_names, target_idx, target_class_idx
        )

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return super().compute_3d_surface(
            model_obj, grid_spec, n_features, target_class_idx, target_idx
        )

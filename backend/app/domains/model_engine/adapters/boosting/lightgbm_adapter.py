# backend/app/domains/model_engine/adapters/boosting/lightgbm_adapter.py
"""Phase 34 — LightGBM leaf-wise adapters (native Exact TreeSHAP + dump_model).

Covers ``LGBMClassifier`` / ``LGBMRegressor`` / ``LGBMRanker``, native
``lgb.Booster`` artifacts (``.txt`` / ``.model`` / ``.lgb`` / ``.pkl``) and
``Pipeline`` / ``GridSearchCV``-style wrappers — inference runs through the
full container while staged tracing unwraps to the fitted booster.

Blueprint mapping (phase-34-model-lightgbm.md):

- One booster per call feeds ``pred_contrib`` (Exact C++ TreeSHAP),
  ``pred_leaf`` and the ``num_iteration`` staged trajectory (§5.2).
  Multiclass SHAP arrives as a flattened 2D array
  ``(n, C * (f + 1))`` and is reshaped to ``(n, C, f + 1)`` (edge #1);
  ``pred_contrib`` is singular — ``pred_contribs`` is XGBoost-only (edge #2).
- Single-tree ``pred_leaf`` may collapse to 1D ``(n,)`` — always reshape
  to ``(n, 1)`` (edge #3); multiclass leaves ``(n, iters * C)`` reshape to
  ``(n, iters, C)`` for the target class column.
- Topology comes from a single ``dump_model(num_iteration=100)`` pass
  (§6.1 — 95% RAM saving vs full dump); categorical splits use
  ``decision_type == "=="`` with ``"0||3||5"`` bitset thresholds (edge #4).
- Sparse input may return ``scipy.sparse`` SHAP — densified via
  ``toarray()`` (edge #5); positional ``np.ndarray`` inference keeps
  ``Column_0``-style auto names aligned (edge #6).
- ``num_thread=2`` pins OpenMP threads so workers never thrash the
  ComputePool (§6.3); staged curves and finals stay in linked
  (probability / raw value) space.
"""

from __future__ import annotations

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

try:  # Phase 34: lightgbm is optional at import time (registry stays importable).
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

    _LGB_AVAILABLE = True
    _CLASSIFIER_TYPES = (LGBMClassifier,)
    _REGRESSOR_TYPES = (LGBMRegressor, LGBMRanker)
except ImportError:  # pragma: no cover - exercised only without lightgbm
    lgb = None  # type: ignore[assignment]
    _LGB_AVAILABLE = False
    _CLASSIFIER_TYPES = ()
    _REGRESSOR_TYPES = ()

MAX_TRAJECTORY_POINTS = 50
MAX_EXPORTED_TREES = 100
MAX_TOPOLOGY_TREES = 10
MAX_SPLIT_LINES = 30
MAX_SURFACE_RESOLUTION = 100
TOP_STAGES = 8


def _unwrap_lgb_container(model_obj: Any) -> tuple[Any, Any]:
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


def _final_lgb_estimator(model_obj: Any) -> Any:
    return _unwrap_lgb_container(model_obj)[1]


def _is_native_booster(obj: Any) -> bool:
    return _LGB_AVAILABLE and isinstance(obj, lgb.Booster)


def _as_booster(est: Any) -> Any | None:
    """Return the native booster for wrappers, boosters, or ``None``."""
    if _is_native_booster(est):
        return est
    booster = getattr(est, "booster_", None)
    if _is_native_booster(booster):
        return booster
    return None


def _dump_objective(booster: Any) -> tuple[str, int]:
    """Parse ``(objective, num_class)`` from ``dump_model()`` (no full dump cost)."""
    try:
        dump = booster.dump_model(num_iteration=1)
        objective = str(dump.get("objective", "") or "")
        try:
            num_class = int(dump.get("num_class", 0) or 0)
        except (TypeError, ValueError):
            num_class = 0
        return objective, num_class
    except Exception:
        return "", 0


def _booster_feature_names(booster: Any) -> list[str]:
    try:
        names = list(booster.feature_name() or [])
    except (AttributeError, TypeError, ValueError):
        return []
    return [str(n) for n in names]


def _booster_counts(booster: Any) -> tuple[int, int, int]:
    """Return ``(n_trees_total, n_classes_internal, n_iterations)``."""
    try:
        total = int(booster.num_trees())
    except (AttributeError, TypeError, ValueError):
        total = 0
    try:
        per_iter = int(booster.num_model_per_iteration())
    except (AttributeError, TypeError, ValueError):
        per_iter = 1
    per_iter = max(per_iter, 1)
    iters = max(total // per_iter, 1) if total > 0 else 0
    return total, per_iter, iters


def _as_dense_float32(matrix: Any) -> np.ndarray:
    try:
        import scipy.sparse as _sp  # noqa: PLC0415

        if bool(_sp.issparse(matrix)):
            matrix = matrix.toarray()
    except ImportError:
        pass
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))


def _transform_for_estimators(predict_obj: Any, X_f32: np.ndarray) -> np.ndarray:
    """Apply pipeline preprocessing (all steps but last) so columns align."""
    steps = getattr(predict_obj, "steps", None)
    if not (isinstance(steps, list) and steps):
        return X_f32
    try:
        Xt: Any = X_f32
        for _, step_est in steps[:-1]:
            if hasattr(step_est, "transform"):
                Xt = step_est.transform(Xt)
            else:
                return X_f32
        return _as_dense_float32(Xt)
    except (ValueError, AttributeError, TypeError):
        return X_f32


def _classes_of(estimator: Any) -> list[Any]:
    raw = getattr(estimator, "classes_", None)
    if raw is None:
        return [0, 1]
    return [(v.item() if hasattr(v, "item") else v) for v in list(np.asarray(raw).ravel())]


def _predict_kwargs() -> dict[str, Any]:
    return {"num_thread": 2}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _split_categorical_threshold(raw: Any) -> list[int]:
    """Parse LightGBM ``"0||3||5"`` bitset thresholds (edge #4)."""
    if raw is None:
        return []
    if isinstance(raw, (int, np.integer)):
        return [int(raw)]
    if isinstance(raw, float | np.floating):
        return [int(raw)] if float(raw).is_integer() else []
    try:
        parts = str(raw).split("||")
        return sorted({int(p) for p in parts if p.strip() not in ("",)})
    except (TypeError, ValueError):
        return []


def _flatten_lgb_tree(tree_structure: Any) -> tuple[list[dict[str, Any]], dict[int, int], int, int]:
    """Flatten one LightGBM tree into id-linked nodes.

    Returns ``(flat, leaf_depth_map, min_depth, max_depth)`` where internal
    nodes carry ``yes``/``no`` child ids so the frontend can walk the
    root → leaf active-branch path (Phase 33 analogue).
    """
    flat: list[dict[str, Any]] = []
    leaf_depths: dict[int, int] = {}
    counter = 0

    def visit(node: Any, depth: int) -> int:
        nonlocal counter
        node_id = counter
        counter += 1
        if not isinstance(node, dict) or "split_index" not in node:
            try:
                leaf_index = int(node.get("leaf_index", node_id)) if isinstance(node, dict) else node_id
            except (TypeError, ValueError):
                leaf_index = node_id
            try:
                leaf_value = round(float(node.get("leaf_value", 0.0)), 5) if isinstance(node, dict) else 0.0
            except (TypeError, ValueError):
                leaf_value = 0.0
            try:
                leaf_count = int(node.get("leaf_count", 0)) if isinstance(node, dict) else 0
            except (TypeError, ValueError):
                leaf_count = 0
            flat.append(
                {"id": node_id, "leaf_index": leaf_index, "leaf_value": leaf_value,
                 "leaf_count": leaf_count, "depth": depth}
            )
            leaf_depths[leaf_index] = depth
            return node_id
        try:
            split_feature = int(node.get("split_feature", -1))
        except (TypeError, ValueError):
            split_feature = -1
        decision = str(node.get("decision_type", "<="))
        raw_threshold = node.get("threshold")
        if decision == "==":
            categories = _split_categorical_threshold(raw_threshold)
            threshold: Any = None
        else:
            categories = []
            try:
                threshold = round(float(raw_threshold), 5)
            except (TypeError, ValueError):
                threshold = None
        try:
            gain = round(float(node.get("split_gain", 0.0)), 4)
        except (TypeError, ValueError):
            gain = 0.0
        try:
            count = int(node.get("internal_count", 0))
        except (TypeError, ValueError):
            count = 0
        entry: dict[str, Any] = {
            "id": node_id,
            "split_index": node.get("split_index", node_id),
            "split_feature_idx": split_feature,
            "threshold": threshold,
            "categories": categories,
            "decision_type": decision,
            "default_left": bool(node.get("default_left", True)),
            "gain": gain,
            "count": count,
            "depth": depth,
        }
        flat.append(entry)
        left = visit(node.get("left_child"), depth + 1)
        right = visit(node.get("right_child"), depth + 1)
        entry["yes"] = left
        entry["no"] = right
        return node_id

    if tree_structure is None:
        return flat, leaf_depths, 0, 0
    visit(tree_structure, 0)
    depths = list(leaf_depths.values())
    return flat, leaf_depths, (min(depths) if depths else 0), (max(depths) if depths else 0)


class _LightGBMBase(BaseModelAdapter):
    is_classifier: bool = True
    supports_nan_input: bool = True

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        booster = _as_booster(_final_lgb_estimator(estimator))
        width = max(int(n_features), 0)
        if booster is None:
            return [None for _ in range(width)]
        return [float(v) for v in self._importances(booster, width, "gain")]

    def _importances(self, booster: Any, width: int, metric: str) -> list[float]:
        out = [0.0 for _ in range(max(width, 0))]
        try:
            scores = list(booster.feature_importance(importance_type=metric))
        except (AttributeError, TypeError, ValueError):
            return out
        for i, val in enumerate(scores[:width]):
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            out[i] = f if np.isfinite(f) else 0.0
        return out

    # -- inference ------------------------------------------------------
    def _linked_predict(
        self, predict_obj: Any, booster: Any, X_in: np.ndarray, resolved: int = 0
    ) -> np.ndarray:
        """Linked-space predictions honoring wrapper probability outputs."""
        if predict_obj is not booster and hasattr(predict_obj, "predict"):
            try:
                if self.is_classifier and hasattr(predict_obj, "predict_proba"):
                    try:
                        proba = np.asarray(predict_obj.predict_proba(_as_dense_float32(X_in)))
                    except (ValueError, AttributeError, TypeError, NotImplementedError):
                        proba = None
                    if proba is not None and proba.size:
                        if proba.ndim == 2 and proba.shape[1] > 1:
                            return np.asarray(proba[:, int(min(resolved, proba.shape[1] - 1))])
                        return np.asarray(proba).ravel()
                return np.asarray(predict_obj.predict(_as_dense_float32(X_in)))
            except (ValueError, AttributeError, TypeError):
                pass
        arr = np.asarray(booster.predict(_as_dense_float32(X_in), **_predict_kwargs()))
        if arr.ndim == 2 and arr.shape[1] > 1:
            return np.asarray(arr[:, int(min(resolved, arr.shape[1] - 1))])
        return np.asarray(arr).ravel()

    def _full_proba_dist(self, predict_obj: Any, booster: Any, X_arr: np.ndarray) -> np.ndarray | None:
        if predict_obj is not booster and hasattr(predict_obj, "predict_proba"):
            try:
                probas = np.asarray(predict_obj.predict_proba(_as_dense_float32(X_arr)), dtype=np.float64)
            except (ValueError, AttributeError, TypeError, NotImplementedError):
                probas = None
            if probas is not None and probas.size:
                if probas.ndim == 1:
                    probas = probas.reshape(-1, 1)
                return np.where(np.isfinite(probas), probas, 0.0)
        arr = np.asarray(booster.predict(_as_dense_float32(X_arr), **_predict_kwargs()), dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return np.where(np.isfinite(arr), arr, 0.0)
        flat = np.asarray(arr).ravel()
        p = np.clip(np.where(np.isfinite(flat), flat, 0.0), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, est = _unwrap_lgb_container(model_obj)
        booster = _as_booster(est)
        if booster is None:
            raise ValueError(f"Estimator '{type(est).__name__}' exposes no lightgbm.Booster")
        X_in = _as_dense_float32(X_float32)
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if self.is_classifier:
            probas = self._full_proba_dist(predict_obj, booster, X_in)
            if probas is None or not probas.size:
                probas = np.zeros((X_in.shape[0], 2), dtype=np.float64)
            preds = np.argmax(probas, axis=1)
            probabilities = [
                [float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()
            ]
            confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            py_preds: list[float | int | str] = [int(v) for v in np.asarray(preds).ravel().tolist()]
        else:
            vals = np.asarray(self._linked_predict(predict_obj, booster, X_in), dtype=np.float64).ravel()
            py_preds = [float(v) if np.isfinite(v) else 0.0 for v in vals.tolist()]
        return PredictionBatchResult(
            predictions=py_preds, probabilities=probabilities, confidences=confidences
        )

    # -- tracing --------------------------------------------------------
    def _trace_common(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        predict_obj, est = _unwrap_lgb_container(model_obj)
        booster = _as_booster(est)
        if booster is None:
            raise ValueError(f"Estimator '{type(est).__name__}' exposes no lightgbm.Booster")
        objective, num_class = _dump_objective(booster)
        n_total, n_per_iter, n_iters = _booster_counts(booster)
        if _is_native_booster(predict_obj) or predict_obj is booster:
            classes: list[Any] = (
                list(range(num_class)) if objective.startswith("multiclass") and num_class > 1
                else ([0, 1] if self.is_classifier else [])
            )
        else:
            raw_classes = _classes_of(predict_obj)
            classes = raw_classes or ([0, 1] if self.is_classifier else [])
        n_classes = len(classes) if self.is_classifier else 0

        X_native = _as_dense_float32(X_float32)
        Xt = _transform_for_estimators(predict_obj, X_native)
        n_features = int(X_native.shape[1]) if X_native.ndim == 2 else 0
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        width = max(n_features, 0)

        # 1) Exact C++ TreeSHAP (edge #2: singular pred_contrib).
        raw_contrib = booster.predict(Xt, pred_contrib=True, **_predict_kwargs())
        if hasattr(raw_contrib, "toarray"):  # edge #5: sparse SHAP output
            raw_contrib = raw_contrib.toarray()
        raw_contrib = np.asarray(raw_contrib, dtype=np.float32)
        if self.is_classifier and n_per_iter > 1 and raw_contrib.ndim == 2:
            # Edge #1: multiclass flattened 2D -> explicit class axis.
            n_cols = raw_contrib.shape[1]
            if n_cols == n_per_iter * (width + 1):
                contrib_3d = raw_contrib.reshape(raw_contrib.shape[0], n_per_iter, width + 1)
            else:  # pragma: no cover - defensive against version drift
                contrib_3d = raw_contrib.reshape(raw_contrib.shape[0], 1, n_cols)
                n_per_iter = 1
        else:
            if raw_contrib.ndim == 1:
                raw_contrib = raw_contrib.reshape(raw_contrib.shape[0], -1)
            contrib_3d = raw_contrib.reshape(raw_contrib.shape[0], 1, raw_contrib.shape[1])
        if self.is_classifier and n_per_iter > 1:
            if target_class_idx is not None and n_classes:
                resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
            elif n_classes == 2:
                resolved = 1
            else:
                resolved = int(min(max(int(target_idx), 0), max(n_per_iter - 1, 0)))
            grp = int(min(max(resolved, 0), contrib_3d.shape[1] - 1))
        else:
            if self.is_classifier and n_classes == 2 and target_class_idx is not None:
                resolved = int(min(max(int(target_class_idx), 0), 1))
            else:
                resolved = 1 if (self.is_classifier and n_classes == 2) else 0
            grp = 0
        phi = np.asarray(contrib_3d[:, grp, :width] if contrib_3d.shape[2] > width else contrib_3d[:, grp, :-1],
                         dtype=np.float64)
        bias_col = np.asarray(contrib_3d[:, grp, -1], dtype=np.float64)
        if self.is_classifier and n_per_iter <= 1 and n_classes == 2 and resolved == 0:
            phi, bias_col = -phi, -bias_col  # negated class-0 view (XGB parity)
        phi = np.where(np.isfinite(phi), phi, 0.0)
        bias_col = np.where(np.isfinite(bias_col), bias_col, 0.0)
        contribs = np.zeros((X_native.shape[0], width), dtype=np.float64)
        take = min(width, phi.shape[1])
        if take > 0:
            contribs[:, :take] = phi[:, :take]
        base_raw = float(np.mean(bias_col)) if bias_col.size else 0.0

        # 2) Leaf indices (edge #3: 1D guard + multiclass reshape).
        raw_leaves = np.asarray(booster.predict(Xt, pred_leaf=True, **_predict_kwargs()))
        if raw_leaves.ndim == 1:  # pragma: no cover - single-tree collapse guard
            raw_leaves = raw_leaves.reshape(X_native.shape[0], -1)
        raw_leaves = np.asarray(raw_leaves, dtype=np.int64)
        if self.is_classifier and n_per_iter > 1 and raw_leaves.shape[1] == max(n_iters, 1) * n_per_iter:
            leaves_3d = raw_leaves.reshape(raw_leaves.shape[0], max(n_iters, 1), n_per_iter)
            leaf_matrix = leaves_3d[:, :, int(min(resolved, n_per_iter - 1))]
        else:
            leaf_matrix = raw_leaves

        # 3) Staged linked trajectory via num_iteration.
        staged = np.zeros((X_native.shape[0], 0), dtype=np.float64)
        step_iters: list[int] = []
        if n_iters > 0:
            k = min(n_iters, MAX_TRAJECTORY_POINTS)
            step_iters = sorted({int(v) for v in np.unique(np.linspace(1, n_iters, k, dtype=int))})
            cols: list[np.ndarray] = []
            for it in step_iters:
                arr = np.asarray(booster.predict(Xt, num_iteration=int(it), **_predict_kwargs()))
                if arr.ndim == 2 and arr.shape[1] > 1:
                    c_idx = int(min(resolved, arr.shape[1] - 1))
                    col = np.asarray(arr[:, c_idx], dtype=np.float64)
                else:
                    col = np.asarray(arr, dtype=np.float64).ravel()
                if self.is_classifier and n_per_iter <= 1 and n_classes == 2 and resolved == 0:
                    col = 1.0 - col
                cols.append(col)
            if cols:
                staged = np.column_stack(cols)
                staged = np.where(np.isfinite(staged), staged, 0.0)

        # 4) Native linked finals for parity display (raw X_native: wrappers
        # apply their own preprocessing; booster-direct paths use Xt above).
        linked_resolved = np.asarray(
            self._linked_predict(predict_obj, booster, X_native, resolved), dtype=np.float64
        ).ravel()
        native_probas: np.ndarray | None = None
        if self.is_classifier:
            native_probas = self._full_proba_dist(predict_obj, booster, X_native)
            native_labels = np.argmax(native_probas, axis=1) if native_probas is not None else (
                np.clip(linked_resolved, 0.0, 1.0) >= 0.5).astype(int)
        else:
            native_labels = np.asarray(linked_resolved, dtype=np.float64).ravel()

        try:
            learning_rate = float(getattr(predict_obj, "learning_rate", 0.1))
        except (TypeError, ValueError):
            learning_rate = 0.1
        k_traj = staged.shape[1]
        sample_idx = [(r - 1) for r in step_iters]
        k_top = min(TOP_STAGES, n_iters)
        k_export = min(leaf_matrix.shape[1], MAX_EXPORTED_TREES) if leaf_matrix.size else 0

        booster_names = _booster_feature_names(booster)
        topo, split_lines, cat_idx, depth_stats = self._topology_parts(booster, width)
        importances = {
            "gain": [round(float(v), 5) for v in self._importances(booster, width, "gain")],
            "split": [round(float(v), 5) for v in self._importances(booster, width, "split")],
        }

        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < X_native.shape[0]):
                continue
            row_phi = contribs[row_id]
            order = np.argsort(np.abs(row_phi))[::-1][:k_top] if width else []
            order_sorted = sorted(int(v) for v in np.asarray(order).tolist())
            stages: list[dict[str, Any]] = []
            for f_pos, f_idx in enumerate(order_sorted):
                stages.append(
                    {
                        "rank": int(f_pos + 1),
                        "feature_idx": int(f_idx),
                        "feature_name": names[f_idx] if f_idx < len(names) else f"feature_{f_idx}",
                        "shap_value": round(float(row_phi[f_idx]), 5),
                    }
                )
            traj = [round(float(staged[row_id, s]), 4) for s in range(k_traj)] if k_traj else []
            row_nan = bool(np.isnan(np.asarray(X_native[row_id], dtype=np.float64)).any())
            leaf_proba = (
                [float(v) for v in native_probas[row_id].tolist()]
                if native_probas is not None and row_id < native_probas.shape[0]
                else []
            )
            try:
                final_pred: Any = native_labels[row_id]
            except (IndexError, TypeError):
                final_pred = classes[resolved] if resolved < len(classes) else resolved
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            final_margin = float(base_raw + float(np.sum(row_phi)))
            # Active leaf depth for HUD gauge (first sampled tree).
            active_depth: int | None = None
            if topo and leaf_matrix.size and row_id < leaf_matrix.shape[0]:
                first = topo[0]
                depth_map = {int(k): int(v) for k, v in first.get("leaf_depths", {}).items()}
                active_depth = depth_map.get(int(leaf_matrix[row_id, 0]))
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "algorithm": type(_final_lgb_estimator(model_obj)).__name__,
                "objective": objective,
                "num_class": int(n_per_iter),
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "model_family": "lightgbm",
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.1,
                "n_iterations": int(n_iters),
                "n_boost_rounds": int(n_iters),
                "n_rounds_total": int(n_iters),
                "n_trees_total": int(n_total),
                "num_leaves": int(topo[0].get("num_leaves", 0)) if topo else 0,
                "negated_target_zero_view": bool(
                    self.is_classifier and n_per_iter <= 1 and n_classes == 2 and resolved == 0),
                "has_missing_values": row_nan,
                "base_raw_score": float(base_raw),
                "final_raw_score": float(final_margin),
                "sampled_iteration_indices": [int(s) for s in sample_idx],
                "staged_trajectory": traj,
                "leaf_indices_sample": (
                    [int(v) for v in leaf_matrix[row_id, :k_export].tolist()]
                    if leaf_matrix.size and row_id < leaf_matrix.shape[0]
                    else []
                ),
                "active_leaf_depth": active_depth,
                "top_shap_features": stages,
                "probabilities": leaf_proba,
                "predicted_class": final_pred,
                "linked_prediction": round(float(linked_resolved[row_id]), 5)
                if row_id < np.asarray(linked_resolved).size
                else None,
                "winning_class_idx": int(resolved) if self.is_classifier else 0,
                "resolved_class_idx": int(resolved),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base_raw),
                    final_prediction=final_pred,  # type: ignore[arg-type]
                    feature_contributions={
                        (names[i] if i < len(names) else f"feature_{i}"): float(row_phi[i])
                        for i in range(int(width))
                    },
                    extra_mechanics=extra,
                )
            )
        for tr in traces:
            tr.extra_mechanics["split_lines_sample"] = split_lines
            tr.extra_mechanics["categorical_feature_idx"] = cat_idx
            tr.extra_mechanics["feature_importances"] = importances
            tr.extra_mechanics["tree_topology_rounds"] = [t["round"] for t in topo]
            tr.extra_mechanics["tree_depth_stats"] = depth_stats
        return traces

    def _topology_parts(
        self, booster: Any, width: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[dict[str, Any]]]:
        """Parse one capped ``dump_model`` pass: sampled topologies + top splits."""
        try:
            dump = booster.dump_model(num_iteration=MAX_EXPORTED_TREES)
        except (AttributeError, TypeError, ValueError):
            return [], [], [], []
        tree_info = dump.get("tree_info", []) or []
        try:
            pandas_cats = dump.get("pandas_categorical", []) or []
        except AttributeError:
            pandas_cats = []
        _ = pandas_cats  # surfaced per-node via categories; kept for parity logging
        topo: list[dict[str, Any]] = []
        splits: list[dict[str, Any]] = []
        cat_features: set[int] = set()
        depth_stats: list[dict[str, Any]] = []
        for r, tree in enumerate(tree_info[:MAX_EXPORTED_TREES]):
            struct = tree.get("tree_structure") if isinstance(tree, dict) else None
            flat, leaf_depths, d_min, d_max = _flatten_lgb_tree(struct)
            try:
                num_leaves = int(tree.get("num_leaves", len(leaf_depths)))
            except (AttributeError, TypeError, ValueError):
                num_leaves = len(leaf_depths)
            try:
                shrinkage = float(tree.get("shrinkage", 0.0))
            except (TypeError, ValueError):
                shrinkage = 0.0
            if r < MAX_TOPOLOGY_TREES:
                topo.append(
                    {"round": int(r), "num_leaves": num_leaves, "shrinkage": shrinkage,
                     "nodes": flat, "leaf_depths": leaf_depths}
                )
                depth_stats.append(
                    {"round": int(r), "min_leaf_depth": int(d_min),
                     "max_leaf_depth": int(d_max), "num_leaves": num_leaves}
                )
            for node in flat:
                if "leaf_index" in node:
                    continue
                f_idx = node.get("split_feature_idx", -1)
                if node.get("categories"):
                    if isinstance(f_idx, int) and f_idx >= 0:
                        cat_features.add(f_idx)
                splits.append(
                    {"round": int(r), "feature_idx": f_idx, "gain": node.get("gain", 0.0),
                     "count": node.get("count", 0)}
                )
        splits.sort(key=lambda s: float(s.get("gain", 0.0)), reverse=True)
        return topo, splits[:MAX_SPLIT_LINES], sorted(cat_features), depth_stats

    # -- surface --------------------------------------------------------
    def _surface_common(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        predict_obj, est = _unwrap_lgb_container(model_obj)
        booster = _as_booster(est)
        if booster is None:
            raise ValueError(f"Estimator '{type(est).__name__}' exposes no lightgbm.Booster")
        objective, num_class = _dump_objective(booster)
        n_total, n_per_iter, n_iters = _booster_counts(booster)
        if _is_native_booster(predict_obj) or predict_obj is booster:
            classes: list[Any] = (
                list(range(num_class)) if objective.startswith("multiclass") and num_class > 1
                else ([0, 1] if self.is_classifier else [])
            )
        else:
            raw_classes = _classes_of(predict_obj)
            classes = raw_classes or ([0, 1] if self.is_classifier else [])
        n_classes = len(classes) if self.is_classifier else 0
        if self.is_classifier and n_per_iter > 1:
            if target_class_idx is not None and n_classes:
                resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
            elif n_classes == 2:
                resolved = 1
            else:
                resolved = int(min(max(int(target_idx), 0), max(n_per_iter - 1, 0)))
        else:
            if self.is_classifier and n_classes == 2 and target_class_idx is not None:
                resolved = int(min(max(int(target_class_idx), 0), 1))
            else:
                resolved = 1 if (self.is_classifier and n_classes == 2) else 0
        try:
            learning_rate = float(getattr(predict_obj, "learning_rate", 0.1))
        except (TypeError, ValueError):
            learning_rate = 0.1

        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        n_points = res * res
        grid = np.zeros((n_points, max(int(n_features), 1)), dtype=np.float32)
        for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
            if 0 <= feat_idx < grid.shape[1] and np.isfinite(med_val):
                grid[:, feat_idx] = np.float32(med_val)
        ax = grid_spec.axis_x_feature_index
        ay = grid_spec.axis_y_feature_index
        xx, yy = np.meshgrid(x_lin, y_lin, indexing="xy")
        if 0 <= ax < grid.shape[1]:
            grid[:, ax] = xx.ravel()
        if int(n_features) > 1 and ay != ax and 0 <= ay < grid.shape[1]:
            grid[:, ay] = yy.ravel()
        try:
            linked = np.asarray(self._linked_predict(predict_obj, booster, grid, resolved))
            if linked.ndim == 2 and linked.shape[1] > 1:
                c_idx = int(min(resolved, linked.shape[1] - 1))
                z_surface = np.asarray(linked[:, c_idx], dtype=np.float64).reshape(res, res)
            else:
                flat = np.asarray(linked, dtype=np.float64).ravel()
                z_surface = flat.reshape(res, res) if flat.size == n_points else np.zeros((res, res))
            if self.is_classifier and n_per_iter <= 1 and n_classes == 2 and resolved == 0:
                z_surface = 1.0 - z_surface
        except (ValueError, AttributeError, TypeError):
            z_surface = np.zeros((res, res), dtype=np.float64)
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
        topo, split_lines, cat_idx, depth_stats = self._topology_parts(booster, int(n_features))
        importances = {
            "gain": [round(float(v), 5) for v in self._importances(booster, int(n_features), "gain")],
            "split": [round(float(v), 5) for v in self._importances(booster, int(n_features), "split")],
        }
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin],
            y_grid=[float(v) for v in y_lin],
            z_values=[float(v) for v in np.asarray(z_surface, dtype=np.float64).ravel()],
            extra_overlays={
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes],
                "algorithm": type(_final_lgb_estimator(model_obj)).__name__,
                "objective": objective,
                "num_class": int(n_per_iter),
                "model_family": "lightgbm",
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.1,
                "n_iterations": int(n_iters),
                "n_boost_rounds": int(n_iters),
                "n_rounds_total": int(n_iters),
                "n_trees_total": int(n_total),
                "feature_importances": importances,
                "tree_topology_sample": topo,
                "tree_topology_rounds": [t["round"] for t in topo],
                "tree_depth_stats": depth_stats,
                "split_lines_sample": split_lines,
                "categorical_feature_idx": cat_idx,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("lightgbm")
class LightGBMClassifierAdapter(_LightGBMBase):
    display_name = "LightGBM (Leaf-Wise Exact TreeSHAP)"
    family_name = "Gradient Boosting"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = True
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        if not _LGB_AVAILABLE:
            return False
        final_est = _final_lgb_estimator(estimator)
        if isinstance(final_est, _CLASSIFIER_TYPES):
            return True
        booster = _as_booster(final_est)
        if booster is None:
            return False
        objective, num_class = _dump_objective(booster)
        return objective.startswith(("binary", "multiclass", "multiclassova")) or num_class > 1

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
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)


@ModelRegistry.register("lightgbm_regressor")
class LightGBMRegressorAdapter(_LightGBMBase):
    display_name = "LightGBM Regressor/Ranker (Leaf-Wise Exact TreeSHAP)"
    family_name = "Gradient Boosting"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = False
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        if not _LGB_AVAILABLE:
            return False
        final_est = _final_lgb_estimator(estimator)
        if isinstance(final_est, _REGRESSOR_TYPES):
            return True
        booster = _as_booster(final_est)
        if booster is None:
            return False
        objective, num_class = _dump_objective(booster)
        if objective.startswith(("binary", "multiclass", "multiclassova")) or num_class > 1:
            return False
        return True

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
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)

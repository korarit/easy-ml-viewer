# backend/app/domains/model_engine/adapters/boosting/hist_gbm.py
"""Phase 32 — HistGradientBoosting staged-residual adapters (quantized bin terraces).

Covers ``HistGradientBoostingClassifier`` and
``HistGradientBoostingRegressor``, plus ``Pipeline`` /
``GridSearchCV``-style ``best_estimator_`` / ``TransformedTargetRegressor``
wrappers — inference runs through the full container while staged tracing
unwraps to the fitted booster.

Blueprint mapping (phase-32-model-hist-gradient-boosting.md):

- No ``.estimators_`` / ``.decision_path()`` / ``.apply()`` here: all
  trees live in ``model._predictors`` (``(n_iter, n_trees_per_iter)``,
  ``1`` column for binary/regression, ``C`` for multiclass) as Cython
  ``TreePredictor`` objects exposing the ``nodes`` structured array
  (blueprint §2.2). A NumPy walker replays each predictor exactly:
  NaN routes via ``missing_go_to_left`` first (edge #2), categorical
  nodes test the raw bin against ``binned_left_cat_bitsets`` (edge #3),
  numeric nodes go left iff ``binned <= bin_threshold``.
- ``nodes['value']`` already carries the ``learning_rate`` shrink applied
  at grow time — it is NEVER multiplied again (edge #4). Base score
  ``F_0`` comes from ``model._baseline_prediction`` (edge #6).
- Per-edge deltas ``Δ = nodes[v]['value'] - nodes[u]['value']`` are
  credited to the split feature, so ``base + ΣΦ == F_M`` holds exactly in
  raw space; the probability curve resolves through sklearn's own
  ``staged_predict_proba`` (every ``loss_`` link handled natively).
- No ``feature_importances_`` exists (edge #1): global importances are
  summed live from ``nodes['gain']`` over every internal node (<1 ms).
- Real iteration count always comes from ``len(model._predictors)``
  (edge #5, early stopping). Path attribution walks the first
  ``MAX_TRACED_ITERS``; later iterations distribute uniformly so the
  waterfall identity stays exact for ``max_iter > 100`` models.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
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

MAX_TRAJECTORY_POINTS = 50
MAX_TRACED_ITERS = 100
MAX_SURFACE_RESOLUTION = 100
MAX_BIN_EXPORT = 64
MAX_ACTIVE_SPLITS = 200
TOP_STAGES = 8

_CLASSIFIER_TYPES = (HistGradientBoostingClassifier,)
_REGRESSOR_TYPES = (HistGradientBoostingRegressor,)


def _unwrap_hist_container(model_obj: Any) -> tuple[Any, Any]:
    """Return ``(predict_obj, booster_estimator)``.

    Follows ``GridSearchCV``-style ``best_estimator_`` chains, unwraps
    ``Pipeline`` ``steps``, and follows ``TransformedTargetRegressor``
    ``regressor_``.
    """
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


def _final_hist_estimator(model_obj: Any) -> Any:
    return _unwrap_hist_container(model_obj)[1]


def _is_sparse(matrix: Any) -> bool:
    try:
        import scipy.sparse as _sp  # noqa: PLC0415

        return bool(_sp.issparse(matrix))
    except ImportError:
        return False


def _as_dense_float32(matrix: Any) -> np.ndarray:
    if _is_sparse(matrix):
        matrix = matrix.toarray()
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))


def _transform_for_estimators(predict_obj: Any, X_f32: np.ndarray) -> np.ndarray:
    """Apply pipeline preprocessing (all steps but last) so bins align."""
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


def _is_ttr(predict_obj: Any, booster: Any) -> bool:
    return predict_obj is not booster and hasattr(predict_obj, "transformer_")


def _inverse_ttr(predict_obj: Any, values: np.ndarray) -> np.ndarray:
    """Map staged regressor outputs back through a TTR transformer (best-effort)."""
    try:
        transformer = getattr(predict_obj, "transformer_", None)
        if transformer is None or values.size == 0:
            return values
        flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        back = np.asarray(transformer.inverse_transform(flat), dtype=np.float64).ravel()
        return back.reshape(values.shape)
    except (ValueError, AttributeError, TypeError):
        return values


def _binned_matrix(booster: Any, Xt: np.ndarray) -> np.ndarray | None:
    """Bin the (preprocessed) batch with the fitted ``_bin_mapper``.

    Returns ``None`` when the private mapper is unavailable so callers can
    fall back to raw-threshold walking.
    """
    try:
        mapper = getattr(booster, "_bin_mapper", None)
        if mapper is None or not hasattr(mapper, "transform"):
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            binned = np.asarray(mapper.transform(Xt))
        if binned.ndim != 2 or binned.shape[0] != Xt.shape[0]:
            return None
        return binned
    except (ValueError, AttributeError, TypeError):
        return None


def _walk_predictor(
    predictor: Any,
    x_raw: np.ndarray,
    x_bin: np.ndarray | None,
    contribs_row: np.ndarray,
    width: int,
) -> tuple[int, float, int | None]:
    """Walk one ``TreePredictor`` for a single row.

    Returns ``(leaf_id, leaf_value, nan_feature_idx)`` where the last item
    records which feature routed a NaN at its split (``None`` when the row
    held no NaN on this path). ``contribs_row`` accumulates per-edge
    ``nodes[v]['value'] - nodes[u]['value']`` into the split feature.
    """
    nodes = predictor.nodes
    try:
        bitsets = [np.asarray(b, dtype=np.uint32).ravel() for b in predictor.binned_left_cat_bitsets]
    except (AttributeError, TypeError, ValueError):
        bitsets = []
    u = 0
    nan_feat: int | None = None
    guard = 0
    while True:
        try:
            is_leaf = bool(nodes[u]["is_leaf"])
        except (IndexError, TypeError, ValueError):
            break
        if is_leaf:
            break
        guard += 1
        if guard > 4096:  # pragma: no cover - corrupt node graph guard
            break
        try:
            f_idx = int(nodes[u]["feature_idx"])
            go_left_flag = bool(nodes[u]["missing_go_to_left"])
            is_cat = bool(nodes[u]["is_categorical"])
            left = int(nodes[u]["left"])
            right = int(nodes[u]["right"])
            node_val = float(nodes[u]["value"])
        except (TypeError, ValueError):
            break
        x_val = float(x_raw[f_idx]) if 0 <= f_idx < x_raw.size else np.nan
        if np.isnan(x_val):
            go_left = go_left_flag  # edge #2: NaN never compares <= threshold
            if 0 <= f_idx < width and nan_feat is None:
                nan_feat = f_idx
        elif is_cat and x_bin is not None and 0 <= f_idx < x_bin.size:
            try:
                bit_idx = int(nodes[u]["bitset_idx"])
                b = int(x_bin[f_idx])
                arr = bitsets[bit_idx] if 0 <= bit_idx < len(bitsets) else None
                go_left = bool((int(arr[b // 32]) >> (b % 32)) & 1) if arr is not None else bool(
                    x_val <= float(nodes[u]["num_threshold"])
                )
            except (TypeError, ValueError, IndexError):
                go_left = bool(x_val <= float(nodes[u]["num_threshold"]))
        elif x_bin is not None and 0 <= f_idx < x_bin.size:
            try:
                go_left = bool(int(x_bin[f_idx]) <= int(nodes[u]["bin_threshold"]))
            except (TypeError, ValueError):
                go_left = bool(x_val <= float(nodes[u]["num_threshold"]))
        else:
            go_left = bool(x_val <= float(nodes[u]["num_threshold"]))
        v = left if go_left else right
        try:
            child_val = float(nodes[v]["value"])
        except (TypeError, ValueError, IndexError):
            break
        if 0 <= f_idx < width:
            delta = child_val - node_val
            if np.isfinite(delta):
                contribs_row[f_idx] += delta
        u = v
    try:
        leaf_val = float(nodes[u]["value"])
    except (TypeError, ValueError, IndexError):
        leaf_val = 0.0
    return int(u), (leaf_val if np.isfinite(leaf_val) else 0.0), nan_feat


def _link_function(booster: Any, is_classifier: bool, n_classes: int) -> str:
    """Raw→prediction link: ``logit``/``softmax`` (classifier),
    ``log`` for gamma/poisson else ``identity`` (regressor)."""
    if is_classifier:
        return "softmax" if n_classes > 2 else "logit"
    loss_name = str(getattr(booster, "loss_", getattr(booster, "loss", "")))
    return "log" if loss_name in ("gamma", "poisson") else "identity"


def _gain_importances(booster: Any, width: int) -> list[float]:
    """Global importances from summed ``nodes['gain']`` (edge #1)."""
    gains = np.zeros(max(width, 0), dtype=np.float64)
    try:
        predictors = list(getattr(booster, "_predictors", []) or [])
    except TypeError:
        return [0.0 for _ in range(max(width, 0))]
    for iter_preds in predictors:
        try:
            cols = list(iter_preds)
        except TypeError:
            continue
        for pred in cols:
            try:
                nodes = pred.nodes
                internal = np.asarray(nodes["is_leaf"]) == 0
                feats = np.asarray(nodes["feature_idx"])[internal]
                vals = np.asarray(nodes["gain"], dtype=np.float64)[internal]
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
            for f_idx, g in zip(feats.tolist(), vals.tolist()):
                f = int(f_idx)
                if 0 <= f < gains.size and np.isfinite(g):
                    gains[f] += float(g)
    total = float(np.sum(gains))
    if total > 0:
        gains = gains / total
    return [round(float(v), 5) for v in gains.tolist()]


def _export_bin_thresholds(booster: Any) -> dict[str, list[float]]:
    """Downsampled ``_bin_mapper.bin_thresholds_`` (≤ ``MAX_BIN_EXPORT`` per feature)."""
    out: dict[str, list[float]] = {}
    try:
        thresholds = list(getattr(getattr(booster, "_bin_mapper", None), "bin_thresholds_", []) or [])
    except TypeError:
        return out
    for f_idx, th in enumerate(thresholds):
        try:
            arr = np.asarray(th, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        step = max(1, int(-(-arr.size // MAX_BIN_EXPORT)))  # ceil: output <= cap
        out[str(f_idx)] = [round(float(v), 5) for v in arr[::step].tolist()]
    return out


class _HistGBMBase(BaseModelAdapter):
    is_classifier: bool = True
    supports_nan_input: bool = True

    def _resolve_target(
        self, booster: Any, n_cols: int, target_idx: int, target_class_idx: int | None
    ) -> tuple[int, int, bool, list[Any]]:
        """Return ``(resolved_class, tree_col, negate_binary, classes)``.

        Binary HistGBM grows the positive-class residual tree in column 0;
        an explicit ``target_class_idx == 0`` negates the raw margin so the
        waterfall still sums exactly (Phase 31 §edge #5 analogue).
        """
        classes = _classes_of(booster)
        n_classes = len(classes)
        if self.is_classifier and n_cols == 1:
            if target_class_idx is not None:
                negate = bool(int(target_class_idx) == 0 and n_classes == 2)
            else:
                negate = False
            return 1 if n_classes == 2 else 0, 0, negate, classes
        if self.is_classifier:
            if target_class_idx is not None and n_classes:
                resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
            elif n_classes == 2:
                resolved = 1
            else:
                resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
            tree_col = int(min(max(resolved, 0), max(n_cols - 1, 0)))
            return resolved, tree_col, False, classes
        return 0, 0, False, classes

    def _predictors_of(self, booster: Any) -> tuple[list[list[Any]], int, int]:
        try:
            predictors = list(getattr(booster, "_predictors", []) or [])
        except TypeError:
            predictors = []
        cols: list[list[Any]] = []
        for it in predictors:
            try:
                cols.append(list(it))
            except TypeError:
                cols.append([])
        n_iter = len(cols)
        n_cols = len(cols[0]) if n_iter and cols[0] else 0
        return cols, n_iter, n_cols

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        est = _final_hist_estimator(estimator)
        width = max(int(n_features), 0)
        gains = _gain_importances(est, width)
        if len(gains) != width:
            return [None for _ in range(width)]
        return [float(v) for v in gains]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, _ = _unwrap_hist_container(model_obj)
        X_in = _as_dense_float32(X_float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                preds = np.asarray(predict_obj.predict(X_in)).ravel()
            except (ValueError, AttributeError, TypeError):
                preds = np.zeros((X_in.shape[0],))
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if self.is_classifier:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    probas = np.asarray(predict_obj.predict_proba(X_in), dtype=np.float64)
                if probas.ndim == 1:
                    probas = probas.reshape(-1, 1)
                probabilities = [
                    [float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()
                ]
                confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            except (AttributeError, NotImplementedError, ValueError, TypeError):
                probabilities, confidences = None, None
        py_preds: list[float | int | str] = []
        for v in preds.tolist():
            if isinstance(v, (bool, np.bool_)):
                py_preds.append(int(v))
            elif isinstance(v, (np.integer, int)):
                py_preds.append(int(v))
            elif isinstance(v, (np.floating, float)):
                f = float(v)
                py_preds.append(int(f) if f.is_integer() else f)
            else:
                py_preds.append(v)
        return PredictionBatchResult(
            predictions=py_preds, probabilities=probabilities, confidences=confidences
        )

    def _trace_common(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        predict_obj, booster = _unwrap_hist_container(model_obj)
        predictors, n_iter, n_cols = self._predictors_of(booster)
        if n_iter == 0 or n_cols == 0:
            raise ValueError(f"Estimator '{type(booster).__name__}' exposes no fitted _predictors")
        resolved, tree_col, negate, classes = self._resolve_target(
            booster, n_cols, target_idx, target_class_idx
        )
        n_classes = len(classes)
        is_binary = bool(self.is_classifier and n_classes == 2)
        ttr_reg = bool(not self.is_classifier and _is_ttr(predict_obj, booster))

        X_native = _as_dense_float32(X_float32)
        Xt = _transform_for_estimators(predict_obj, X_native)
        n_features = int(X_native.shape[1]) if X_native.ndim == 2 else 0
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        try:
            base_vec = np.asarray(getattr(booster, "_baseline_prediction"), dtype=np.float64).ravel()
        except (AttributeError, TypeError, ValueError):
            base_vec = np.zeros((max(n_cols, 1),), dtype=np.float64)
        base_raw = float(base_vec[min(tree_col, base_vec.size - 1)]) if base_vec.size else 0.0
        if not np.isfinite(base_raw):
            base_raw = 0.0
        if negate:
            base_raw = -base_raw

        try:
            learning_rate = float(getattr(booster, "learning_rate", 0.1))
        except (TypeError, ValueError):
            learning_rate = 0.1

        binned = _binned_matrix(booster, Xt)
        n_rows = X_native.shape[0]
        width = max(n_features, 0)

        # Stage leaf values for ALL iters (cheap: the walker lands on the
        # leaf regardless), path attribution for the first MAX_TRACED_ITERS,
        # uniform fallback beyond (exact waterfall for max_iter > 100).
        traced = min(n_iter, MAX_TRACED_ITERS)
        stage_deltas = np.zeros((n_rows, n_iter), dtype=np.float64)
        leaf_matrix = np.zeros((n_rows, n_iter), dtype=np.int64)
        nan_routing: list[list[int | None]] = [[None] * n_iter for _ in range(n_rows)]
        sum_c = np.zeros((n_rows, width), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            for m in range(n_iter):
                cols_m = predictors[m] if m < len(predictors) else []
                if tree_col >= len(cols_m):
                    continue
                pred_m = cols_m[tree_col]
                for i in range(n_rows):
                    x_raw = np.asarray(Xt[i], dtype=np.float64).ravel()
                    x_bin = (
                        np.asarray(binned[i]).ravel()
                        if binned is not None and i < binned.shape[0]
                        else None
                    )
                    if m < traced:
                        row_c = np.zeros((width,), dtype=np.float64)
                        leaf_id, leaf_val, nan_feat = _walk_predictor(
                            pred_m, x_raw, x_bin, row_c, width
                        )
                        # Class-0 view negates the positive-class margin: the
                        # path attribution must flip sign with the deltas.
                        sum_c[i] += -row_c if negate else row_c
                    else:
                        leaf_id, leaf_val, nan_feat = _walk_predictor(
                            pred_m, x_raw, x_bin, np.zeros((0,), dtype=np.float64), 0
                        )
                        if width:
                            signed = -leaf_val if negate else leaf_val
                            sum_c[i, :] += signed / width
                    d = -leaf_val if negate else leaf_val
                    stage_deltas[i, m] = d
                    leaf_matrix[i, m] = leaf_id
                    nan_routing[i][m] = nan_feat

        staged = self._staged_matrix(predict_obj, booster, X_native, resolved, negate)
        if staged.shape[1] != n_iter and staged.size:
            n_iter = int(staged.shape[1])
            stage_deltas = stage_deltas[:, :n_iter]
            leaf_matrix = leaf_matrix[:, :n_iter]
        if ttr_reg and staged.size:
            with np.errstate(invalid="ignore"):
                ttr_base = float(np.mean(staged[:, 0]))
            base_raw = ttr_base if np.isfinite(ttr_base) else 0.0
            prev = np.full((staged.shape[0], 1), base_raw, dtype=np.float64)
            stage_deltas = np.diff(np.hstack([prev, staged]), axis=1)
            # Re-attribute TTR-space deltas uniformly (original units are
            # nonlinear in the grown leaves; uniform keeps ΣΦ exact).
            sum_c[:, :] = 0.0
            if width:
                for m in range(n_iter):
                    sum_c[:, :] += (stage_deltas[:, m].reshape(-1, 1) / width) * np.ones((1, width))

        try:
            native_preds = list(np.asarray(predict_obj.predict(X_native)).ravel().tolist())
        except (ValueError, AttributeError, TypeError):
            native_preds = [classes[0] if classes else 0 for _ in row_indices]
        native_probas: np.ndarray | None = None
        if self.is_classifier:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    native_probas = np.asarray(predict_obj.predict_proba(X_native), dtype=np.float64)
            except (AttributeError, NotImplementedError, ValueError, TypeError):
                native_probas = None

        k_traj = min(n_iter, MAX_TRAJECTORY_POINTS)
        step_indices = (
            np.unique(np.linspace(0, n_iter - 1, k_traj, dtype=int)).tolist() if n_iter else []
        )
        k_top = min(TOP_STAGES, n_iter)
        k_export = min(n_iter, MAX_TRACED_ITERS)

        try:
            mono_raw = getattr(booster, "monotonic_cst", None)
            mono = list(mono_raw) if isinstance(mono_raw, (list, tuple, np.ndarray)) else []
            monotonic = [int(v) for v in mono if int(v) in (-1, 0, 1)]
        except (TypeError, ValueError):
            monotonic = []
        try:
            # NOTE: the unfitted ``categorical_features`` param defaults to
            # the string ``'from_dtype'`` — always read a fitted boolean
            # mask (``is_categorical_`` in modern sklearn) instead.
            cat_raw = getattr(booster, "is_categorical_", None)
            if cat_raw is None:
                cat_raw = getattr(booster, "categorical_features_", None)
            if cat_raw is None or isinstance(cat_raw, str):
                categorical_idx = []
            else:
                cat_mask = list(np.asarray(cat_raw).ravel().tolist())
                categorical_idx = [int(i) for i, v in enumerate(cat_mask) if bool(v)]
        except (TypeError, ValueError):
            categorical_idx = []
        gain_importances = _gain_importances(booster, width)
        bin_thresholds = _export_bin_thresholds(booster)

        traces: list[InternalFlowRowTrace] = []
        for row_id in row_indices:
            if not (0 <= row_id < n_rows):
                continue
            contribs = sum_c[row_id]
            try:
                final_pred = native_preds[row_id]
            except (IndexError, TypeError):
                final_pred = classes[resolved] if resolved < len(classes) else resolved
            if isinstance(final_pred, (np.integer, bool, np.bool_)):
                final_pred = int(final_pred)
            elif isinstance(final_pred, (np.floating, float)):
                f = float(final_pred)
                final_pred = int(f) if f.is_integer() else f
            leaf_proba = (
                [float(v) for v in native_probas[row_id].tolist()]
                if native_probas is not None and row_id < native_probas.shape[0]
                else []
            )
            row_deltas = stage_deltas[row_id] if n_iter else np.zeros((0,))
            top_idx = np.argsort(np.abs(row_deltas))[::-1][:k_top] if n_iter else []
            top_sorted = sorted(int(v) for v in np.asarray(top_idx).tolist())
            running = float(base_raw)
            cum_by_stage: list[float] = []
            for mm in range(n_iter):
                running += float(row_deltas[mm])
                cum_by_stage.append(running)
            stages: list[dict[str, Any]] = []
            for mm in top_sorted:
                stages.append(
                    {
                        "stage": int(mm),
                        "leaf_index": int(leaf_matrix[row_id, mm]) if mm < leaf_matrix.shape[1] else -1,
                        "delta_raw": round(float(row_deltas[mm]), 5),
                        "cumulative_raw": round(float(cum_by_stage[mm]), 5),
                        "cumulative_pred": round(float(staged[row_id, mm]), 5)
                        if staged.size and mm < staged.shape[1]
                        else None,
                        "nan_routed_feature_idx": nan_routing[row_id][mm]
                        if row_id < len(nan_routing) and mm < len(nan_routing[row_id])
                        else None,
                    }
                )
            traj = [round(float(staged[row_id, s]), 4) for s in step_indices] if staged.size else []
            final_raw = float(base_raw + float(np.sum(row_deltas)))
            row_nan = bool(np.isnan(np.asarray(X_native[row_id], dtype=np.float64)).any())
            loss_name = str(getattr(booster, "loss_", getattr(booster, "loss", "")))
            link_function = _link_function(booster, self.is_classifier, n_classes)
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(is_binary),
                "algorithm": type(booster).__name__,
                "loss": loss_name,
                "link_function": link_function,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "model_family": "hist_gbm",
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.1,
                "n_iter_total": int(n_iter),
                "n_iter_traced": int(traced),
                "negated_target_zero_view": bool(negate),
                "base_raw_score": float(base_raw),
                "final_raw_score": float(final_raw),
                "sampled_stage_indices": [int(s) for s in step_indices],
                "staged_predictions": traj,
                "stage_raw_deltas_sample": [
                    round(float(v), 5) for v in row_deltas[:k_export].tolist()
                ],
                "leaf_indices_per_stage": [
                    int(v) for v in leaf_matrix[row_id, :k_export].tolist()
                ]
                if n_iter
                else [],
                "top_residual_stages": stages,
                "has_missing_values": row_nan,
                "monotonic_cst": monotonic,
                "categorical_feature_idx": categorical_idx,
                "gain_feature_importances": gain_importances,
                "probabilities": leaf_proba,
                "predicted_class": final_pred,
                "winning_class_idx": int(resolved) if self.is_classifier else 0,
                "resolved_class_idx": int(resolved),
            }
            traces.append(
                InternalFlowRowTrace(
                    row_index=int(row_id),
                    base_value=float(base_raw),
                    final_prediction=final_pred,  # type: ignore[arg-type]
                    feature_contributions={
                        (names[i] if i < len(names) else f"feature_{i}"): float(contribs[i])
                        for i in range(int(width))
                    },
                    extra_mechanics=extra,
                )
            )
        return traces

    def _staged_matrix(
        self,
        predict_obj: Any,
        booster: Any,
        X_raw_f32: np.ndarray,
        resolved: int,
        negate_binary: bool,
    ) -> np.ndarray:
        """Staged predictions ``(n_samples, n_iter)`` for ``resolved``."""
        cols: list[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                if self.is_classifier:
                    staged = booster.staged_predict_proba(
                        _transform_for_estimators(predict_obj, X_raw_f32)
                    )
                    for proba in staged:
                        p = np.asarray(proba, dtype=np.float64)
                        if p.ndim == 1:
                            p = p.reshape(-1, 1)
                        idx = int(min(max(resolved, 0), p.shape[1] - 1)) if p.size else 0
                        cols.append(p[:, idx] if p.size else np.zeros((X_raw_f32.shape[0],)))
                else:
                    staged = booster.staged_predict(
                        _transform_for_estimators(predict_obj, X_raw_f32)
                    )
                    for pred in staged:
                        cols.append(np.asarray(pred, dtype=np.float64).ravel())
            except (AttributeError, ValueError, TypeError):
                cols = []
        if not cols:
            return np.zeros((X_raw_f32.shape[0], 0), dtype=np.float64)
        mat = np.column_stack(cols)
        if self.is_classifier and negate_binary:
            mat = 1.0 - mat
        if not self.is_classifier and _is_ttr(predict_obj, booster):
            mat = _inverse_ttr(predict_obj, mat)
        return np.where(np.isfinite(mat), mat, 0.0)

    def _surface_common(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        predict_obj, booster = _unwrap_hist_container(model_obj)
        _, n_iter, n_cols = self._predictors_of(booster)
        resolved, _tree_col, negate, classes = self._resolve_target(
            booster, n_cols, target_idx, target_class_idx
        )
        n_classes = len(classes)
        try:
            learning_rate = float(getattr(booster, "learning_rate", 0.1))
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                if self.is_classifier:
                    probas = np.asarray(predict_obj.predict_proba(grid), dtype=np.float64)
                    c_idx = int(min(resolved, probas.shape[1] - 1)) if probas.size else 0
                    z_surface = (
                        np.asarray(probas[:, c_idx], dtype=np.float64).reshape(res, res)
                        if probas.size
                        else np.zeros((res, res))
                    )
                    if negate:
                        z_surface = 1.0 - z_surface
                else:
                    preds = np.asarray(predict_obj.predict(grid), dtype=np.float64).ravel()
                    z_surface = preds.reshape(res, res) if preds.size == n_points else np.zeros((res, res))
            except (ValueError, AttributeError, TypeError):
                z_surface = np.zeros((res, res), dtype=np.float64)
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
        try:
            mono_raw = getattr(booster, "monotonic_cst", None)
            mono = list(mono_raw) if isinstance(mono_raw, (list, tuple, np.ndarray)) else []
            monotonic = [int(v) for v in mono if int(v) in (-1, 0, 1)]
        except (TypeError, ValueError):
            monotonic = []
        try:
            cat_raw = getattr(booster, "is_categorical_", None)
            if cat_raw is None:
                cat_raw = getattr(booster, "categorical_features_", None)
            if cat_raw is None or isinstance(cat_raw, str):
                categorical_idx = []
            else:
                cat_mask = list(np.asarray(cat_raw).ravel().tolist())
                categorical_idx = [int(i) for i, v in enumerate(cat_mask) if bool(v)]
        except (TypeError, ValueError):
            categorical_idx = []
        bin_thresholds = _export_bin_thresholds(booster)
        axis_bins = {
            "x": bin_thresholds.get(str(ax), []),
            "y": bin_thresholds.get(str(ay), []),
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
                "algorithm": type(booster).__name__,
                "loss": str(getattr(booster, "loss_", getattr(booster, "loss", ""))),
                "link_function": _link_function(booster, self.is_classifier, n_classes),
                "model_family": "hist_gbm",
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.1,
                "n_iter_total": int(n_iter),
                "n_estimators_actual": int(n_iter),
                "gain_feature_importances": _gain_importances(booster, int(n_features)),
                "bin_thresholds": bin_thresholds,
                "axis_bin_thresholds": axis_bins,
                "monotonic_cst": monotonic,
                "categorical_feature_idx": categorical_idx,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("hist_gbm")
class HistGBMClassifierAdapter(_HistGBMBase):
    display_name = "Hist Gradient Boosting (Quantized Bins)"
    family_name = "Gradient Boosting"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = True
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_hist_estimator(estimator), _CLASSIFIER_TYPES)

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


@ModelRegistry.register("hist_gbm_regressor")
class HistGBMRegressorAdapter(_HistGBMBase):
    display_name = "Hist Gradient Boosting Regressor (Quantized Bins)"
    family_name = "Gradient Boosting"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = False
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        return isinstance(_final_hist_estimator(estimator), _REGRESSOR_TYPES)

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

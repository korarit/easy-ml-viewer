# backend/app/domains/model_engine/adapters/boosting/catboost_adapter.py
"""Phase 35 — CatBoost symmetric oblivious-tree adapters (native Pool + C++ SHAP).

Covers ``CatBoostClassifier`` / ``CatBoostRegressor``, native ``.cbm``
artifacts (auto-detected Classifier vs Regressor via ``loss_function``
probing) and ``.pkl`` / ``.joblib`` wrappers — inference runs through the
full container while staged tracing unwraps to the fitted CatBoost estimator.

Blueprint mapping (phase-35-model-catboost.md):

- 2-stage ``.cbm`` probing: load with ``CatBoostClassifier`` then inspect
  ``get_all_params()["loss_function"]``; regression losses
  (``RMSE``/``MAE``/``Quantile``/...) switch to ``CatBoostRegressor`` (§2.2).
- Single ``catboost.Pool`` per chunk preserves native categorical columns
  as ``str`` (``__MISSING__`` for NaN) with numerics cast to ``float32``
  (§2.3, §6.1); the same pool feeds ``predict``/``calc_leaf_indexes`` /
  ``get_feature_importance`` then is deleted.
- Exact C++ TreeSHAP via ``get_feature_importance(type="ShapValues")``:
  ``(N, F+1)`` for binary/regression (last col = base value) and
  ``(N, C, F+1)`` for multiclass with winning-class slice (§2.3).
- Oblivious level splits parsed from ``_get_tree_splits(tree_idx, pool)``
  (``"<feat>, bin=<thr>"`` numeric / ``"<feat>, value=<cat>"``
  categorical) into symmetric orthogonal grid lines (§2.1).
- ``thread_count = max(1, physical_cores - 1)`` caps the C++ runtime (§6.2);
  SHAP downcast to ``float32`` and leaves to ``uint16`` (§6.3).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

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

try:  # Phase 35: catboost is optional at import time (registry stays importable).
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    _CB_AVAILABLE = True
    _CLASSIFIER_TYPES = (CatBoostClassifier,)
    _REGRESSOR_TYPES = (CatBoostRegressor,)
except ImportError:  # pragma: no cover - exercised only without catboost
    CatBoostClassifier = None  # type: ignore[assignment,misc]
    CatBoostRegressor = None  # type: ignore[assignment,misc]
    Pool = None  # type: ignore[assignment,misc]
    _CB_AVAILABLE = False
    _CLASSIFIER_TYPES = ()
    _REGRESSOR_TYPES = ()

MAX_TRAJECTORY_POINTS = 50
MAX_EXPORTED_TREES = 100
MAX_TOPOLOGY_TREES = 10
MAX_SPLIT_LINES = 30
MAX_SURFACE_RESOLUTION = 100
TOP_STAGES = 8

_REGRESSION_LOSSES = frozenset(
    {
        "RMSE",
        "MAE",
        "MAPE",
        "Poisson",
        "Quantile",
        "LogLinQuantile",
        "RMSEWithUncertainty",
        "Huber",
        "Tweedie",
        "Lq",
    }
)

_NUM_SPLIT_RE = re.compile(r"^(?P<feat>.+),\s*bin=(?P<thr>[-+eE0-9.]+)\s*$")
_CAT_SPLIT_RE = re.compile(r"^(?P<feat>.+),\s*value=(?P<val>.*)\s*$")


def _thread_count() -> int:
    try:
        import psutil  # noqa: PLC0415

        n = int(psutil.cpu_count(logical=False) or 0)
        if n > 1:
            return max(1, n - 1)
    except (ImportError, ValueError, TypeError):
        pass
    return 2


def _unwrap_cb_container(model_obj: Any) -> tuple[Any, Any]:
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


def _final_cb_estimator(model_obj: Any) -> Any:
    return _unwrap_cb_container(model_obj)[1]


def _is_catboost_obj(obj: Any) -> bool:
    if not _CB_AVAILABLE:
        return False
    if isinstance(obj, _CLASSIFIER_TYPES + _REGRESSOR_TYPES):
        return True
    # Duck-type for unpickled / native-loaded estimators.
    return type(obj).__name__ in ("CatBoostClassifier", "CatBoostRegressor")


def _loss_of(est: Any) -> str:
    try:
        params = est.get_all_params()
        return str(params.get("loss_function", "") or "")
    except (AttributeError, TypeError, ValueError):
        return ""


def _is_regression_estimator(est: Any) -> bool:
    if isinstance(est, _REGRESSOR_TYPES):
        return True
    if isinstance(est, _CLASSIFIER_TYPES):
        return False
    name = type(est).__name__
    if name == "CatBoostRegressor":
        return True
    if name == "CatBoostClassifier":
        return False
    loss = _loss_of(est)
    base = loss.split(":")[0].strip()
    return base in _REGRESSION_LOSSES


def _cat_indices(est: Any) -> list[int]:
    try:
        raw = est.get_cat_feature_indices()
        if raw is None:
            return []
        return sorted({int(v) for v in list(raw)})
    except (AttributeError, TypeError, ValueError):
        return []


def _feature_names(est: Any, n: int) -> list[str]:
    for attr in ("feature_names_", "feature_names_in_", "feature_names"):
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


def _to_frame(X: np.ndarray, feature_names: list[str], cat_idx: list[int]) -> pd.DataFrame:
    """Build a mixed-dtype frame: categoricals as str, numerics as float32."""
    n = int(X.shape[1]) if X.ndim == 2 else len(feature_names)
    names = feature_names if len(feature_names) == n else [f"feature_{i}" for i in range(n)]
    cat_set = set(cat_idx)
    data: dict[str, Any] = {}
    for j in range(n):
        col = X[:, j] if X.ndim == 2 else np.asarray(X).ravel()
        if j in cat_set:
            vals: list[str] = []
            for v in list(col):
                try:
                    import math  # noqa: PLC0415

                    if v is None or (isinstance(v, float) and (math.isnan(v) or v != v)):
                        vals.append("__MISSING__")
                        continue
                except (TypeError, ValueError):
                    pass
                s = str(v)
                vals.append("__MISSING__" if s in ("", "nan", "None", "NaN") else s)
            data[names[j]] = pd.Series(vals, dtype="object")
        else:
            arr = pd.to_numeric(pd.Series(list(col)), errors="coerce").to_numpy(
                dtype=np.float32, na_value=np.nan
            )
            data[names[j]] = pd.Series(np.asarray(arr, dtype=np.float32))
    return pd.DataFrame(data, columns=names)


def _build_pool(df: pd.DataFrame, cat_idx: list[int]) -> Any:
    valid = [int(c) for c in cat_idx if 0 <= int(c) < int(df.shape[1])]
    return Pool(data=df, cat_features=valid if valid else None, thread_count=_thread_count())


def _sigmoid(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _shap_decompose(
    est: Any, pool: Any, n_rows: int, width: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(phi_2d, base_1d, n_classes_internal)`` in float64."""
    raw = est.get_feature_importance(data=pool, type="ShapValues", thread_count=_thread_count())
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:
        # (N, C, F+1) multiclass — caller slices the winning class.
        n_c = int(arr.shape[1])
        return arr, np.zeros((arr.shape[0],), dtype=np.float64), n_c
    if arr.ndim == 1:
        arr = arr.reshape(n_rows, -1)
    if arr.shape[1] == width + 1:
        return arr[:, :width], arr[:, width], 1
    if arr.shape[1] > width:  # defensive: trim trailing extras
        return arr[:, :width], arr[:, -1], 1
    phi = np.zeros((n_rows, width), dtype=np.float64)
    take = min(width, arr.shape[1])
    if take > 0:
        phi[:, :take] = arr[:, :take]
    return phi, np.zeros((n_rows,), dtype=np.float64), 1


def _parse_split_text(text: str, name_to_idx: dict[str, int]) -> dict[str, Any] | None:
    m = _NUM_SPLIT_RE.match(text.strip())
    if m:
        feat = m.group("feat").strip()
        try:
            thr = float(m.group("thr"))
        except (TypeError, ValueError):
            return None
        return {"feature_name": feat, "feature_idx": int(name_to_idx.get(feat, -1)),
                "kind": "numeric", "threshold": round(thr, 5), "category": None}
    m2 = _CAT_SPLIT_RE.match(text.strip())
    if m2:
        feat = m2.group("feat").strip()
        return {"feature_name": feat, "feature_idx": int(name_to_idx.get(feat, -1)),
                "kind": "categorical", "threshold": None, "category": m2.group("val")}
    return None


def _oblivious_topology(
    est: Any, pool: Any, feature_names: list[str], width: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Parse first trees' symmetric level splits; single capped pass."""
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    try:
        n_trees = int(est.tree_count_)
    except (AttributeError, TypeError, ValueError):
        n_trees = 0
    if n_trees < 1:
        return [], [], []
    topo: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    cat_set: set[int] = set()
    try:
        importances = [float(v) for v in list(est.get_feature_importance(pool, type="PredictionValuesChange"))]
    except Exception:
        importances = [0.0 for _ in range(width)]
    for r in range(min(n_trees, MAX_TOPOLOGY_TREES)):
        try:
            texts = est._get_tree_splits(r, pool)
        except Exception:
            continue
        if not isinstance(texts, (list, tuple)):
            continue
        levels: list[dict[str, Any]] = []
        for d, t in enumerate(list(texts)):
            parsed = _parse_split_text(str(t), name_to_idx)
            if parsed is None:
                continue
            f_idx = int(parsed["feature_idx"])
            gain = float(importances[f_idx]) if 0 <= f_idx < len(importances) else 0.0
            entry = {"depth": int(d), **parsed, "gain": round(gain, 4)}
            levels.append(entry)
            splits.append({"round": int(r), "depth": int(d),
                           "feature_idx": f_idx, "gain": round(gain, 4),
                           "kind": parsed["kind"],
                           "threshold": parsed["threshold"], "category": parsed["category"]})
            if parsed["kind"] == "categorical" and f_idx >= 0:
                cat_set.add(f_idx)
        try:
            leaf_values = [float(v) for v in list(est._get_tree_leaf_values(r))]
        except Exception:
            leaf_values = []
        n_leaves = len(leaf_values) if leaf_values else (2 ** len(levels) if levels else 0)
        # Bitmask labels for the HUD: leaf l <-> binary path of depth D.
        bitmasks = [format(int(l), f"0{len(levels)}b") if levels else "0" for l in range(n_leaves)]
        topo.append({"round": int(r), "depth": int(len(levels)), "num_leaves": int(n_leaves),
                     "levels": levels, "leaf_values": [round(float(v), 5) for v in leaf_values[:64]],
                     "leaf_bitmasks": bitmasks[:64]})
    splits.sort(key=lambda s: float(s.get("gain", 0.0)), reverse=True)
    return topo, splits[:MAX_SPLIT_LINES], sorted(cat_set)


class _CatBoostBase(BaseModelAdapter):
    is_classifier: bool = True
    supports_nan_input: bool = True

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        width = max(int(n_features), 0)
        est = _final_cb_estimator(estimator)
        try:
            df = pd.DataFrame(
                np.zeros((min(8, max(1, width)), width), dtype=np.float32),
                columns=_feature_names(est, width),
            )
            pool = _build_pool(df, [c for c in _cat_indices(est) if c < width])
            try:
                scores = list(est.get_feature_importance(pool, type="PredictionValuesChange"))
            finally:
                del pool
            out: list[float | None] = []
            for i in range(width):
                try:
                    f = float(scores[i]) if i < len(scores) else 0.0
                    out.append(f if np.isfinite(f) else 0.0)
                except (TypeError, ValueError):
                    out.append(0.0)
            return out
        except Exception:
            return [None for _ in range(width)]

    # -- inference ------------------------------------------------------
    def _linked_predict(self, predict_obj: Any, est: Any, df: pd.DataFrame,
                        pool: Any, resolved: int = 0) -> np.ndarray:
        if predict_obj is not est and hasattr(predict_obj, "predict"):
            try:
                if self.is_classifier and hasattr(predict_obj, "predict_proba"):
                    try:
                        proba = np.asarray(predict_obj.predict_proba(df))
                    except (ValueError, AttributeError, TypeError, NotImplementedError):
                        proba = None
                    if proba is not None and np.asarray(proba).size:
                        proba = np.asarray(proba)
                        if proba.ndim == 2 and proba.shape[1] > 1:
                            return np.asarray(proba[:, int(min(resolved, proba.shape[1] - 1))])
                        return np.asarray(proba).ravel()
                return np.asarray(predict_obj.predict(df))
            except (ValueError, AttributeError, TypeError):
                pass
        try:
            if self.is_classifier:
                proba = np.asarray(
                    est.predict(pool, prediction_type="Probability",
                                thread_count=_thread_count()))
            else:
                return np.asarray(
                    est.predict(pool, thread_count=_thread_count())).ravel()
        except (AttributeError, TypeError, ValueError):
            if self.is_classifier:
                proba = np.asarray(est.predict_proba(df))
            else:
                return np.asarray(est.predict(df)).ravel()
        if proba.ndim == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, int(min(resolved, proba.shape[1] - 1))])
        return np.asarray(proba).ravel()

    def _full_proba_dist(self, predict_obj: Any, est: Any, df: pd.DataFrame,
                         pool: Any) -> np.ndarray | None:
        if predict_obj is not est and hasattr(predict_obj, "predict_proba"):
            try:
                probas = np.asarray(predict_obj.predict_proba(df), dtype=np.float64)
                if probas.size:
                    if probas.ndim == 1:
                        probas = probas.reshape(-1, 1)
                    return np.where(np.isfinite(probas), probas, 0.0)
            except (ValueError, AttributeError, TypeError, NotImplementedError):
                pass
        try:
            arr = np.asarray(est.predict(pool, prediction_type="Probability",
                                         thread_count=_thread_count()), dtype=np.float64)
        except (AttributeError, TypeError, ValueError):
            try:
                arr = np.asarray(est.predict_proba(df), dtype=np.float64)
            except (AttributeError, TypeError, ValueError):
                return None
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            p = np.clip(np.where(np.isfinite(arr.ravel()), arr.ravel(), 0.0), 0.0, 1.0)
            return np.column_stack([1.0 - p, p])
        return np.where(np.isfinite(arr), arr, 0.0)

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, est = _unwrap_cb_container(model_obj)
        if not _is_catboost_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not a CatBoost model")
        X = np.asarray(X_float32)
        width = int(X.shape[1]) if X.ndim == 2 else 0
        names = _feature_names(est, width)
        cat_idx = [c for c in _cat_indices(est) if c < width]
        df = _to_frame(X, names, cat_idx)
        pool = _build_pool(df, cat_idx)
        try:
            if self.is_classifier:
                probas = self._full_proba_dist(predict_obj, est, df, pool)
                if probas is None or not np.asarray(probas).size:
                    probas = np.zeros((df.shape[0], 2), dtype=np.float64)
                preds = np.argmax(np.asarray(probas), axis=1)
                probabilities = [[float(v) if np.isfinite(v) else 0.0 for v in row]
                                 for row in np.asarray(probas).tolist()]
                confidences = [float(max(row)) if row else 0.0 for row in probabilities]
                py_preds: list[float | int | str] = [int(v) for v in np.asarray(preds).ravel().tolist()]
            else:
                vals = np.asarray(self._linked_predict(predict_obj, est, df, pool),
                                  dtype=np.float64).ravel()
                py_preds = [float(v) if np.isfinite(v) else 0.0 for v in vals.tolist()]
                probabilities, confidences = None, None
        finally:
            del pool
        return PredictionBatchResult(predictions=py_preds, probabilities=probabilities,
                                     confidences=confidences)

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
        predict_obj, est = _unwrap_cb_container(model_obj)
        if not _is_catboost_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not a CatBoost model")
        X = np.asarray(X_float32)
        n_rows = int(X.shape[0]) if X.ndim == 2 else 0
        width = int(X.shape[1]) if X.ndim == 2 else 0
        names = feature_names if len(feature_names) == width else _feature_names(est, width)
        # Native categorical columns follow the fitted model, intersected
        # with the caller's width (positional inference stays aligned).
        fitted_cats = set(_cat_indices(est))
        cat_idx = sorted({c for c in fitted_cats if 0 <= c < width})
        name_to_idx = {str(n): i for i, n in enumerate(names)}
        try:
            extra_cats = [int(name_to_idx[n]) for n in _feature_names(est, width)
                          if str(n) in name_to_idx and int(_feature_names(est, width).index(n)) in fitted_cats]
        except (TypeError, ValueError):
            extra_cats = []
        cat_idx = sorted(set(cat_idx) | {c for c in extra_cats if 0 <= c < width})
        df = _to_frame(X, names, cat_idx)
        pool = _build_pool(df, cat_idx)
        try:
            try:
                n_trees = int(est.tree_count_)
            except (AttributeError, TypeError, ValueError):
                n_trees = 0
            classes: list[Any] = _classes_of(predict_obj if predict_obj is not est
                                             and hasattr(predict_obj, "classes_") else est)
            n_classes = len(classes) if self.is_classifier else 0
            loss = _loss_of(est)

            # 1) Exact C++ SHAP.
            shap_raw = est.get_feature_importance(data=pool, type="ShapValues",
                                                  thread_count=_thread_count())
            shap_arr = np.asarray(shap_raw, dtype=np.float64)
            is_multi = shap_arr.ndim == 3
            if self.is_classifier and (is_multi or (n_classes > 2)):
                n_c = int(shap_arr.shape[1]) if shap_arr.ndim == 3 else int(n_classes or 0)
                if target_class_idx is not None and n_classes:
                    resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
                elif n_classes == 2:
                    resolved = 1
                else:
                    resolved = int(min(max(int(target_idx), 0), max(n_c - 1, 0)))
                grp = int(min(max(resolved, 0), (shap_arr.shape[1] - 1) if shap_arr.ndim == 3 else resolved))
                if shap_arr.ndim == 3:
                    phi = np.asarray(shap_arr[:, grp, :width], dtype=np.float64)
                    bias_col = np.asarray(shap_arr[:, grp, width], dtype=np.float64)
                else:  # defensive: flat fallback keeps binary view
                    phi = np.zeros((n_rows, width), dtype=np.float64)
                    bias_col = np.zeros((n_rows,), dtype=np.float64)
            else:
                if self.is_classifier and n_classes == 2 and target_class_idx is not None:
                    resolved = int(min(max(int(target_class_idx), 0), 1))
                else:
                    resolved = 1 if (self.is_classifier and n_classes == 2) else 0
                grp = 0
                if shap_arr.ndim == 1:
                    shap_arr = shap_arr.reshape(n_rows, -1)
                take_w = min(width, shap_arr.shape[1] - 1) if shap_arr.shape[1] > 0 else 0
                phi = np.asarray(shap_arr[:, :take_w], dtype=np.float64) if take_w > 0 else np.zeros((n_rows, 0))
                if phi.shape[1] < width:
                    pad = np.zeros((n_rows, width - phi.shape[1]), dtype=np.float64)
                    phi = np.concatenate([phi, pad], axis=1)
                bias_col = np.asarray(shap_arr[:, -1], dtype=np.float64) if shap_arr.shape[1] > 0 else np.zeros((n_rows,))
                if self.is_classifier and n_classes == 2 and resolved == 0:
                    phi, bias_col = -phi, -bias_col
            phi = np.where(np.isfinite(phi), phi, 0.0)
            bias_col = np.where(np.isfinite(bias_col), bias_col, 0.0)
            contribs = np.zeros((n_rows, width), dtype=np.float64)
            take = min(width, phi.shape[1])
            if take > 0:
                contribs[:, :take] = phi[:, :take]
            base_raw = float(np.mean(bias_col)) if bias_col.size else 0.0

            # 2) Bitwise leaf indexes (uint16 downcast: depth<=16 -> <65536).
            raw_leaves = np.asarray(est.calc_leaf_indexes(pool, thread_count=_thread_count()))
            if raw_leaves.ndim == 1:
                raw_leaves = raw_leaves.reshape(n_rows, -1)
            raw_leaves = np.asarray(raw_leaves, dtype=np.int64)
            leaf_u16 = np.asarray(np.clip(raw_leaves, 0, 65535), dtype=np.uint16)
            n_leaf_cols = int(leaf_u16.shape[1]) if leaf_u16.size else 0

            # 3) Staged linked trajectory via ntree_end slices.
            staged = np.zeros((n_rows, 0), dtype=np.float64)
            step_iters: list[int] = []
            if n_trees > 0:
                k = min(n_trees, MAX_TRAJECTORY_POINTS)
                step_iters = sorted({int(v) for v in np.unique(
                    np.linspace(1, n_trees, k, dtype=int))})
                cols: list[np.ndarray] = []
                for it in step_iters:
                    try:
                        if self.is_classifier:
                            arr = np.asarray(est.predict(pool, prediction_type="Probability",
                                                         ntree_start=0, ntree_end=int(it),
                                                         thread_count=_thread_count()), dtype=np.float64)
                            if arr.ndim == 2 and arr.shape[1] > 1:
                                col = np.asarray(arr[:, int(min(resolved, arr.shape[1] - 1))], dtype=np.float64)
                            else:
                                col = np.asarray(arr, dtype=np.float64).ravel()
                                if col.size != n_rows:
                                    col = np.asarray(self._linked_predict(predict_obj, est, df, pool, resolved),
                                                     dtype=np.float64).ravel()
                        else:
                            col = np.asarray(est.predict(pool, ntree_start=0, ntree_end=int(it),
                                                         thread_count=_thread_count()), dtype=np.float64).ravel()
                    except (AttributeError, TypeError, ValueError):
                        col = np.asarray(self._linked_predict(predict_obj, est, df, pool, resolved),
                                         dtype=np.float64).ravel()
                    # CatBoost predict(Probability) already returns per-class
                    # columns, so arr[:, 0] is the class-0 probability — no
                    # 1-p flip (unlike single-output LightGBM/XGB boosters).
                    cols.append(np.asarray(col, dtype=np.float64).ravel())
                if cols:
                    staged = np.column_stack(cols)
                    staged = np.where(np.isfinite(staged), staged, 0.0)

            # 4) Native linked finals for parity display.
            linked_resolved = np.asarray(self._linked_predict(predict_obj, est, df, pool, resolved),
                                         dtype=np.float64).ravel()
            native_probas: np.ndarray | None = None
            if self.is_classifier:
                native_probas = self._full_proba_dist(predict_obj, est, df, pool)
                native_labels = np.argmax(native_probas, axis=1) if native_probas is not None else np.clip(
                    linked_resolved, 0.0, 1.0).round().astype(int)
            else:
                native_labels = np.asarray(linked_resolved, dtype=np.float64).ravel()

            try:
                learning_rate = float(getattr(predict_obj, "learning_rate", 0.03) or 0.03)
            except (TypeError, ValueError):
                learning_rate = 0.03
            try:
                depth = int(getattr(est, "get_params", lambda: {})().get("depth", 6))
            except (TypeError, ValueError, AttributeError):
                depth = 6
            k_traj = int(staged.shape[1])
            sample_idx = [(r - 1) for r in step_iters]
            k_top = min(TOP_STAGES, width)
            k_export = min(n_leaf_cols, MAX_EXPORTED_TREES) if leaf_u16.size else 0
            topo, split_lines, topo_cats, = _oblivious_topology(est, pool, names, width)[:3]
            # Merge fitted + topology-discovered categorical columns.
            cat_union = sorted(set(cat_idx) | set(topo_cats))
            try:
                pvc = [round(float(v), 5) for v in list(
                    est.get_feature_importance(pool, type="PredictionValuesChange",
                                               thread_count=_thread_count()))][:width]
            except Exception:
                pvc = [0.0 for _ in range(width)]
            try:
                lfc = [round(float(v), 5) for v in list(
                    est.get_feature_importance(pool, type="LossFunctionChange",
                                               thread_count=_thread_count()))][:width]
            except Exception:
                lfc = [0.0 for _ in range(width)]
            importances = {"prediction_values_change": (pvc + [0.0] * width)[:width],
                           "loss_function_change": (lfc + [0.0] * width)[:width],
                           "gain": (pvc + [0.0] * width)[:width],
                           "split": (lfc + [0.0] * width)[:width]}

            traces: list[InternalFlowRowTrace] = []
            for row_id in row_indices:
                if not (0 <= row_id < n_rows):
                    continue
                row_phi = contribs[row_id]
                order = np.argsort(np.abs(row_phi))[::-1][:k_top] if width else []
                order_sorted = sorted(int(v) for v in np.asarray(order).tolist())
                stages = [{"rank": int(f_pos + 1), "feature_idx": int(f_idx),
                           "feature_name": names[f_idx] if f_idx < len(names) else f"feature_{f_idx}",
                           "shap_value": round(float(row_phi[f_idx]), 5)}
                          for f_pos, f_idx in enumerate(order_sorted)]
                traj = [round(float(staged[row_id, s]), 4) for s in range(k_traj)] if k_traj else []
                row_nan = bool(np.isnan(np.asarray(
                    pd.to_numeric(pd.Series(list(np.asarray(X[row_id]).ravel())), errors="coerce"),
                    dtype=np.float64)).any())
                leaf_proba = ([float(v) for v in native_probas[row_id].tolist()]
                              if native_probas is not None and row_id < native_probas.shape[0] else [])
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
                # Bitwise leaf badges for the HUD (first sampled tree).
                leaf_badges: list[dict[str, Any]] = []
                if leaf_u16.size and row_id < leaf_u16.shape[0]:
                    for t in range(min(k_export, 4)):
                        lv = int(leaf_u16[row_id, t])
                        depth_t = int(topo[t]["depth"]) if t < len(topo) else int(depth)
                        leaf_badges.append({"tree": int(t), "leaf": lv,
                                            "bitmask": format(lv, f"0{max(depth_t, 1)}b")})
                # CTR inspector pill: top contributor is a native categorical.
                top_is_cat = bool(stages and int(stages[0]["feature_idx"]) in set(cat_union))
                extra: dict[str, Any] = {
                    "target_idx": int(resolved),
                    "target_class_idx": int(resolved),
                    "is_classifier": bool(self.is_classifier),
                    "is_regressor": bool(not self.is_classifier),
                    "is_binary": bool(self.is_classifier and n_classes == 2),
                    "algorithm": type(_final_cb_estimator(model_obj)).__name__,
                    "loss_function": loss,
                    "objective": loss,
                    "num_class": int(n_classes) if self.is_classifier else 0,
                    "classes": [str(c) for c in classes],
                    "class_labels": classes,
                    "model_family": "catboost",
                    "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.03,
                    "depth": int(depth),
                    "n_iterations": int(n_trees),
                    "n_boost_rounds": int(n_trees),
                    "n_rounds_total": int(n_trees),
                    "n_trees_total": int(n_trees),
                    "num_leaves": int(topo[0]["num_leaves"]) if topo else 0,
                    "negated_target_zero_view": bool(
                        self.is_classifier and n_classes == 2 and resolved == 0),
                    "has_missing_values": row_nan,
                    "native_categorical_features": [names[c] for c in cat_union if c < len(names)],
                    "categorical_feature_idx": cat_union,
                    "top_feature_is_categorical": top_is_cat,
                    "ctr_inspector": bool(top_is_cat),
                    "base_raw_score": float(base_raw),
                    "final_raw_score": float(final_margin),
                    "sampled_iteration_indices": [int(s) for s in sample_idx],
                    "staged_trajectory": traj,
                    "leaf_indices_sample": ([int(v) for v in leaf_u16[row_id, :k_export].tolist()]
                                            if leaf_u16.size and row_id < leaf_u16.shape[0] else []),
                    "leaf_bitmask_badges": leaf_badges,
                    "active_leaf_depth": int(topo[0]["depth"]) if topo else int(depth),
                    "top_shap_features": stages,
                    "probabilities": leaf_proba,
                    "predicted_class": final_pred,
                    "linked_prediction": (round(float(linked_resolved[row_id]), 5)
                                          if row_id < np.asarray(linked_resolved).size else None),
                    "winning_class_idx": int(resolved) if self.is_classifier else 0,
                    "resolved_class_idx": int(resolved),
                }
                traces.append(InternalFlowRowTrace(
                    row_index=int(row_id), base_value=float(base_raw),
                    final_prediction=final_pred,  # type: ignore[arg-type]
                    feature_contributions={names[i] if i < len(names) else f"feature_{i}":
                                           float(row_phi[i]) for i in range(int(width))},
                    extra_mechanics=extra))
            for tr in traces:
                tr.extra_mechanics["split_lines_sample"] = split_lines
                tr.extra_mechanics["feature_importances"] = importances
                tr.extra_mechanics["tree_topology_rounds"] = [t["round"] for t in topo]
                tr.extra_mechanics["oblivious_tree_topology"] = topo
            return traces
        finally:
            del pool

    # -- surface --------------------------------------------------------
    def _surface_common(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        predict_obj, est = _unwrap_cb_container(model_obj)
        if not _is_catboost_obj(est):
            raise ValueError(f"Estimator '{type(est).__name__}' is not a CatBoost model")
        width = max(int(n_features), 1)
        names = _feature_names(est, width)
        fitted_cats = set(_cat_indices(est))
        cat_idx = sorted({c for c in fitted_cats if 0 <= c < width})
        classes: list[Any] = _classes_of(predict_obj if predict_obj is not est
                                         and hasattr(predict_obj, "classes_") else est)
        n_classes = len(classes) if self.is_classifier else 0
        loss = _loss_of(est)
        if self.is_classifier:
            if n_classes > 2:
                if target_class_idx is not None and n_classes:
                    resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
                else:
                    resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
            else:
                resolved = 1 if not (target_class_idx == 0) else 0
                if target_class_idx is not None:
                    resolved = int(min(max(int(target_class_idx), 0), 1))
                elif target_idx == 0 and n_classes != 2:
                    resolved = 0
        else:
            resolved = 0
        try:
            learning_rate = float(getattr(predict_obj, "learning_rate", 0.03) or 0.03)
        except (TypeError, ValueError):
            learning_rate = 0.03
        try:
            depth = int(getattr(est, "get_params", lambda: {})().get("depth", 6))
        except (TypeError, ValueError, AttributeError):
            depth = 6

        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        x_lin = np.linspace(grid_spec.x_min, grid_spec.x_max, res, dtype=np.float32)
        y_lin = np.linspace(grid_spec.y_min, grid_spec.y_max, res, dtype=np.float32)
        n_points = res * res
        # Median frame: categorical axes hold their mode string; numeric hold medians.
        med_frame: dict[int, Any] = {}
        for feat_idx, med_val in grid_spec.fixed_feature_medians.items():
            if 0 <= feat_idx < width:
                med_frame[feat_idx] = med_val
        ax, ay = grid_spec.axis_x_feature_index, grid_spec.axis_y_feature_index
        ax_is_cat = ax in set(cat_idx)
        ay_is_cat = ay in set(cat_idx)
        # Categorical axes use the fitted category vocabulary when known.
        cat_values: dict[int, list[str]] = {}
        # NOTE: full unique-category mining needs the training frame; the
        # preview path falls back to a stable placeholder ladder so the
        # mesh never collapses on categorical axes (edge #4).
        for c in (ax, ay):
            if c in set(cat_idx):
                cat_values[c] = [f"cat_{i}" for i in range(min(res, 8))]
        if ax_is_cat:
            xs_labels = cat_values[ax]
        else:
            xs_labels = []
        if ay_is_cat:
            ys_labels = cat_values[ay]
        else:
            ys_labels = []
        grid_rows: list[dict[str, Any]] = []
        for r in range(res):
            for c_ in range(res):
                row: dict[str, Any] = {}
                for j in range(width):
                    if j == ax:
                        row[j] = xs_labels[c_ % len(xs_labels)] if ax_is_cat else float(x_lin[c_])
                    elif j == ay:
                        row[j] = ys_labels[r % len(ys_labels)] if ay_is_cat else float(y_lin[r])
                    else:
                        row[j] = med_frame.get(j, 0.0)
                grid_rows.append(row)
        grid_df = pd.DataFrame(
            [{names[j] if j < len(names) else f"feature_{j}": grid_rows[i][j] for j in range(width)}
             for i in range(n_points)], columns=names)
        # Coerce numerics (categoricals stay str).
        for j in range(width):
            if j not in set(cat_idx):
                grid_df[names[j]] = pd.to_numeric(grid_df[names[j]], errors="coerce").astype(np.float32)
            else:
                grid_df[names[j]] = grid_df[names[j]].astype(str)
        grid_pool = _build_pool(grid_df, cat_idx)
        try:
            try:
                if self.is_classifier:
                    arr = np.asarray(est.predict(grid_pool, prediction_type="Probability",
                                                 thread_count=_thread_count()), dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] > 1:
                        z_flat = np.asarray(arr[:, int(min(resolved, arr.shape[1] - 1))],
                                            dtype=np.float64).ravel()
                    else:
                        z_flat = np.asarray(arr, dtype=np.float64).ravel()
                else:
                    z_flat = np.asarray(est.predict(grid_pool, thread_count=_thread_count()),
                                        dtype=np.float64).ravel()
            except (ValueError, AttributeError, TypeError):
                z_flat = np.zeros((n_points,), dtype=np.float64)
            if z_flat.size != n_points:
                z_flat = np.zeros((n_points,), dtype=np.float64)
            z_surface = np.where(np.isfinite(z_flat), z_flat, 0.0).reshape(res, res)
            topo, split_lines, topo_cats = _oblivious_topology(est, grid_pool, names, width)
            cat_union = sorted(set(cat_idx) | set(topo_cats))
            try:
                pvc = [round(float(v), 5) for v in list(
                    est.get_feature_importance(grid_pool, type="PredictionValuesChange",
                                               thread_count=_thread_count()))][:width]
            except Exception:
                pvc = [0.0 for _ in range(width)]
            importances = {"prediction_values_change": (pvc + [0.0] * width)[:width],
                           "gain": (pvc + [0.0] * width)[:width]}
        finally:
            del grid_pool
            del grid_df
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
        try:
            n_trees = int(est.tree_count_)
        except (AttributeError, TypeError, ValueError):
            n_trees = 0
        return SurfaceMesh3DData(
            visual_type=self.visual_3d_type,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in x_lin] if not ax_is_cat else [float(i) for i in range(len(x_lin))],
            y_grid=[float(v) for v in y_lin] if not ay_is_cat else [float(i) for i in range(len(y_lin))],
            z_values=[float(v) for v in np.asarray(z_surface, dtype=np.float64).ravel()],
            extra_overlays={
                "target_idx": int(resolved),
                "resolved_target_class_idx": int(resolved),
                "classes": [str(c) for c in classes],
                "algorithm": type(_final_cb_estimator(model_obj)).__name__,
                "loss_function": loss,
                "objective": loss,
                "num_class": int(n_classes) if self.is_classifier else 0,
                "model_family": "catboost",
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.03,
                "depth": int(depth),
                "n_iterations": int(n_trees),
                "n_boost_rounds": int(n_trees),
                "n_rounds_total": int(n_trees),
                "n_trees_total": int(n_trees),
                "feature_importances": importances,
                "tree_topology_sample": topo,
                "tree_topology_rounds": [t["round"] for t in topo],
                "oblivious_tree_topology": topo,
                "split_lines_sample": split_lines,
                "categorical_feature_idx": cat_union,
                "categorical_ticks_x": xs_labels,
                "categorical_ticks_y": ys_labels,
                "axis_x_is_categorical": bool(ax_is_cat),
                "axis_y_is_categorical": bool(ay_is_cat),
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("catboost")
class CatBoostClassifierAdapter(_CatBoostBase):
    display_name = "CatBoost (Symmetric Oblivious Exact SHAP)"
    family_name = "Gradient Boosting"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = True
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        if not _CB_AVAILABLE:
            return False
        final_est = _final_cb_estimator(estimator)
        if isinstance(final_est, _CLASSIFIER_TYPES):
            return True
        if type(final_est).__name__ == "CatBoostClassifier":
            return True
        if _is_catboost_obj(final_est):
            return not _is_regression_estimator(final_est)
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
        return self._trace_common(model_obj, X_float32, row_indices, feature_names,
                                  target_idx, target_class_idx)

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)


@ModelRegistry.register("catboost_regressor")
class CatBoostRegressorAdapter(_CatBoostBase):
    display_name = "CatBoost Regressor (Symmetric Oblivious Exact SHAP)"
    family_name = "Gradient Boosting"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = False
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        if not _CB_AVAILABLE:
            return False
        final_est = _final_cb_estimator(estimator)
        if isinstance(final_est, _REGRESSOR_TYPES):
            return True
        if type(final_est).__name__ == "CatBoostRegressor":
            return True
        if _is_catboost_obj(final_est):
            return _is_regression_estimator(final_est)
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
        return self._trace_common(model_obj, X_float32, row_indices, feature_names,
                                  target_idx, target_class_idx)

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        return self._surface_common(model_obj, grid_spec, n_features, target_class_idx, target_idx)


def load_catboost_from_path(file_path: Any) -> Any:
    """Load a ``.cbm`` artifact with 2-stage Classifier/Regressor probing (§2.2)."""
    from pathlib import Path  # noqa: PLC0415

    path = Path(file_path)
    if not _CB_AVAILABLE:
        raise ValueError("catboost is not installed")
    probe = CatBoostClassifier()
    probe.load_model(str(path), format="cbm")
    loss = str(probe.get_all_params().get("loss_function", "") or "")
    base = loss.split(":")[0].strip()
    if base in _REGRESSION_LOSSES:
        model = CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        return model
    return probe

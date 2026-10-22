# backend/app/domains/model_engine/adapters/boosting/xgb_adapter.py
"""Phase 33 — XGBoost staged booster adapters (native C++ TreeSHAP leaf flow).

Covers ``XGBClassifier`` / ``XGBRegressor`` (including ``XGBRF*`` forest
mode), native ``xgb.Booster`` artifacts (``.pkl`` / ``.json`` / ``.ubj`` /
``.model``), and ``Pipeline`` / ``GridSearchCV``-style wrappers — inference
runs through the full container while staged tracing unwraps to the
fitted booster.

Blueprint mapping (phase-33-model-xgboost.md):

- One shared ``DMatrix`` per call feeds ``pred_contribs`` (Exact C++
  TreeSHAP), ``pred_leaf`` and the ``iteration_range`` staged trajectory
  (blueprint §6.1). ``strict_shape=True`` normalizes every output so
  binary/regression is ``(n, 1, f+1)`` / ``(n, rounds, 1, parallel)`` and
  multiclass gains an explicit group axis (edge #1).
- Exact additivity ``Σφ + bias == margin`` holds against
  ``predict(output_margin=True)`` for every objective; the staged curve
  and final prediction stay in linked space (``predict`` without margin).
- ``base_score`` is never parsed from config: ``base_value`` is the
  finite-guarded mean of the per-row SHAP bias column (blueprint §2.2).
- Native boosters without embedded names predict positionally with
  ``feature_names=None``; boosters trained with names reuse exactly those
  names positionally (edge #2 — mismatch raises inside XGBoost).
- Effective rounds honor ``best_iteration`` (wrapper attr or booster attr,
  edge #3); ``gblinear`` skips leaves/dump (edge #4); categorical splits
  surface as ``split_condition`` lists (edge #5); DART trajectories are
  capped to 20 points (edge #6).
- Topology comes from a single ``get_dump(json, with_stats=True)`` pass
  (blueprint §6.3): first ``MAX_TOPOLOGY_TREES`` trees ship as flat node
  lists for the 3D branch-glow explorer, plus top-gain split lines.
"""

from __future__ import annotations

import json
import re
import warnings
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

try:  # Phase 33: xgboost is optional at import time (registry stays importable).
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

    _XGB_AVAILABLE = True
    _CLASSIFIER_TYPES = (XGBClassifier, XGBRFClassifier)
    _REGRESSOR_TYPES = (XGBRegressor, XGBRFRegressor)
except ImportError:  # pragma: no cover - exercised only without xgboost
    xgb = None  # type: ignore[assignment]
    _XGB_AVAILABLE = False
    _CLASSIFIER_TYPES = ()
    _REGRESSOR_TYPES = ()

MAX_TRAJECTORY_POINTS = 50
MAX_DART_TRAJECTORY_POINTS = 20
MAX_EXPORTED_ROUNDS = 100
MAX_TOPOLOGY_TREES = 10
MAX_SPLIT_LINES = 30
MAX_SURFACE_RESOLUTION = 100
TOP_STAGES = 8
NTHREAD_CAP = 2

_FNAME_RE = re.compile(r"^f(\d+)$")


def _unwrap_xgb_container(model_obj: Any) -> tuple[Any, Any]:
    """Return ``(predict_obj, booster_or_estimator)``.

    Follows ``best_estimator_`` chains, ``Pipeline`` ``steps`` and
    ``TransformedTargetRegressor`` ``regressor_`` (Phase 31/32 analogue).
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


def _final_xgb_estimator(model_obj: Any) -> Any:
    return _unwrap_xgb_container(model_obj)[1]


def _is_native_booster(obj: Any) -> bool:
    return _XGB_AVAILABLE and isinstance(obj, xgb.Booster)


def _as_booster(est: Any) -> Any | None:
    """Return the native booster for wrappers, boosters, or ``None``."""
    if _is_native_booster(est):
        return est
    get_booster = getattr(est, "get_booster", None)
    if callable(get_booster):
        try:
            booster = get_booster()
            return booster if _is_native_booster(booster) else None
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _booster_config(booster: Any) -> dict[str, Any]:
    """Parse objective / booster type / class count from ``save_config``."""
    out: dict[str, Any] = {"objective": "", "booster_type": "gbtree", "num_class": 0}
    try:
        cfg = json.loads(booster.save_config())
        learner = cfg.get("learner", {})
        out["objective"] = str(learner.get("objective", {}).get("name", ""))
        out["booster_type"] = str(learner.get("gradient_booster", {}).get("name", "gbtree"))
        try:
            out["num_class"] = int(learner.get("learner_model_param", {}).get("num_class", 0) or 0)
        except (TypeError, ValueError):
            out["num_class"] = 0
    except Exception:
        pass
    return out


def _objective_of(predict_obj: Any, booster: Any | None) -> str:
    if booster is not None:
        objective = str(_booster_config(booster).get("objective", ""))
        if objective:
            return objective
    for attr in ("objective",):
        try:
            raw = getattr(predict_obj, attr, None)
        except (AttributeError, TypeError):
            raw = None
        if isinstance(raw, str) and raw:
            return raw
        if callable(raw):
            return "custom"
    return ""


def _effective_rounds(predict_obj: Any, booster: Any) -> int:
    """Boosted rounds honoring ``best_iteration`` (edge #3)."""
    try:
        total = int(booster.num_boosted_rounds())
    except (AttributeError, TypeError, ValueError):
        total = 0
    best: Any = getattr(predict_obj, "best_iteration", None)
    if best is None:
        try:
            best = booster.attr("best_iteration")
        except (AttributeError, TypeError, ValueError):
            best = None
    try:
        b = int(best) if best is not None else None
    except (TypeError, ValueError):
        b = None
    if b is not None and total > 0 and 0 <= b < total:
        return b + 1
    return max(total, 0)


def _cap_nthread(booster: Any) -> None:
    """Pin C++ OpenMP threads so workers never thrash the ComputePool (edge §6.2)."""
    try:
        booster.set_param({"nthread": NTHREAD_CAP})
    except (AttributeError, TypeError, ValueError):
        pass


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


def _native_classes(objective: str, num_class: int) -> list[Any] | None:
    if objective.startswith("binary:"):
        return [0, 1]
    if objective.startswith("multi:") and num_class > 1:
        return list(range(num_class))
    return None


def _is_ttr(predict_obj: Any, booster: Any) -> bool:
    est = _final_xgb_estimator(predict_obj)
    return predict_obj is not est and hasattr(predict_obj, "transformer_")


def _inverse_ttr_vals(predict_obj: Any, values: np.ndarray) -> np.ndarray:
    """Map staged booster outputs back through a TTR transformer (best-effort)."""
    try:
        transformer = getattr(predict_obj, "transformer_", None)
        if transformer is None or values.size == 0:
            return values
        flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        back = np.asarray(transformer.inverse_transform(flat), dtype=np.float64).ravel()
        return back.reshape(values.shape)
    except (ValueError, AttributeError, TypeError):
        return values


def _ttr_needs_remap(predict_obj: Any) -> bool:
    """True when a TTR wrapper applies an effective (non-identity) transform.

    Self-calibrating probe: identity transformers (the TTR default
    ``FunctionTransformer``) keep the exact C++ SHAP path; real
    transformers fall back to inverse-staged attribution.
    """
    try:
        transformer = getattr(predict_obj, "transformer_", None)
    except (AttributeError, TypeError):
        return False
    if transformer is None:
        return False
    try:
        probe = np.array([[0.5], [2.0]], dtype=np.float64)
        back = np.asarray(transformer.inverse_transform(probe), dtype=np.float64)
        return not np.allclose(back, probe, rtol=1e-9, atol=1e-12)
    except (ValueError, AttributeError, TypeError):
        return False


def _make_dmatrix(booster: Any, X_f32: np.ndarray) -> Any:
    """Build one shared ``DMatrix`` (blueprint §6.1).

    Boosters trained with embedded names require those exact names;
    columns are positional so the embedded list is reused verbatim
    (edge #2). Nameless boosters predict without names.
    """
    embedded = list(getattr(booster, "feature_names", None) or [])
    fn = embedded if embedded and len(embedded) == X_f32.shape[1] else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        return xgb.DMatrix(_as_dense_float32(X_f32), feature_names=fn)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _feature_index(name: Any, booster_names: list[str] | None, width: int) -> int:
    """Map a dump ``split`` token (name or ``f{j}``) to a column index."""
    if isinstance(name, (int, np.integer)):
        return int(name) if 0 <= int(name) < width else -1
    token = str(name)
    if booster_names and token in booster_names:
        return booster_names.index(token)
    m = _FNAME_RE.match(token)
    if m:
        try:
            j = int(m.group(1))
            return j if 0 <= j < width else -1
        except ValueError:
            return -1
    return -1


def _flatten_dump_tree(tree: Any, booster_names: list[str] | None, width: int) -> list[dict[str, Any]]:
    """Flatten one parsed dump tree into node dicts (edge #5 aware)."""
    flat: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "leaf" in node:
            try:
                flat.append(
                    {
                        "id": int(node.get("nodeid", -1)),
                        "leaf": round(float(node["leaf"]), 5),
                        "cover": round(float(node.get("cover", 0.0)), 4),
                    }
                )
            except (TypeError, ValueError):
                pass
            return
        cond = node.get("split_condition")
        if isinstance(cond, list):
            threshold: Any = None
            try:
                categories = sorted({int(v) for v in cond})
            except (TypeError, ValueError):
                categories = []
        else:
            try:
                threshold = round(float(cond), 5) if cond is not None else None
            except (TypeError, ValueError):
                threshold = None
            categories = []
        try:
            flat.append(
                {
                    "id": int(node.get("nodeid", -1)),
                    "split_feature_idx": _feature_index(node.get("split"), booster_names, width),
                    "split_feature": str(node.get("split")),
                    "threshold": threshold,
                    "categories": categories,
                    "yes": int(node.get("yes", -1)),
                    "no": int(node.get("no", -1)),
                    "missing": int(node.get("missing", -1)),
                    "gain": round(float(node.get("gain", 0.0)), 4),
                    "cover": round(float(node.get("cover", 0.0)), 4),
                }
            )
        except (TypeError, ValueError):
            pass
        for child in node.get("children", []) or []:
            visit(child)

    visit(tree)
    return flat


class _XGBBase(BaseModelAdapter):
    is_classifier: bool = True
    supports_nan_input: bool = True

    def _resolve_target(
        self, n_groups: int, target_idx: int, target_class_idx: int | None, classes: list[Any]
    ) -> tuple[int, int, bool]:
        """Return ``(resolved_class, group_col, negate_binary)``.

        Binary XGBoost grows a single group; an explicit
        ``target_class_idx == 0`` negates the margin so the waterfall
        still sums exactly (Phase 31/32 analogue).
        """
        n_classes = len(classes)
        if self.is_classifier and n_groups <= 1:
            if target_class_idx is not None:
                negate = bool(int(target_class_idx) == 0 and n_classes == 2)
            else:
                negate = False
            return (1 if n_classes == 2 else 0), 0, negate
        if self.is_classifier:
            if target_class_idx is not None and n_classes:
                resolved = int(min(max(int(target_class_idx), 0), n_classes - 1))
            elif n_classes == 2:
                resolved = 1
            else:
                resolved = int(min(max(int(target_idx), 0), max(n_classes - 1, 0)))
            return resolved, int(min(max(resolved, 0), max(n_groups - 1, 0))), False
        return 0, 0, False

    def _booster_parts(self, model_obj: Any) -> tuple[Any, Any, dict[str, Any]]:
        predict_obj, est = _unwrap_xgb_container(model_obj)
        booster = _as_booster(est)
        if booster is None:
            raise ValueError(f"Estimator '{type(est).__name__}' exposes no xgb.Booster")
        _cap_nthread(booster)
        return predict_obj, booster, _booster_config(booster)

    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        booster = _as_booster(_final_xgb_estimator(estimator))
        width = max(int(n_features), 0)
        if booster is None:
            return [None for _ in range(width)]
        return [float(v) for v in self._metric_importances(booster, width, "gain")]

    def _metric_importances(self, booster: Any, width: int, metric: str) -> list[float]:
        """Index-aligned importances from ``get_score`` (gain/weight/cover/...)."""
        out = [0.0 for _ in range(max(width, 0))]
        try:
            scores = dict(booster.get_score(importance_type=metric) or {})
        except (AttributeError, TypeError, ValueError):
            return out
        names = list(getattr(booster, "feature_names", None) or [])
        for key, val in scores.items():
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(f):
                continue
            idx = _feature_index(key, names if names else None, width)
            if 0 <= idx < len(out):
                out[idx] = f
        return out

    # -- inference ------------------------------------------------------
    def _linked_predict(
        self,
        predict_obj: Any,
        booster: Any,
        X_in: np.ndarray,
        objective: str,
        eff_rounds: int,
        resolved: int = 0,
    ) -> np.ndarray:
        """Linked-space predictions for ``resolved`` honoring ``best_iteration`` (edge #3).

        Classifier wrappers resolve through ``predict_proba`` (``predict``
        returns labels, not link values); native boosters use
        ``iteration_range``-capped ``predict``.
        """
        if predict_obj is not booster and hasattr(predict_obj, "predict"):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    if self.is_classifier and hasattr(predict_obj, "predict_proba"):
                        try:
                            proba = np.asarray(
                                predict_obj.predict_proba(_as_dense_float32(X_in))
                            )
                        except (ValueError, AttributeError, TypeError, NotImplementedError):
                            proba = None
                        if proba is not None and proba.size:
                            if proba.ndim == 2 and proba.shape[1] > 1:
                                return np.asarray(proba[:, int(min(resolved, proba.shape[1] - 1))])
                            return np.asarray(proba).ravel()
                    return np.asarray(predict_obj.predict(_as_dense_float32(X_in)))
            except (ValueError, AttributeError, TypeError):
                pass
        dmat = _make_dmatrix(booster, _as_dense_float32(X_in))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            raw: Any = None
            if eff_rounds > 0:
                try:
                    raw = booster.predict(dmat, iteration_range=(0, eff_rounds))
                except (TypeError, ValueError):
                    raw = None
            if raw is None:
                raw = booster.predict(dmat)
            arr = np.asarray(raw)
            if arr.ndim == 2 and arr.shape[1] > 1:
                return np.asarray(arr[:, int(min(resolved, arr.shape[1] - 1))])
            return np.asarray(arr).ravel()

    def _full_proba_dist(
        self,
        predict_obj: Any,
        booster: Any,
        X_arr: np.ndarray,
        objective: str,
        eff_rounds: int,
    ) -> np.ndarray | None:
        """Full ``(n, C)`` class distribution in linked (probability) space.

        Wrapper ``predict_proba`` first; native boosters map their raw
        ``predict`` through the objective link (``binary:hinge`` and
        ``multi:softmax`` become one-hot rows).
        """
        if self.is_classifier and predict_obj is not booster and hasattr(
            predict_obj, "predict_proba"
        ):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    probas = np.asarray(
                        predict_obj.predict_proba(_as_dense_float32(X_arr)), dtype=np.float64
                    )
            except (ValueError, AttributeError, TypeError, NotImplementedError):
                probas = None
            if probas is not None and probas.size:
                if probas.ndim == 1:
                    probas = probas.reshape(-1, 1)
                return np.where(np.isfinite(probas), probas, 0.0)
        dmat = _make_dmatrix(booster, _as_dense_float32(X_arr))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            raw: Any = None
            if eff_rounds > 0:
                try:
                    raw = booster.predict(dmat, iteration_range=(0, eff_rounds))
                except (TypeError, ValueError):
                    raw = None
            if raw is None:
                raw = booster.predict(dmat)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return np.where(np.isfinite(arr), arr, 0.0)
        flat = np.asarray(arr).ravel()
        if objective == "binary:hinge" or objective.startswith("multi:"):
            labels = np.clip(flat.astype(int), 0, None)
            n_cls = int(max(labels.max(initial=0), 1) + 1) if labels.size else 2
            if objective.startswith("binary:"):
                n_cls = 2
            one_hot = np.zeros((flat.size, n_cls), dtype=np.float64)
            one_hot[np.arange(flat.size), np.clip(labels, 0, n_cls - 1)] = 1.0
            return one_hot
        if objective == "binary:logitraw":
            flat = _sigmoid(flat)
        p = np.clip(np.where(np.isfinite(flat), flat, 0.0), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        predict_obj, booster, cfg = self._booster_parts(model_obj)
        objective = _objective_of(predict_obj, booster)
        eff = _effective_rounds(predict_obj, booster)
        X_in = _as_dense_float32(X_float32)
        probabilities: list[list[float]] | None = None
        confidences: list[float] | None = None
        if self.is_classifier:
            probas = self._full_proba_dist(predict_obj, booster, X_in, objective, eff)
            if probas is None or not probas.size:
                probas = np.zeros((X_in.shape[0], 2), dtype=np.float64)
            preds = np.argmax(probas, axis=1)
            probabilities = [
                [float(v) if np.isfinite(v) else 0.0 for v in row] for row in probas.tolist()
            ]
            confidences = [float(max(row)) if row else 0.0 for row in probabilities]
            py_preds: list[float | int | str] = [int(v) for v in np.asarray(preds).ravel().tolist()]
        else:
            vals = np.asarray(
                self._linked_predict(predict_obj, booster, X_in, objective, eff), dtype=np.float64
            ).ravel()
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
        predict_obj, booster, cfg = self._booster_parts(model_obj)
        booster_type = str(cfg.get("booster_type", "gbtree"))
        objective = _objective_of(predict_obj, booster)
        num_class = int(cfg.get("num_class", 0) or 0)
        if _is_native_booster(predict_obj) or predict_obj is booster:
            classes = _native_classes(objective, num_class) or [0, 1]
        else:
            raw_classes = _classes_of(predict_obj)
            classes = _native_classes(objective, num_class) or raw_classes or [0, 1]
            if len(raw_classes) == len(classes):
                classes = raw_classes
        n_classes = len(classes)

        X_native = _as_dense_float32(X_float32)
        Xt = _transform_for_estimators(predict_obj, X_native)
        n_features = int(X_native.shape[1]) if X_native.ndim == 2 else 0
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        width = max(n_features, 0)

        eff_rounds = _effective_rounds(predict_obj, booster)
        dmat = _make_dmatrix(booster, Xt)

        # 1) Exact C++ TreeSHAP (edge: gblinear has no trees but has SHAP).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            try:
                raw_contribs = np.asarray(
                    booster.predict(dmat, pred_contribs=True, strict_shape=True)
                )
            except TypeError:
                raw_contribs = np.asarray(booster.predict(dmat, pred_contribs=True))
        if raw_contribs.ndim == 2:
            # Pre-strict_shape fallback: (n, f+1).
            n_groups = 1
            contribs_3d = raw_contribs.reshape(raw_contribs.shape[0], 1, raw_contribs.shape[1])
        elif raw_contribs.ndim >= 3:
            n_groups = int(raw_contribs.shape[1])
            contribs_3d = raw_contribs.reshape(
                raw_contribs.shape[0], n_groups, int(raw_contribs.shape[2])
            )
        else:  # pragma: no cover - defensive
            raise ValueError("Unexpected pred_contribs shape")
        f_dim = int(contribs_3d.shape[2]) - 1

        resolved, grp, negate = self._resolve_target(
            n_groups, target_idx, target_class_idx, classes
        )
        if self.is_classifier and n_groups <= 1:
            grp = 0
        grp = int(min(max(grp, 0), n_groups - 1))
        phi = np.asarray(contribs_3d[:, grp, : max(f_dim, 0)], dtype=np.float64)
        bias_col = np.asarray(contribs_3d[:, grp, -1], dtype=np.float64)
        if negate:
            phi, bias_col = -phi, -bias_col
        phi = np.where(np.isfinite(phi), phi, 0.0)
        bias_col = np.where(np.isfinite(bias_col), bias_col, 0.0)
        # Positional pad/truncate to the traced width.
        contribs = np.zeros((X_native.shape[0], width), dtype=np.float64)
        take = min(width, phi.shape[1])
        if take > 0:
            contribs[:, :take] = phi[:, :take]
        base_raw = float(np.mean(bias_col)) if bias_col.size else 0.0
        ttr_remap = bool(not self.is_classifier and _ttr_needs_remap(predict_obj))

        # 2) Leaf indices (gbtree/dart only — edge #4).
        leaf_matrix: np.ndarray | None = None
        n_parallel = 1
        if booster_type in ("gbtree", "dart") and eff_rounds > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                try:
                    raw_leaves = np.asarray(booster.predict(dmat, pred_leaf=True, strict_shape=True))
                except TypeError:
                    try:
                        raw_leaves = np.asarray(booster.predict(dmat, pred_leaf=True))
                    except Exception:
                        raw_leaves = np.zeros((0,))
                except Exception:
                    raw_leaves = np.zeros((0,))
            if raw_leaves.size and raw_leaves.ndim == 4:
                n_parallel = int(raw_leaves.shape[3])
                g = int(min(grp, raw_leaves.shape[2] - 1))
                leaf_matrix = np.asarray(raw_leaves[:, :eff_rounds, g, 0], dtype=np.int64)
            elif raw_leaves.size and raw_leaves.ndim == 2:
                leaf_matrix = np.asarray(raw_leaves[:, :eff_rounds], dtype=np.int64)

        # 3) Staged linked trajectory via iteration_range.
        cap = (
            MAX_DART_TRAJECTORY_POINTS if booster_type == "dart" else MAX_TRAJECTORY_POINTS
        )  # edge #6
        staged = np.zeros((X_native.shape[0], 0), dtype=np.float64)
        step_rounds: list[int] = []
        if eff_rounds > 0:
            k = min(eff_rounds, cap)
            step_rounds = sorted({int(v) for v in np.unique(np.linspace(1, eff_rounds, k, dtype=int))})
            cols: list[np.ndarray] = []
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                for r in step_rounds:
                    try:
                        pred_r = np.asarray(booster.predict(dmat, iteration_range=(0, int(r))))
                    except (TypeError, ValueError):
                        pred_r = self._linked_predict(predict_obj, booster, Xt, objective, eff_rounds)
                    if pred_r.ndim == 2 and pred_r.shape[1] > 1:
                        c_idx = int(min(resolved, pred_r.shape[1] - 1))
                        col = np.asarray(pred_r[:, c_idx], dtype=np.float64)
                    else:
                        col = np.asarray(pred_r, dtype=np.float64).ravel()
                    if negate and col.size == X_native.shape[0]:
                        col = 1.0 - col
                    cols.append(col)
            if cols:
                staged = np.column_stack(cols)
                staged = np.where(np.isfinite(staged), staged, 0.0)
        if ttr_remap and staged.size:
            # Nonlinear transformed-target space: re-derive base/deltas from
            # the inverse-transformed staged curve and attribute uniformly
            # so base + ΣΦ == predict() holds in original units.
            staged = _inverse_ttr_vals(predict_obj, staged)
            with np.errstate(invalid="ignore"):
                base_raw = float(np.mean(staged[:, 0]))
            base_raw = base_raw if np.isfinite(base_raw) else 0.0
            prev = np.full((staged.shape[0], 1), base_raw, dtype=np.float64)
            remapped = np.diff(np.hstack([prev, staged]), axis=1)
            contribs[:, :] = 0.0
            if width:
                for m in range(remapped.shape[1]):
                    contribs[:, :] += (remapped[:, m].reshape(-1, 1) / width) * np.ones((1, width))

        # 4) Native linked finals for parity display.
        linked_resolved = np.asarray(
            self._linked_predict(predict_obj, booster, Xt, objective, eff_rounds, resolved),
            dtype=np.float64,
        ).ravel()
        native_probas: np.ndarray | None = None
        native_labels: np.ndarray
        if self.is_classifier:
            native_probas = self._full_proba_dist(predict_obj, booster, Xt, objective, eff_rounds)
            if native_probas is not None and native_probas.size:
                native_labels = np.argmax(native_probas, axis=1)
            else:
                native_probas = None
                native_labels = (np.clip(linked_resolved, 0.0, 1.0) >= 0.5).astype(int)
        else:
            native_labels = np.asarray(linked_resolved, dtype=np.float64).ravel()

        try:
            learning_rate = float(getattr(predict_obj, "learning_rate", 0.3))
        except (TypeError, ValueError):
            learning_rate = 0.3
        k_traj = staged.shape[1]
        sample_idx = [(r - 1) for r in step_rounds]
        k_top = min(TOP_STAGES, eff_rounds)
        k_export = min(eff_rounds, MAX_EXPORTED_ROUNDS)

        try:
            booster_names = list(getattr(booster, "feature_names", None) or [])
        except TypeError:
            booster_names = []
        topo, split_lines, cat_idx = self._topology_parts(booster, booster_type, width, booster_names)
        importances = {
            m: [round(float(v), 5) for v in self._metric_importances(booster, width, m)]
            for m in ("gain", "weight", "cover", "total_gain", "total_cover")
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
            extra: dict[str, Any] = {
                "target_idx": int(resolved),
                "target_class_idx": int(resolved),
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "algorithm": type(_final_xgb_estimator(model_obj)).__name__,
                "objective": objective,
                "booster_type": booster_type,
                "classes": [str(c) for c in classes],
                "class_labels": classes,
                "model_family": "xgboost",
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.3,
                "n_boost_rounds": int(eff_rounds),
                "n_rounds_total": int(eff_rounds),
                "n_parallel_trees": int(n_parallel),
                "negated_target_zero_view": bool(negate),
                "has_missing_values": row_nan,
                "base_raw_score": float(base_raw),
                "final_raw_score": float(final_margin),
                "sampled_round_indices": [int(s) for s in sample_idx],
                "staged_trajectory": traj,
                "leaf_indices_sample": (
                    [int(v) for v in leaf_matrix[row_id, :k_export].tolist()]
                    if leaf_matrix is not None and row_id < leaf_matrix.shape[0]
                    else []
                ),
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
        # Attach topology once per call payload via the first trace is wasteful;
        # surface carries it — but keep split/gain context on every trace.
        for tr in traces:
            tr.extra_mechanics["split_lines_sample"] = split_lines
            tr.extra_mechanics["categorical_feature_idx"] = cat_idx
            tr.extra_mechanics["feature_importances"] = importances
            tr.extra_mechanics["tree_topology_rounds"] = [t["round"] for t in topo]
        return traces

    def _topology_parts(
        self, booster: Any, booster_type: str, width: int, booster_names: list[str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
        """Parse ``get_dump`` once: sampled topologies + top-gain splits."""
        if booster_type == "gblinear":
            return [], [], []
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                dumps = booster.get_dump(dump_format="json", with_stats=True)
        except (AttributeError, TypeError, ValueError):
            return [], [], []
        names = booster_names if booster_names else None
        topo: list[dict[str, Any]] = []
        splits: list[dict[str, Any]] = []
        cat_features: set[int] = set()
        for r, dump_str in enumerate(dumps[:MAX_EXPORTED_ROUNDS]):
            try:
                tree = json.loads(dump_str)
            except (TypeError, ValueError):
                continue
            flat = _flatten_dump_tree(tree, names, width)
            if r < MAX_TOPOLOGY_TREES:
                topo.append({"round": int(r), "nodes": flat})
            for node in flat:
                if "leaf" in node:
                    continue
                if node.get("categories"):
                    for f_idx in [node.get("split_feature_idx", -1)]:
                        if isinstance(f_idx, int) and f_idx >= 0:
                            cat_features.add(f_idx)
                splits.append(
                    {
                        "round": int(r),
                        "feature_idx": node.get("split_feature_idx", -1),
                        "gain": node.get("gain", 0.0),
                        "cover": node.get("cover", 0.0),
                    }
                )
        splits.sort(key=lambda s: float(s.get("gain", 0.0)), reverse=True)
        return topo, splits[:MAX_SPLIT_LINES], sorted(cat_features)

    # -- surface --------------------------------------------------------
    def _surface_common(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        predict_obj, booster, cfg = self._booster_parts(model_obj)
        booster_type = str(cfg.get("booster_type", "gbtree"))
        objective = _objective_of(predict_obj, booster)
        num_class = int(cfg.get("num_class", 0) or 0)
        if _is_native_booster(predict_obj) or predict_obj is booster:
            classes = _native_classes(objective, num_class) or [0, 1]
        else:
            raw_classes = _classes_of(predict_obj)
            classes = _native_classes(objective, num_class) or raw_classes or [0, 1]
            if len(raw_classes) == len(classes):
                classes = raw_classes
        n_classes = len(classes)
        # Group probing via a 1-row contribs call.
        probe = np.zeros((1, max(int(n_features), 1)), dtype=np.float32)
        try:
            probe_c = np.asarray(
                booster.predict(_make_dmatrix(booster, probe), pred_contribs=True, strict_shape=True)
            )
            n_groups = int(probe_c.shape[1]) if probe_c.ndim >= 3 else 1
        except Exception:
            n_groups = num_class if num_class > 1 else 1
        resolved, grp, negate = self._resolve_target(n_groups, target_idx, target_class_idx, classes)
        if self.is_classifier and n_groups <= 1:
            grp = 0
        grp = int(min(max(grp, 0), max(n_groups - 1, 0)))
        try:
            learning_rate = float(getattr(predict_obj, "learning_rate", 0.3))
        except (TypeError, ValueError):
            learning_rate = 0.3
        eff_rounds = _effective_rounds(predict_obj, booster)

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
                linked = np.asarray(
                    self._linked_predict(predict_obj, booster, grid, objective, eff_rounds, resolved)
                )
                if linked.ndim == 2 and linked.shape[1] > 1:
                    c_idx = int(min(resolved, linked.shape[1] - 1))
                    z_surface = np.asarray(linked[:, c_idx], dtype=np.float64).reshape(res, res)
                else:
                    flat = np.asarray(linked, dtype=np.float64).ravel()
                    if objective == "binary:logitraw" and self.is_classifier:
                        flat = _sigmoid(flat)
                    z_surface = flat.reshape(res, res) if flat.size == n_points else np.zeros((res, res))
                if negate:
                    z_surface = 1.0 - z_surface
            except (ValueError, AttributeError, TypeError):
                z_surface = np.zeros((res, res), dtype=np.float64)
        z_surface = np.where(np.isfinite(z_surface), z_surface, 0.0)
        if not self.is_classifier and z_surface.size:
            z_min, z_max = float(np.min(z_surface)), float(np.max(z_surface))
        else:
            z_min, z_max = 0.0, 1.0
        try:
            booster_names = list(getattr(booster, "feature_names", None) or [])
        except TypeError:
            booster_names = []
        topo, split_lines, cat_idx = self._topology_parts(
            booster, booster_type, int(n_features), booster_names
        )
        importances = {
            m: [round(float(v), 5) for v in self._metric_importances(booster, int(n_features), m)]
            for m in ("gain", "weight", "cover", "total_gain", "total_cover")
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
                "algorithm": type(_final_xgb_estimator(model_obj)).__name__,
                "objective": objective,
                "booster_type": booster_type,
                "model_family": "xgboost",
                "is_classifier": bool(self.is_classifier),
                "is_regressor": bool(not self.is_classifier),
                "is_binary": bool(self.is_classifier and n_classes == 2),
                "learning_rate": float(learning_rate) if np.isfinite(learning_rate) else 0.3,
                "n_boost_rounds": int(eff_rounds),
                "n_rounds_total": int(eff_rounds),
                "feature_importances": importances,
                "tree_topology_sample": topo,
                "tree_topology_rounds": [t["round"] for t in topo],
                "split_lines_sample": split_lines,
                "categorical_feature_idx": cat_idx,
                "regression_z_range": [float(z_min), float(z_max)],
                "boundary_threshold": None if not self.is_classifier else 0.5,
            },
        )


@ModelRegistry.register("xgboost")
class XGBClassifierAdapter(_XGBBase):
    display_name = "XGBoost (Exact TreeSHAP Leaf Flow)"
    family_name = "Gradient Boosting"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = True
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        if not _XGB_AVAILABLE:
            return False
        final_est = _final_xgb_estimator(estimator)
        if isinstance(final_est, _CLASSIFIER_TYPES):
            return True
        booster = _as_booster(final_est)
        if booster is None:
            return False
        objective = str(_booster_config(booster).get("objective", ""))
        return objective.startswith("binary:") or objective.startswith("multi:")

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


@ModelRegistry.register("xgboost_regressor")
class XGBRegressorAdapter(_XGBBase):
    display_name = "XGBoost Regressor (Exact TreeSHAP Leaf Flow)"
    family_name = "Gradient Boosting"
    task_type = TaskType.REGRESSION
    visual_3d_type = Visual3DType.ENSEMBLE_SMOOTH_TERRAIN
    is_classifier = False
    supports_nan_input = True

    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        if not _XGB_AVAILABLE:
            return False
        final_est = _final_xgb_estimator(estimator)
        if isinstance(final_est, _REGRESSOR_TYPES):
            return True
        booster = _as_booster(final_est)
        if booster is None:
            return False
        objective = str(_booster_config(booster).get("objective", ""))
        return not (objective.startswith("binary:") or objective.startswith("multi:"))

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

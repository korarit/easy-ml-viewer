# backend/app/domains/model_engine/adapters/neural_onnx/pipeline_adapter.py
"""Phase 37 — Scikit-Learn ``Pipeline`` & ``ColumnTransformer`` unwrapper (composite delegation).

Covers ``sklearn.pipeline.Pipeline`` artifacts (plain ``.pkl`` / ``.joblib``)
whose terminal step is any Phase 04 / 18–36 estimator, with arbitrary
preprocessing prefix (``SimpleImputer`` → ``StandardScaler`` / ``RobustScaler`` /
``OneHotEncoder`` / ``PolynomialFeatures``), including a nested
``ColumnTransformer`` (itself possibly holding nested ``Pipeline`` objects).

Blueprint mapping (phase-37-sklearn-pipeline-unwrapper.md):

- §2.1 composite function ``y_hat = M_final(T_K(...T_1(x_raw)))`` — inference
  always runs through the full ``pipeline.predict[_proba]``; staged tracing
  unwraps to ``pipeline.steps[-1][1]`` and delegates to its Phase-specific
  adapter (``ModelRegistry.resolve_adapter(final_est)``).
- §2.2 lineage graph ``G: raw -> [trans...]`` built from
  ``preprocessor.get_feature_names_out()`` with per-transformer fallback, so
  ``Phi_raw[j] = sum_{m in G(j)} phi_trans[m]`` (Shapley additivity).
- §2.3 raw-space 3D slicing — the ``res x res`` grid is synthesized in raw
  domain units (``pd.DataFrame`` with ``feature_names_in_`` columns) and pushed
  through ``pipeline[:-1].transform`` before delegating to the inner adapter.
- §6 memory guards — every ``transform`` runs in ``<= 2_500``-row chunks,
  ``scipy.sparse`` output is densified per-chunk to ``float32``, and step
  previews are truncated to the top-16 non-zero columns.

Edge cases (§7):

1. ``ColumnTransformer`` with string column selectors **requires** a
   ``pd.DataFrame`` — never feed it a bare ``np.ndarray``.
2. ``OneHotEncoder(sparse_output=True)`` returns ``csr_matrix`` — densify
   per chunk immediately.
3. Nested ``Pipeline`` inside ``ColumnTransformer`` — recursive unwrapping.
4. What-If in raw units — callers must route raw edits through
   ``pipeline[:-1].transform`` (see ``transform_raw_frame``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

PREDICT_CHUNK_SIZE = 2_500
MAX_TRANSFORMED_COLS_TELEMETRY = 512
MAX_STEP_PREVIEW_COLS = 16
MAX_SURFACE_RESOLUTION = 100


# ---------------------------------------------------------------------------
# Detection + unwrapping helpers
# ---------------------------------------------------------------------------

def is_sklearn_pipeline(obj: Any) -> bool:
    """True when ``obj`` is a fitted ``sklearn`` Pipeline (or GridSearchCV wrapper)."""
    target = obj
    seen = 0
    while seen < 8:
        seen += 1
        best = getattr(target, "best_estimator_", None)
        if best is not None:
            target = best
            continue
        break
    steps = getattr(target, "steps", None)
    return (
        isinstance(steps, list)
        and len(steps) >= 1
        and all(isinstance(s, tuple) and len(s) == 2 for s in steps)
    )


def _resolve_pipeline_root(obj: Any) -> Any:
    """Follow ``best_estimator_`` chains to the Pipeline root."""
    seen = 0
    while seen < 8:
        seen += 1
        best = getattr(obj, "best_estimator_", None)
        if best is not None:
            obj = best
            continue
        break
    return obj


def final_estimator_of(obj: Any) -> Any:
    """Terminal learner of a Pipeline (or the object itself when bare)."""
    root = _resolve_pipeline_root(obj)
    steps = getattr(root, "steps", None)
    if isinstance(steps, list) and steps:
        last = steps[-1]
        if isinstance(last, (tuple, list)) and len(last) == 2:
            return last[1]
        return last
    return root


def preprocessing_steps_of(obj: Any) -> list[tuple[str, Any]]:
    """All ``(name, transformer)`` pairs before the terminal estimator."""
    root = _resolve_pipeline_root(obj)
    steps = getattr(root, "steps", None)
    if isinstance(steps, list) and len(steps) >= 2:
        out: list[tuple[str, Any]] = []
        for s in steps[:-1]:
            if isinstance(s, (tuple, list)) and len(s) == 2:
                out.append((str(s[0]), s[1]))
        return out
    return []


def raw_feature_names_of(pipe: Any, n_fallback: int = 0) -> list[str]:
    """Raw input schema (``feature_names_in_`` from step 0)."""
    for attr in ("feature_names_in_", "feature_names_"):
        try:
            raw = getattr(pipe, attr, None)
            if raw is not None and len(list(raw)) >= 1:
                return [str(v) for v in list(raw)]
        except (TypeError, ValueError):
            continue
    root = _resolve_pipeline_root(pipe)
    steps = getattr(root, "steps", None)
    if isinstance(steps, list):
        for s in steps:
            est = s[1] if isinstance(s, (tuple, list)) and len(s) == 2 else s
            for attr in ("feature_names_in_", "feature_names_"):
                try:
                    raw = getattr(est, attr, None)
                    if raw is not None and len(list(raw)) >= 1:
                        return [str(v) for v in list(raw)]
                except (TypeError, ValueError):
                    continue
    return [f"feature_{i}" for i in range(max(int(n_fallback), 0))]


def _is_sparse(matrix: Any) -> bool:
    try:
        import scipy.sparse as _sp  # noqa: PLC0415

        return bool(_sp.issparse(matrix))
    except ImportError:
        return False


def _densify_float32(matrix: Any) -> np.ndarray:
    if _is_sparse(matrix):
        matrix = matrix.toarray()
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _trans_output_names(trans: Any, input_cols: list[str]) -> list[str]:
    """Output column names of one transformer (recursive for nested Pipelines)."""
    if trans is None or isinstance(trans, str):
        if trans == "drop":
            return []
        return list(input_cols)  # "passthrough" and friends preserve names
    # Nested Pipeline (e.g. num_pipe = imputer -> scaler): names preserved
    # unless an inner step expands them (rare) — recurse faithfully.
    nested_steps = getattr(trans, "steps", None)
    if isinstance(nested_steps, list) and nested_steps:
        cur = list(input_cols)
        for s in nested_steps:
            est = s[1] if isinstance(s, (tuple, list)) and len(s) == 2 else s
            if est is None or isinstance(est, str):
                if est == "drop":
                    return []
                continue
            fn = getattr(est, "get_feature_names_out", None)
            if callable(fn):
                try:
                    cur = [str(v) for v in list(fn(cur))]
                    continue
                except (TypeError, ValueError, AttributeError):
                    pass
                try:
                    cur = [str(v) for v in list(fn())]
                    continue
                except (TypeError, ValueError, AttributeError):
                    pass
        return cur
    fn = getattr(trans, "get_feature_names_out", None)
    if callable(fn):
        try:
            return [str(v) for v in list(fn(input_cols))]
        except (TypeError, ValueError, AttributeError):
            pass
        try:
            return [str(v) for v in list(fn())]
        except (TypeError, ValueError, AttributeError):
            pass
    # Transformers without name expansion (imputers / scalers) preserve width
    # when possible; otherwise fall back to input names.
    try:
        n_out = int(getattr(trans, "n_features_out_", len(input_cols)) or len(input_cols))
    except (TypeError, ValueError):
        n_out = len(input_cols)
    if n_out == len(input_cols):
        return list(input_cols)
    return [f"{input_cols[0] if input_cols else 'x'}__{i}" for i in range(max(n_out, 0))]


def _ct_transformers(ct: Any) -> list[tuple[str, Any, list[str]]]:
    """Normalize ``ColumnTransformer.transformers_`` to ``(name, trans, cols)``."""
    raw = getattr(ct, "transformers_", None)
    if raw is None:
        raw = getattr(ct, "transformers", None)
    out: list[tuple[str, Any, list[str]]] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            continue
        name, trans, cols = entry
        try:
            if isinstance(cols, (str, int)):
                cols_list = [str(cols)]
            else:
                cols_list = [str(c) for c in list(cols)]
        except (TypeError, ValueError):
            continue
        out.append((str(name), trans, cols_list))
    return out


def build_lineage_map(
    pre_steps: list[tuple[str, Any]], raw_names: list[str]
) -> tuple[list[str], dict[str, list[str]]]:
    """Return ``(transformed_names, lineage)`` for a preprocessing prefix.

    ``lineage[raw_col]`` lists the transformed columns derived from it.
    Steps that preserve width map 1-1; ``ColumnTransformer`` branches fan out
    per ``(name, trans, cols)`` group; ``OneHotEncoder`` / ``PolynomialFeatures``
    fan out via ``get_feature_names_out``.
    """
    current = list(raw_names)
    lineage: dict[str, list[str]] = {name: [name] for name in raw_names}
    for step_name, trans in pre_steps:
        # ColumnTransformer branch — the only fanning step in practice.
        if _ct_transformers(trans):
            new_current: list[str] = []
            new_lineage: dict[str, list[str]] = {name: [] for name in raw_names}
            try:
                verbose = bool(getattr(trans, "verbose_feature_names_out", True))
            except (AttributeError, TypeError, ValueError):
                verbose = True
            for branch, branch_trans, branch_cols in _ct_transformers(trans):
                branch_out = _trans_output_names(branch_trans, branch_cols)
                prefixed = (
                    [f"{branch}__{c}" for c in branch_out]
                    if verbose
                    else list(branch_out)
                )
                new_current.extend(prefixed)
                # Attribute each output back to its raw inputs: 1-1 widths pair
                # exactly; OHE expansions partition by input-name prefix
                # (``city_Bangkok`` -> ``city``) so every transformed column
                # has exactly one raw owner (Shapley additivity preserved).
                if len(branch_out) == len(branch_cols):
                    for raw_col, full in zip(branch_cols, prefixed):
                        if raw_col in new_lineage:
                            new_lineage[raw_col].append(full)
                else:
                    for unprefixed, full in zip(branch_out, prefixed):
                        owner: str | None = None
                        for raw_col in branch_cols:
                            if unprefixed == raw_col or unprefixed.startswith(raw_col + "_"):
                                owner = raw_col
                                break
                        if owner is not None and owner in new_lineage:
                            new_lineage[owner].append(full)
                        else:
                            # Fallback partition: round-robin by output index so
                            # each transformed column still has one owner.
                            idx = len(new_current) - len(prefixed) + list(prefixed).index(full)
                            fallback = branch_cols[idx % len(branch_cols)] if branch_cols else None
                            if fallback is not None and fallback in new_lineage:
                                new_lineage[fallback].append(full)
            # ``remainder='passthrough'`` columns keep their names.
            try:
                remainder = getattr(trans, "remainder", "drop")
            except (AttributeError, TypeError, ValueError):
                remainder = "drop"
            if remainder == "passthrough":
                covered = {c for _, _, cols in _ct_transformers(trans) for c in cols}
                for name in current:
                    if name not in covered:
                        new_current.append(name)
                        if name in new_lineage:
                            new_lineage[name].append(name)
            current = new_current
            lineage = new_lineage
            _ = step_name
            continue
        # Plain Pipeline-style step (imputer / scaler / encoder / poly).
        fn = getattr(trans, "get_feature_names_out", None)
        if callable(fn):
            try:
                expanded = [str(v) for v in list(fn(current))]
            except (TypeError, ValueError, AttributeError):
                try:
                    expanded = [str(v) for v in list(fn())]
                except (TypeError, ValueError, AttributeError):
                    expanded = list(current)
            if len(expanded) == len(current):
                current = expanded
                # 1-1 rename: rebuild positional pairing when each raw maps to one col.
                if all(len(v) == 1 for v in lineage.values()) and len(expanded) == len(raw_names):
                    lineage = {raw: [new] for raw, new in zip(raw_names, expanded)}
            else:
                # Fan-out (bare OneHotEncoder / PolynomialFeatures).
                lineage = {raw: list(expanded) for raw in raw_names}
                current = expanded
        # Imputers / scalers without get_feature_names_out preserve names.
    return current, lineage


def _dataframe_from_chunk(X_chunk: Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(X_chunk, pd.DataFrame):
        try:
            return X_chunk.reindex(columns=feature_names)
        except (TypeError, ValueError, AttributeError):
            return X_chunk
    arr = np.asarray(X_chunk, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    width = int(arr.shape[1]) if arr.ndim == 2 else len(feature_names)
    cols = feature_names if len(feature_names) == width else [f"feature_{i}" for i in range(width)]
    try:
        df = pd.DataFrame(arr, columns=cols)
    except (TypeError, ValueError):
        df = pd.DataFrame([list(r) for r in arr.tolist()], columns=cols)
    # Normalize Python None -> NaN so SimpleImputer treats them as missing.
    try:
        df = df.replace({None: np.nan})
    except (TypeError, ValueError, AttributeError):
        pass
    return df


def _step_kind(trans: Any) -> str:
    name = type(trans).__name__ if trans is not None else "drop"
    lowered = name.lower()
    if "imputer" in lowered:
        return "imputer"
    if "onehot" in lowered or "ordinal" in lowered or "encoder" in lowered:
        return "encoder"
    if "scaler" in lowered or "normalizer" in lowered or "standard" in lowered:
        return "scaler"
    if "polynomial" in lowered:
        return "polynomial"
    if "columntransformer" in lowered:
        return "column_transformer"
    if "pipeline" in lowered:
        return "pipeline"
    return "transformer"


@dataclass
class PipelineStepTrace:
    """Per-step summary for the HUD tracer (JSON-serializable via ``to_dict``)."""

    step_index: int = 0
    step_name: str = ""
    kind: str = "transformer"
    n_in: int = 0
    n_out: int = 0
    input_columns: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "step_name": self.step_name,
            "kind": self.kind,
            "n_in": int(self.n_in),
            "n_out": int(self.n_out),
            "input_columns": list(self.input_columns),
            "output_columns": list(self.output_columns),
        }


@dataclass
class FeatureLineageMap:
    """Raw → transformed column mapping (``raw -> [trans...]``)."""

    raw_names: list[str] = field(default_factory=list)
    transformed_names: list[str] = field(default_factory=list)
    mapping: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_names": list(self.raw_names),
            "transformed_names": list(self.transformed_names),
            "mapping": {k: list(v) for k, v in self.mapping.items()},
        }


def _inner_adapter_for(model_obj: Any) -> BaseModelAdapter:
    """Resolve the terminal estimator's Phase-specific adapter."""
    final_est = final_estimator_of(model_obj)
    return ModelRegistry.resolve_adapter(final_est, ".joblib")


def _raw_frame_medians(df: pd.DataFrame, raw_names: list[str]) -> dict[str, Any]:
    """Central values per raw column (median for numeric, mode for categorical)."""
    medians: dict[str, Any] = {}
    for col in raw_names:
        if col not in df.columns:
            continue
        series = df[col]
        try:
            if pd.api.types.is_numeric_dtype(series.dtype):
                with np.errstate(all="ignore"):
                    med = float(pd.to_numeric(series, errors="coerce").median())
                medians[col] = float(med) if np.isfinite(med) else 0.0
            else:
                mode = series.mode(dropna=True)
                medians[col] = mode.iloc[0] if len(mode) else ""
        except (TypeError, ValueError):
            medians[col] = 0.0
    return medians


class SklearnPipelineAdapter(BaseModelAdapter):
    """Composite adapter — preprocessing tunnel + delegated terminal estimator."""

    display_name = "Sklearn Pipeline (Raw → Transformed Tunnel)"
    family_name = "Preprocessing Pipeline"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE
    supports_nan_input = True  # raw space routinely holds NaN / unseen categories

    def __init__(self, pipeline: Any | None = None, inner: BaseModelAdapter | None = None) -> None:
        self._bound_pipeline: Any | None = pipeline
        self._bound_inner: BaseModelAdapter | None = inner

    # -- introspection ---------------------------------------------------
    @classmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        try:
            root = _resolve_pipeline_root(estimator)
        except (AttributeError, TypeError, ValueError):
            return False
        steps = getattr(root, "steps", None)
        if not (isinstance(steps, list) and len(steps) >= 1):
            return False
        # Backward compat: bare single-estimator Pipelines and pure
        # scaler-only prefixes stay with the inner adapter so Phase 03/36
        # registry tests keep resolving to e.g. ``linear_regression`` /
        # ``mlp``. Only claim pipelines holding a ColumnTransformer /
        # Imputer / Encoder / PolynomialFeatures (the tunnel actually
        # changes width, dtypes, or missingness).
        if len(steps) == 1:
            return False
        for _, trans in steps[:-1]:
            kind = _step_kind(trans)
            if kind in ("imputer", "encoder", "polynomial", "column_transformer", "pipeline"):
                return True
            if _ct_transformers(trans):
                return True
            if type(trans).__name__ == "ColumnTransformer":
                return True
        return False

    def _resolve(self, model_obj: Any | None) -> tuple[Any, BaseModelAdapter]:
        pipe = model_obj if model_obj is not None else self._bound_pipeline
        if pipe is None:
            raise ValueError("SklearnPipelineAdapter requires a fitted Pipeline object")
        pipe = _resolve_pipeline_root(pipe)
        if not is_sklearn_pipeline(pipe) and getattr(pipe, "steps", None) is None:
            raise ValueError(f"Estimator '{type(pipe).__name__}' is not a sklearn Pipeline")
        inner = self._bound_inner
        if inner is None:
            inner = _inner_adapter_for(pipe)
        return pipe, inner

    def extract_schema(self, pipe: Any | None = None) -> Any:
        """Raw input schema + lineage (mirrors ``UniversalSchemaDetector``)."""
        from app.domains.model_engine.schema_detector import (  # noqa: PLC0415
            UniversalSchemaDetector,
        )

        target = pipe if pipe is not None else self._bound_pipeline
        if target is None:
            raise ValueError("extract_schema requires a Pipeline (pipe or bound)")
        return UniversalSchemaDetector.extract_schema(target)

    def trace_preprocessing_steps(
        self, df_chunk: pd.DataFrame, pipe: Any | None = None
    ) -> tuple[np.ndarray, list[PipelineStepTrace], FeatureLineageMap]:
        """Run ``steps[:-1]`` chunk-wise; densify sparse output to float32."""
        target = pipe if pipe is not None else self._bound_pipeline
        if target is None:
            raise ValueError("trace_preprocessing_steps requires a Pipeline (pipe or bound)")
        return self.trace_preprocessing_steps_for(target, df_chunk)

    def trace_preprocessing_steps_for(
        self, pipe: Any, df_chunk: pd.DataFrame
    ) -> tuple[np.ndarray, list[PipelineStepTrace], FeatureLineageMap]:
        pre = preprocessing_steps_of(pipe)
        raw_names = raw_feature_names_of(pipe, n_fallback=int(df_chunk.shape[1]))
        frame = _dataframe_from_chunk(df_chunk, raw_names)
        current: Any = frame
        current_names = list(raw_names)
        traces: list[PipelineStepTrace] = []
        for idx, (step_name, trans) in enumerate(pre):
            n_in = int(current.shape[1]) if hasattr(current, "shape") else len(current_names)
            try:
                out = trans.transform(current)
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(
                    f"Pipeline step '{step_name}' failed to transform: {exc}"
                ) from exc
            if _is_sparse(out):
                out = _densify_float32(out)
            elif isinstance(out, pd.DataFrame):
                out = out
            else:
                arr = np.asarray(out)
                # Preserve 2D; some transformers squeeze single-column output.
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                out = arr
            if isinstance(out, pd.DataFrame):
                next_names = list(map(str, out.columns))
                current = out
            else:
                out_arr = _densify_float32(out) if not isinstance(out, np.ndarray) or out.dtype != np.float32 else np.ascontiguousarray(out, dtype=np.float32)
                # Infer output names for this single step.
                fn = getattr(trans, "get_feature_names_out", None)
                if callable(fn):
                    try:
                        next_names = [str(v) for v in list(fn(current_names))]
                    except (TypeError, ValueError, AttributeError):
                        try:
                            next_names = [str(v) for v in list(fn())]
                        except (TypeError, ValueError, AttributeError):
                            next_names = (
                                list(current_names)
                                if out_arr.shape[1] == len(current_names)
                                else [f"{step_name}__{i}" for i in range(int(out_arr.shape[1]))]
                            )
                elif _ct_transformers(trans):
                    ct_names, _ = build_lineage_map([(step_name, trans)], current_names)
                    next_names = ct_names
                else:
                    next_names = (
                        list(current_names)
                        if out_arr.shape[1] == len(current_names)
                        else [f"{step_name}__{i}" for i in range(int(out_arr.shape[1]))]
                    )
                current = out_arr
            traces.append(PipelineStepTrace(
                step_index=int(idx),
                step_name=str(step_name),
                kind=_step_kind(trans),
                n_in=int(n_in),
                n_out=int(current.shape[1]) if hasattr(current, "shape") else len(next_names),
                input_columns=list(current_names)[:MAX_TRANSFORMED_COLS_TELEMETRY],
                output_columns=list(next_names)[:MAX_TRANSFORMED_COLS_TELEMETRY],
            ))
            current_names = next_names
        if isinstance(current, pd.DataFrame):
            X_trans = _densify_float32(current.to_numpy())
            trans_names = list(map(str, current.columns))
        else:
            X_trans = np.ascontiguousarray(np.asarray(current, dtype=np.float32))
            trans_names = list(current_names)
        _, lineage = build_lineage_map(pre, raw_names)
        return X_trans, traces, FeatureLineageMap(
            raw_names=list(raw_names), transformed_names=list(trans_names), mapping=dict(lineage)
        )

    def aggregate_lineage_attributions(
        self, transformed_contribs: dict[str, float] | np.ndarray, lineage: FeatureLineageMap | dict[str, list[str]]
    ) -> dict[str, float]:
        """Sum transformed contributions back onto raw columns (``Phi_raw``)."""
        mapping = lineage.mapping if isinstance(lineage, FeatureLineageMap) else dict(lineage)
        if isinstance(transformed_contribs, np.ndarray):
            # Caller must supply names separately — unsupported here; treat by index.
            raise ValueError("aggregate requires a {transformed_name: value} mapping")
        raw: dict[str, float] = {}
        for raw_col, trans_cols in mapping.items():
            total = 0.0
            for tcol in trans_cols:
                try:
                    total += float(transformed_contribs.get(tcol, 0.0))
                except (TypeError, ValueError):
                    continue
            raw[raw_col] = float(total) if np.isfinite(total) else 0.0
        return raw

    def transform_raw_frame(
        self, pipe: Any, df_raw: pd.DataFrame
    ) -> np.ndarray:
        """What-If helper — raw-unit edits → transformed ``float32`` matrix."""
        X_trans, _, _ = self.trace_preprocessing_steps_for(pipe, df_raw)
        return X_trans

    # -- BaseModelAdapter contract -------------------------------------
    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        try:
            pipe, inner = self._resolve(estimator)
            raw_names = raw_feature_names_of(pipe, n_fallback=int(n_features))
            pre = preprocessing_steps_of(pipe)
            trans_names, lineage_map = build_lineage_map(pre, raw_names)
            try:
                inner_weights = inner.extract_feature_weights(final_estimator_of(pipe), len(trans_names))
            except (TypeError, ValueError, AttributeError):
                return [None for _ in range(max(int(n_features), 0))]
            trans_map = {
                name: (w if w is not None and np.isfinite(w) else 0.0)
                for name, w in zip(trans_names, inner_weights)
            }
            out: list[float | None] = []
            for raw in raw_names[: max(int(n_features), 0)]:
                cols = lineage_map.get(raw, [])
                if not cols:
                    out.append(None)
                    continue
                vals = [trans_map.get(c, 0.0) for c in cols]
                out.append(float(sum(vals)))
            while len(out) < max(int(n_features), 0):
                out.append(None)
            return out
        except (TypeError, ValueError, AttributeError):
            return [None for _ in range(max(int(n_features), 0))]

    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        pipe, inner = self._resolve(model_obj)
        # ColumnTransformer demands a DataFrame with raw column names (§7.1).
        if isinstance(X_float32, pd.DataFrame):
            df_raw = X_float32
        else:
            raw_names = raw_feature_names_of(pipe, n_fallback=int(np.asarray(X_float32).shape[1]) if np.asarray(X_float32).ndim == 2 else 0)
            df_raw = _dataframe_from_chunk(X_float32, raw_names)
        n_rows = int(df_raw.shape[0])
        if n_rows == 0:
            return PredictionBatchResult(predictions=[], probabilities=None, confidences=None)
        # Chunked full-pipeline inference (never densify the whole file).
        all_preds: list[Any] = []
        all_proba: list[list[float]] | None = []
        has_proba = True
        for start in range(0, n_rows, PREDICT_CHUNK_SIZE):
            chunk_df = df_raw.iloc[start:start + PREDICT_CHUNK_SIZE]
            try:
                proba = getattr(pipe, "predict_proba", None)
                preds_chunk = np.asarray(pipe.predict(chunk_df)).ravel()
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Pipeline predict failed: {exc}") from exc
            all_preds.extend(preds_chunk.tolist())
            if has_proba and callable(getattr(pipe, "predict_proba", None)):
                try:
                    proba_chunk = np.asarray(pipe.predict_proba(chunk_df), dtype=np.float64)
                    if proba_chunk.ndim == 1:
                        proba_chunk = proba_chunk.reshape(-1, 1)
                    for row in proba_chunk.tolist():
                        all_proba.append([float(v) if np.isfinite(v) else 0.0 for v in row])
                except (TypeError, ValueError, AttributeError, NotImplementedError):
                    has_proba = False
                    all_proba = None
            else:
                has_proba = False
                all_proba = None
        py_preds: list[float | int | str] = []
        for v in all_preds:
            if isinstance(v, (bool, np.bool_)):
                py_preds.append(int(v))
            elif isinstance(v, (np.integer, int)):
                py_preds.append(int(v))
            elif isinstance(v, (np.floating, float)):
                f = float(v)
                py_preds.append(int(f) if float(f).is_integer() else f)
            else:
                py_preds.append(v)
        probabilities = all_proba if has_proba and all_proba is not None else None
        confidences: list[float] | None = None
        if probabilities is not None:
            confidences = [float(max(row)) if row else 0.0 for row in probabilities]
        _ = inner
        return PredictionBatchResult(
            predictions=py_preds, probabilities=probabilities, confidences=confidences
        )

    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
        target_idx: int = 0,
        target_class_idx: int | None = None,
    ) -> list[InternalFlowRowTrace]:
        pipe, inner = self._resolve(model_obj)
        raw_names = (
            list(feature_names)
            if feature_names and len(feature_names) == int(np.asarray(X_float32).shape[1])
            else raw_feature_names_of(pipe, n_fallback=int(np.asarray(X_float32).shape[1]))
        )
        df_raw = _dataframe_from_chunk(X_float32, raw_names)
        X_trans, step_traces, lineage = self.trace_preprocessing_steps_for(pipe, df_raw)
        trans_names = list(lineage.transformed_names)
        # Delegate per-row mechanics to the terminal adapter in transformed space.
        inner_traces = inner.trace_internal_flow(
            final_estimator_of(pipe), X_trans, list(row_indices), trans_names,
            *(() if target_class_idx is None else ()),
        ) if False else None
        # Call with/without target selectors depending on inner signature.
        try:
            inner_traces = inner.trace_internal_flow(  # type: ignore[call-arg]
                final_estimator_of(pipe), X_trans, list(row_indices), trans_names,
                target_idx=target_idx, target_class_idx=target_class_idx,
            )
        except TypeError:
            try:
                inner_traces = inner.trace_internal_flow(  # type: ignore[call-arg]
                    final_estimator_of(pipe), X_trans, list(row_indices), trans_names, target_idx
                )
            except TypeError:
                inner_traces = inner.trace_internal_flow(
                    final_estimator_of(pipe), X_trans, list(row_indices), trans_names
                )
        out: list[InternalFlowRowTrace] = []
        for trace in inner_traces or []:
            trans_contribs = dict(trace.feature_contributions)
            raw_contribs = self.aggregate_lineage_attributions(trans_contribs, lineage)
            # Per-row raw slice + imputation / OOV flags.
            rid = int(trace.row_index)
            raw_row: dict[str, Any] = {}
            imputed_cols: list[str] = []
            try:
                if 0 <= rid < len(df_raw):
                    row = df_raw.iloc[rid]
                    for col in raw_names:
                        try:
                            val = row[col]
                        except (KeyError, TypeError):
                            continue
                        if isinstance(val, (np.generic,)):
                            val = val.item()
                        raw_row[str(col)] = val
                        try:
                            if pd.isna(val):
                                imputed_cols.append(str(col))
                        except (TypeError, ValueError):
                            pass
            except (TypeError, ValueError, IndexError, AttributeError):
                pass
            # Transformed slice truncated to top-16 non-zero actives (§6.3).
            trans_row: dict[str, float] = {}
            try:
                if 0 <= rid < int(X_trans.shape[0]):
                    vec = np.asarray(X_trans[rid], dtype=np.float64).ravel()
                    nz = np.where(np.abs(vec) > 1e-12)[0]
                    order = nz[np.argsort(-np.abs(vec[nz]), kind="stable")][:MAX_STEP_PREVIEW_COLS] if nz.size else np.asarray([], dtype=int)
                    for j in order.tolist():
                        name = trans_names[int(j)] if int(j) < len(trans_names) else f"trans_{int(j)}"
                        v = float(vec[int(j)])
                        trans_row[name] = v if np.isfinite(v) else 0.0
            except (TypeError, ValueError, IndexError):
                pass
            # OOV detection: OHE branch all-zeros while raw held a string.
            oov_cols: list[str] = []
            try:
                if 0 <= rid < int(X_trans.shape[0]):
                    vec = np.asarray(X_trans[rid], dtype=np.float64).ravel()
                    for raw_col, tcols in lineage.mapping.items():
                        idxs = [trans_names.index(c) for c in tcols if c in trans_names]
                        if not idxs:
                            continue
                        raw_val = raw_row.get(raw_col)
                        if isinstance(raw_val, str) and raw_val != "":
                            if all(abs(float(vec[i])) < 1e-12 for i in idxs if 0 <= i < vec.size):
                                oov_cols.append(raw_col)
            except (TypeError, ValueError):
                pass
            extra = dict(trace.extra_mechanics) if isinstance(trace.extra_mechanics, dict) else {}
            extra.update({
                "model_family": "sklearn_pipeline",
                "pipeline_steps": [t.to_dict() for t in step_traces],
                "pipeline_step_names": [t.step_name for t in step_traces],
                "pipeline_n_steps": len(step_traces),
                "raw_feature_names": list(raw_names),
                "transformed_feature_names": list(trans_names),
                "n_transformed_features": len(trans_names),
                "feature_lineage": lineage.to_dict(),
                "raw_row_values": raw_row,
                "transformed_row_top16": trans_row,
                "imputed_columns": imputed_cols,
                "has_imputed_values": bool(imputed_cols),
                "oov_columns": oov_cols,
                "has_oov_category": bool(oov_cols),
                "aggregated_raw_contributions": raw_contribs,
                "waterfall_modes": ["grouped_raw", "expanded_encoded"],
                "inner_model_family": extra.get("model_family", type(final_estimator_of(pipe)).__name__),
                "delegated_estimator": type(final_estimator_of(pipe)).__name__,
            })
            out.append(InternalFlowRowTrace(
                row_index=int(trace.row_index),
                base_value=float(trace.base_value),
                final_prediction=trace.final_prediction,
                feature_contributions={str(k): float(v) for k, v in raw_contribs.items()},
                extra_mechanics=extra,
            ))
        return out

    def compute_3d_surface(
        self,
        model_obj: Any,
        grid_spec: GridSpec3D,
        n_features: int,
        target_class_idx: int | None = None,
        target_idx: int = 0,
    ) -> SurfaceMesh3DData:
        pipe, inner = self._resolve(model_obj)
        raw_names = raw_feature_names_of(pipe, n_fallback=int(n_features))
        width_raw = len(raw_names) if raw_names else max(int(n_features), 1)
        res = int(min(max(grid_spec.resolution, 2), MAX_SURFACE_RESOLUTION))
        # Raw-space grid (§2.3): linspace on raw X/Y, medians/modes elsewhere.
        medians = dict(grid_spec.fixed_feature_medians or {})
        base_row: dict[str, Any] = {}
        for i, name in enumerate(raw_names):
            if i in medians:
                try:
                    base_row[name] = float(medians[i])
                    continue
                except (TypeError, ValueError):
                    base_row[name] = 0.0
            else:
                base_row[name] = 0.0
        ax, ay = int(grid_spec.axis_x_feature_index), int(grid_spec.axis_y_feature_index)
        x_name = raw_names[ax] if 0 <= ax < len(raw_names) else f"feature_{ax}"
        y_name = raw_names[ay] if 0 <= ay < len(raw_names) else f"feature_{ay}"
        x_lin = np.linspace(float(grid_spec.x_min), float(grid_spec.x_max), res).astype(np.float64)
        y_lin = np.linspace(float(grid_spec.y_min), float(grid_spec.y_max), res).astype(np.float64)
        # Categorical pinning: replace numeric pins on categorical raws with modes.
        cat_cols = self._categorical_raw_columns(pipe, raw_names)
        # Discover modes from fitted encoders when the median pin is numeric.
        for col in cat_cols:
            if col in (x_name, y_name):
                continue
            mode_val = self._preferred_category(pipe, col)
            if mode_val is not None:
                base_row[col] = mode_val
        # When an axis itself is categorical, tile its known categories.
        x_vals: list[Any] = [float(v) for v in x_lin]
        y_vals: list[Any] = [float(v) for v in y_lin]
        if x_name in cat_cols:
            cats = self._known_categories(pipe, x_name)
            if cats:
                x_vals = [cats[i % len(cats)] for i in range(res)]
                x_lin = np.arange(res, dtype=np.float64)
        if y_name in cat_cols:
            cats = self._known_categories(pipe, y_name)
            if cats:
                y_vals = [cats[i % len(cats)] for i in range(res)]
                y_lin = np.arange(res, dtype=np.float64)
        rows: list[dict[str, Any]] = []
        for yv in y_vals:
            for xv in x_vals:
                r = dict(base_row)
                r[x_name] = xv
                r[y_name] = yv
                rows.append(r)
        df_grid = pd.DataFrame(rows, columns=raw_names)
        # Chunked raw → transformed → inner predict.
        z_flat = np.zeros((res * res,), dtype=np.float64)
        for start in range(0, res * res, PREDICT_CHUNK_SIZE):
            chunk_df = df_grid.iloc[start:start + PREDICT_CHUNK_SIZE]
            X_trans_chunk, _, _ = self.trace_preprocessing_steps_for(pipe, chunk_df)
            chunk_res = inner.predict_batch(final_estimator_of(pipe), X_trans_chunk)
            if chunk_res.probabilities is not None:
                mat = np.asarray(chunk_res.probabilities, dtype=np.float64)
                if mat.ndim == 1:
                    mat = mat.reshape(-1, 1)
                resolved = int(target_class_idx if target_class_idx is not None else target_idx)
                resolved = min(max(resolved, 0), mat.shape[1] - 1) if mat.shape[1] else 0
                z_flat[start:start + len(chunk_df)] = np.asarray(mat[:, resolved], dtype=np.float64).ravel() if mat.shape[1] else 0.0
            else:
                vals = np.asarray(list(chunk_res.predictions), dtype=np.float64).ravel()
                z_flat[start:start + len(chunk_df)] = vals
            del chunk_df, X_trans_chunk, chunk_res
        z_flat = np.where(np.isfinite(z_flat), z_flat, 0.0)
        z_surface = z_flat.reshape(res, res)
        try:
            inner_visual = getattr(inner, "visual_3d_type", Visual3DType.FLAT_HYPERPLANE)
            visual = inner_visual if isinstance(inner_visual, Visual3DType) else Visual3DType.FLAT_HYPERPLANE
        except (AttributeError, TypeError, ValueError):
            visual = Visual3DType.FLAT_HYPERPLANE
        pre = preprocessing_steps_of(pipe)
        _trans_names, _lineage_map = build_lineage_map(pre, raw_names)
        return SurfaceMesh3DData(
            visual_type=visual,
            resolution=int(len(x_lin)),
            x_grid=[float(v) for v in np.asarray(x_lin).tolist()],
            y_grid=[float(v) for v in np.asarray(y_lin).tolist()],
            z_values=[float(v) for v in np.asarray(z_surface, dtype=np.float64).ravel().tolist()],
            extra_overlays={
                "model_family": "sklearn_pipeline",
                "delegated_estimator": type(final_estimator_of(pipe)).__name__,
                "delegated_visual_type": getattr(visual, "value", str(visual)),
                "coordinate_space": "raw_domain_units",
                "coordinate_space_toggle": ["raw_domain_units", "transformed_zscore_space"],
                "raw_feature_names": list(raw_names),
                "raw_axis_x": x_name,
                "raw_axis_y": y_name,
                "n_transformed_features": len(_trans_names),
                "feature_lineage": dict(_lineage_map),
                "pipeline_step_names": [str(n) for n, _ in pre] + [type(final_estimator_of(pipe)).__name__],
                "pipeline_n_steps": len(pre) + 1,
                "categorical_raw_columns": sorted(cat_cols),
                "is_classifier": bool(getattr(inner, "task_type", TaskType.REGRESSION) in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)),
            },
        )

    # -- categorical helpers -------------------------------------------
    @staticmethod
    def _categorical_raw_columns(pipe: Any, raw_names: list[str]) -> set[str]:
        cats: set[str] = set()
        for _, trans in preprocessing_steps_of(pipe):
            for _, branch_trans, branch_cols in _ct_transformers(trans):
                lname = type(branch_trans).__name__.lower() if branch_trans is not None else ""
                if "onehot" in lname or "ordinal" in lname or "encoder" in lname:
                    cats.update(branch_cols)
                # Nested pipeline holding an encoder.
                nested = getattr(branch_trans, "steps", None)
                if isinstance(nested, list):
                    for _, inner_step in nested:
                        if "onehot" in type(inner_step).__name__.lower() or "ordinal" in type(inner_step).__name__.lower():
                            cats.update(branch_cols)
                            break
        return {c for c in cats if c in raw_names}

    @staticmethod
    def _known_categories(pipe: Any, raw_col: str) -> list[Any]:
        for _, trans in preprocessing_steps_of(pipe):
            for _, branch_trans, branch_cols in _ct_transformers(trans):
                if raw_col not in branch_cols:
                    continue
                enc = branch_trans
                nested = getattr(branch_trans, "steps", None)
                if isinstance(nested, list) and nested:
                    for _, s in nested:
                        lname = type(s).__name__.lower()
                        if "onehot" in lname or "ordinal" in lname:
                            enc = s
                            break
                cats = getattr(enc, "categories_", None)
                if cats is not None:
                    try:
                        idx = list(branch_cols).index(raw_col)
                        return [c.item() if hasattr(c, "item") else c for c in list(cats[idx])]
                    except (TypeError, ValueError, IndexError, AttributeError):
                        continue
        return []

    def _preferred_category(self, pipe: Any, raw_col: str) -> Any | None:
        cats = self._known_categories(pipe, raw_col)
        if cats:
            return cats[0]
        return None


@ModelRegistry.register("sklearn_pipeline")
class SklearnPipelineClassifierAdapter(SklearnPipelineAdapter):
    display_name = "Sklearn Pipeline (Raw → Transformed Tunnel)"
    family_name = "Preprocessing Pipeline"
    task_type = TaskType.BINARY_CLASSIFICATION
    visual_3d_type = Visual3DType.FLAT_HYPERPLANE


class SklearnPipelineAdapterAlias(SklearnPipelineAdapter):
    """Spec facade: ``SklearnPipelineAdapter(pipeline, inner)`` direct use."""

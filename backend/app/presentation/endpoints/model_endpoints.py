"""Model REST endpoints — /api/v1/models/* (Phases 03 + 04 vertical slice)."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, Field

from app.application.model_use_case import ModelIngestUseCase
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import GridSpec3D

router = APIRouter(prefix="/models", tags=["models"])


def get_model_use_case() -> ModelIngestUseCase:
    return ModelIngestUseCase()


class TracePreviewRequest(BaseModel):
    # Phase 35: CatBoost native categoricals arrive as strings — accept Any cells.
    rows: list[list[Any]] = Field(min_length=1)
    row_indices: list[int] = Field(min_length=1)
    actual_targets: list[float | None] | None = None
    feature_names: list[str] | None = None
    target_idx: int = Field(default=0, ge=0)


class SurfacePreviewRequest(BaseModel):
    axis_x_feature_index: int = Field(ge=0)
    axis_y_feature_index: int = Field(ge=0)
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: int = Field(default=50, ge=2, le=200)
    fixed_feature_medians: dict[int, float] = Field(default_factory=dict)
    target_idx: int = Field(default=0, ge=0)


# Phase 04 §5.3: memoize identical surface requests (model + axes + grid).
_surface_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SURFACE_CACHE_MAX = 32


def _surface_cache_key(model_id: str, req: SurfacePreviewRequest) -> str:
    medians = ",".join(f"{k}={v}" for k, v in sorted(req.fixed_feature_medians.items()))
    return (
        f"{model_id}|tx={req.axis_x_feature_index}|ty={req.axis_y_feature_index}|"
        f"res={req.resolution}|x=[{req.x_min},{req.x_max}]|y=[{req.y_min},{req.y_max}]|"
        f"med={medians}|t={req.target_idx}"
    )


@router.post("/upload", summary="Upload .joblib/.pkl model and introspect schema")
async def upload_model(
    file: UploadFile = File(...),
    use_case: ModelIngestUseCase = Depends(get_model_use_case),
) -> dict[str, Any]:
    contents = await file.read()
    metadata = use_case.ingest_upload_contents(contents, filename=file.filename or "model.joblib")
    return metadata.model_dump(mode="json")


@router.post("/load-sample/{sample_key}", summary="Load a bundled sample model")
def load_sample_model(
    sample_key: str,
    use_case: ModelIngestUseCase = Depends(get_model_use_case),
) -> dict[str, Any]:
    return use_case.load_sample(sample_key).model_dump(mode="json")


@router.get("/supported", summary="List registered model algorithms")
def list_supported_models() -> dict[str, Any]:
    from app.domains.model_engine.registry import ModelRegistry

    return {"algorithms": ModelRegistry.supported_algorithms()}


@router.post("/{model_id}/trace-preview", summary="Phase 04: per-row linear equation trace")
def trace_preview(
    model_id: str,
    body: TracePreviewRequest,
    use_case: ModelIngestUseCase = Depends(get_model_use_case),
) -> dict[str, Any]:
    from app.domains.model_engine.adapters.neural_onnx.pipeline_adapter import (  # noqa: PLC0415
        SklearnPipelineAdapter,
        is_sklearn_pipeline,
    )

    model_obj = use_case.get_model(model_id)
    predict_obj, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
    _is_pipe = bool(is_sklearn_pipeline(predict_obj) or is_sklearn_pipeline(model_obj))
    if _is_pipe:
        try:
            schema = UniversalSchemaDetector.extract_schema(predict_obj)
        except Exception:
            schema = UniversalSchemaDetector.extract_schema(final_est)
        adapter: Any = SklearnPipelineAdapter()
    else:
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")
        schema = UniversalSchemaDetector.extract_schema(final_est)
    feature_names = body.feature_names or [f.name for f in schema.features]
    try:
        X = np.asarray(body.rows, dtype=np.float32)
    except (TypeError, ValueError):
        # Phase 35: mixed categorical (str) rows — keep object dtype for the Pool.
        X = np.asarray(body.rows, dtype=object)
    if X.ndim != 2 or X.shape[1] != schema.n_features_in:
        from app.core.exceptions import SchemaMismatchError

        raise SchemaMismatchError(
            f"Expected {schema.n_features_in} features per row, got shape {X.shape}",
            details={"expected": schema.n_features_in, "got": list(X.shape)},
        )
    try:
        traces = adapter.trace_internal_flow(
            model_obj, X, list(body.row_indices), feature_names, target_idx=body.target_idx
        )
    except TypeError:
        traces = adapter.trace_internal_flow(model_obj, X, list(body.row_indices), feature_names)
    out: list[dict[str, Any]] = []
    for i, tr in enumerate(traces):
        item = tr.model_dump(mode="json")
        actual = None
        if body.actual_targets is not None and i < len(body.actual_targets):
            actual = body.actual_targets[i]
        item["actual_target"] = actual
        try:
            item["residual_error"] = (
                float(actual - float(tr.final_prediction))
                if actual is not None
                else None
            )
        except (TypeError, ValueError):
            item["residual_error"] = None
        out.append(item)
    return {"traces": out}


@router.post("/{model_id}/surface-preview", summary="Phase 04: 3D hyperplane mesh")
def surface_preview(
    model_id: str,
    body: SurfacePreviewRequest,
    use_case: ModelIngestUseCase = Depends(get_model_use_case),
) -> dict[str, Any]:
    cache_key = _surface_cache_key(model_id, body)
    if cache_key in _surface_cache:
        _surface_cache.move_to_end(cache_key)
        return _surface_cache[cache_key]

    from app.domains.model_engine.adapters.neural_onnx.pipeline_adapter import (  # noqa: PLC0415
        SklearnPipelineAdapter as _PipeAdapter,
        is_sklearn_pipeline as _is_pipe_fn,
    )

    model_obj = use_case.get_model(model_id)
    predict_obj, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
    _is_pipe_surface = bool(_is_pipe_fn(predict_obj) or _is_pipe_fn(model_obj))
    if _is_pipe_surface:
        try:
            schema = UniversalSchemaDetector.extract_schema(predict_obj)
        except Exception:
            schema = UniversalSchemaDetector.extract_schema(final_est)
        adapter = _PipeAdapter()
    else:
        adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")
        schema = UniversalSchemaDetector.extract_schema(final_est)
    t0 = time.perf_counter()
    grid_spec = GridSpec3D(
        axis_x_feature_index=body.axis_x_feature_index,
        axis_y_feature_index=body.axis_y_feature_index,
        x_min=body.x_min,
        x_max=body.x_max,
        y_min=body.y_min,
        y_max=body.y_max,
        resolution=body.resolution,
        fixed_feature_medians=body.fixed_feature_medians,
    )
    try:
        # Phase 18: classifiers accept a target-class selector; regressors
        # (Phase 04) raise TypeError here and fall back below.
        surface = adapter.compute_3d_surface(  # type: ignore[call-arg]
            model_obj,
            grid_spec,
            n_features=schema.n_features_in,
            target_class_idx=body.target_idx,
        )
    except TypeError:
        surface = adapter.compute_3d_surface(
            model_obj,
            grid_spec,
            n_features=schema.n_features_in,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    payload = surface.model_dump(mode="json")
    # Telemetry drawer (§2.3): grid stats + compute latency.
    z = surface.z_values
    payload["telemetry"] = {
        "grid_points": len(z),
        "resolution": surface.resolution,
        "z_min": min(z) if z else 0.0,
        "z_max": max(z) if z else 0.0,
        "compute_ms": elapsed_ms,
    }
    _surface_cache[cache_key] = payload
    while len(_surface_cache) > _SURFACE_CACHE_MAX:
        _surface_cache.popitem(last=False)
    return payload

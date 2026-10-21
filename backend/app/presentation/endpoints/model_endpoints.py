"""Model REST endpoints — /api/v1/models/*."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile

from app.application.model_use_case import ModelIngestUseCase

router = APIRouter(prefix="/models", tags=["models"])


def get_model_use_case() -> ModelIngestUseCase:
    return ModelIngestUseCase()


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

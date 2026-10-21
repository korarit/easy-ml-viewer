"""Dataset REST endpoints — /api/v1/datasets/*."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.application.dataset_use_case import DatasetIngestUseCase

router = APIRouter(prefix="/datasets", tags=["datasets"])


def get_dataset_use_case() -> DatasetIngestUseCase:
    return DatasetIngestUseCase()


def _entity_to_response(entity: object) -> dict[str, Any]:
    data = entity.model_dump(mode="json")  # type: ignore[union-attr]
    data.pop("parquet_cache_path", None)
    return data


@router.post("/upload", summary="Upload CSV/TSV/XLSX/Parquet and profile it")
async def upload_dataset(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(default=None),
    use_case: DatasetIngestUseCase = Depends(get_dataset_use_case),
) -> dict[str, Any]:
    contents = await file.read()
    entity = use_case.ingest_upload_contents(
        contents, filename=file.filename or "upload.csv", sheet_name=sheet_name or None
    )
    return _entity_to_response(entity)


@router.post("/load-sample/{sample_name}", summary="Load a bundled sample dataset")
def load_sample_dataset(
    sample_name: str,
    use_case: DatasetIngestUseCase = Depends(get_dataset_use_case),
) -> dict[str, Any]:
    return _entity_to_response(use_case.load_sample(sample_name))

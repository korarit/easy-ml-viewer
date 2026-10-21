"""Mapping REST endpoints — /api/v1/mappings/* (Phase 05)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.application.mapping_use_case import MappingUseCase
from app.domains.mapping.value_objects import FeatureMappingItem

router = APIRouter(prefix="/mappings", tags=["mappings"])


def get_mapping_use_case() -> MappingUseCase:
    return MappingUseCase()


class AutoSuggestRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)


class ValidatePreviewRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    items: list[FeatureMappingItem] = Field(min_length=1)
    target_column_name: str | None = None


class ValidateAndSaveRequest(ValidatePreviewRequest):
    target_label_map: dict[str, int | float] | None = None


def _config_to_response(config: object, report: object) -> dict[str, Any]:
    cfg = config.model_dump(mode="json")  # type: ignore[union-attr]
    rep = report.model_dump(mode="json")  # type: ignore[union-attr]
    return {"mapping": cfg, "report": rep}


@router.post("/auto-suggest", summary="Phase 05: 3-tier fuzzy auto-match + dry-run")
def auto_suggest(body: AutoSuggestRequest, use_case: MappingUseCase = Depends(get_mapping_use_case)):
    config, report = use_case.auto_suggest(body.dataset_id, body.model_id)
    payload = _config_to_response(config, report)
    payload["supports_nan_input"] = use_case.supports_nan(body.model_id)
    return payload


@router.post("/map-by-order", summary="Phase 05: positional 0..N mapping + dry-run")
def map_by_order(body: AutoSuggestRequest, use_case: MappingUseCase = Depends(get_mapping_use_case)):
    config, report = use_case.map_by_order(body.dataset_id, body.model_id)
    payload = _config_to_response(config, report)
    payload["supports_nan_input"] = use_case.supports_nan(body.model_id)
    return payload


@router.post("/validate-preview", summary="Phase 05: dry-run first-100-row float32 check")
def validate_preview(
    body: ValidatePreviewRequest, use_case: MappingUseCase = Depends(get_mapping_use_case)
):
    report = use_case.validate_preview(body.dataset_id, body.model_id, body.items, body.target_column_name)
    payload = report.model_dump(mode="json")
    payload["supports_nan_input"] = use_case.supports_nan(body.model_id)
    # Echo resolved items so the UI can show fill values without a second call.
    payload["items"] = [i.model_dump(mode="json") for i in body.items]
    return payload


@router.post("/validate-and-save", summary="Phase 05: dry-run + commit mapping configuration")
def validate_and_save(
    body: ValidateAndSaveRequest, use_case: MappingUseCase = Depends(get_mapping_use_case)
):
    config, report = use_case.validate_and_save(
        body.dataset_id,
        body.model_id,
        body.items,
        body.target_column_name,
        body.target_label_map,
    )
    payload = _config_to_response(config, report)
    payload["supports_nan_input"] = use_case.supports_nan(body.model_id)
    return payload


@router.get("/{mapping_id}", summary="Phase 05: fetch a committed mapping")
def get_mapping(mapping_id: str, use_case: MappingUseCase = Depends(get_mapping_use_case)):
    return use_case.get(mapping_id).model_dump(mode="json")

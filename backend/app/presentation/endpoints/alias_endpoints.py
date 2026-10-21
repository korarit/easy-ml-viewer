"""Phase 14 — class/feature alias REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.application.alias_orchestrator import (
    AliasAutosuggestRequest,
    AliasOrchestrator,
    AliasUpsertRequest,
    alias_to_response,
)

router = APIRouter(prefix="/alias", tags=["studio-alias"])


def get_alias_orchestrator() -> AliasOrchestrator:
    return AliasOrchestrator()


@router.put("/{session_id}", summary="Phase 14: upsert class/feature alias config")
async def upsert_alias_config(
    session_id: str,
    body: AliasUpsertRequest,
    orchestrator: AliasOrchestrator = Depends(get_alias_orchestrator),
) -> dict[str, Any]:
    if body.session_id != session_id:
        body = AliasUpsertRequest(
            session_id=session_id,
            class_aliases=body.class_aliases,
            feature_aliases=body.feature_aliases,
        )
    entity = orchestrator.upsert(body)
    return alias_to_response(entity)


@router.get("/{session_id}", summary="Phase 14: fetch alias config")
async def get_alias_config(
    session_id: str,
    orchestrator: AliasOrchestrator = Depends(get_alias_orchestrator),
) -> dict[str, Any]:
    return alias_to_response(orchestrator.get(session_id))


@router.post("/autosuggest/preview", summary="Phase 14: auto-suggest aliases")
async def autosuggest_aliases(
    body: AliasAutosuggestRequest,
    orchestrator: AliasOrchestrator = Depends(get_alias_orchestrator),
) -> dict[str, Any]:
    return alias_to_response(orchestrator.autosuggest(body))


@router.get("/{session_id}/export", summary="Phase 14: export alias JSON")
async def export_alias_config(
    session_id: str,
    orchestrator: AliasOrchestrator = Depends(get_alias_orchestrator),
) -> dict[str, Any]:
    return {"alias_json": alias_to_response(orchestrator.get(session_id))}


@router.post("/{session_id}/import", summary="Phase 14: import alias JSON")
async def import_alias_config(
    session_id: str,
    payload: dict[str, Any],
    orchestrator: AliasOrchestrator = Depends(get_alias_orchestrator),
) -> dict[str, Any]:
    blob = payload.get("alias_json", payload)
    if not isinstance(blob, dict):
        from app.core.exceptions import DomainException

        raise DomainException(
            code="ALIAS_BAD_IMPORT",
            message="Import body must be an object with class_aliases/feature_aliases",
            http_status=422,
        )
    body = AliasUpsertRequest(
        session_id=session_id,
        class_aliases=blob.get("class_aliases", {}),
        feature_aliases=blob.get("feature_aliases", {}),
    )
    return alias_to_response(orchestrator.upsert(body))

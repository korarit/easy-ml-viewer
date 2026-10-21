"""Phase 12 — local explainability REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.application.explainability_orchestrator import (
    ExplainabilityOrchestrator,
    ExplainabilitySessionBuildRequest,
    frame_to_response,
    session_to_response,
)

router = APIRouter(prefix="/explainability", tags=["explainability"])


def get_explainability_orchestrator() -> ExplainabilityOrchestrator:
    return ExplainabilityOrchestrator()


@router.post("/sessions/build", summary="Phase 12: build explainability session")
async def build_explainability_session(
    body: ExplainabilitySessionBuildRequest,
    orchestrator: ExplainabilityOrchestrator = Depends(get_explainability_orchestrator),
) -> dict[str, Any]:
    state = await orchestrator.build_session(body)
    return session_to_response(state)


@router.get("/{session_id}/frames/{frame_index}", summary="Phase 12: frame explanation")
async def get_frame_explanation(
    session_id: str,
    frame_index: int,
    top_k: int = Query(default=10, ge=1, le=500),
    orchestrator: ExplainabilityOrchestrator = Depends(get_explainability_orchestrator),
) -> dict[str, Any]:
    entity = orchestrator.get_frame_explanation(session_id, frame_index, top_k=top_k)
    return frame_to_response(entity)

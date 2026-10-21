"""Phase 13 — what-if sandbox + counterfactual REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.application.whatif_orchestrator import (
    WhatIfCounterfactualRequest,
    WhatIfEvaluateRequest,
    WhatIfOrchestrator,
    WhatIfSessionBuildRequest,
    counterfactual_to_response,
    evaluation_to_response,
    session_to_response,
)

router = APIRouter(prefix="/whatif", tags=["whatif"])


def get_whatif_orchestrator() -> WhatIfOrchestrator:
    return WhatIfOrchestrator()


@router.post("/sessions/build", summary="Phase 13: build what-if session")
async def build_whatif_session(
    body: WhatIfSessionBuildRequest,
    orchestrator: WhatIfOrchestrator = Depends(get_whatif_orchestrator),
) -> dict[str, Any]:
    state = await orchestrator.build_session(body)
    ranges = WhatIfOrchestrator.session_feature_ranges(state.session_id)
    return session_to_response(state, ranges)


@router.post("/{session_id}/frames/{frame_index}/evaluate", summary="Phase 13: what-if mutation evaluate")
async def evaluate_whatif_mutation(
    session_id: str,
    frame_index: int,
    body: WhatIfEvaluateRequest,
    orchestrator: WhatIfOrchestrator = Depends(get_whatif_orchestrator),
) -> dict[str, Any]:
    entity = orchestrator.evaluate_frame(
        session_id, frame_index, dict(body.mutated_feature_values or {})
    )
    return evaluation_to_response(entity)


@router.post(
    "/{session_id}/frames/{frame_index}/counterfactual",
    summary="Phase 13: minimum-change counterfactual finder",
)
async def find_counterfactual(
    session_id: str,
    frame_index: int,
    body: WhatIfCounterfactualRequest,
    orchestrator: WhatIfOrchestrator = Depends(get_whatif_orchestrator),
) -> dict[str, Any]:
    entity = orchestrator.find_counterfactual(
        session_id,
        frame_index,
        body.target_class_or_threshold,
        list(body.locked_feature_indices or []),
        list(body.locked_feature_names or []),
    )
    return counterfactual_to_response(entity)

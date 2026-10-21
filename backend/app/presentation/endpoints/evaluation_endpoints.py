"""Phase 07 — Global evaluation REST endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.application.evaluation_use_case import ComputeEvaluationRequest, EvaluationUseCase

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


def get_evaluation_use_case() -> EvaluationUseCase:
    return EvaluationUseCase()


@router.post("/compute", summary="Phase 07: global evaluation + error slicer indices")
async def compute_evaluation(
    body: ComputeEvaluationRequest,
    use_case: EvaluationUseCase = Depends(get_evaluation_use_case),
) -> dict[str, Any]:
    summary = use_case.compute(body)
    return summary.model_dump(mode="json")

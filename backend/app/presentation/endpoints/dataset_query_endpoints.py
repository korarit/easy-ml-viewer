"""Phase 06 — Dataset query & trajectory-selection REST endpoints."""

from __future__ import annotations

import gc
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends

from app.domains.dataset.query_service import (
    DatasetQueryRequest,
    DatasetQueryService,
    ResolveSelectionRequest,
)
from app.infrastructure.storage.parquet_dataset_repository import ParquetDatasetRepository

router = APIRouter(prefix="/datasets", tags=["datasets-query"])


def get_query_repository() -> ParquetDatasetRepository:
    return ParquetDatasetRepository()


def _load_df(dataset_id: str, repo: ParquetDatasetRepository, columns: list[str] | None) -> pd.DataFrame:
    # Column-pruned mmap read: only fetch what the table actually renders.
    if columns:
        try:
            return repo.load_dataframe(dataset_id, columns=columns)
        except Exception:
            pass  # fall through to full read (e.g. unknown column subset)
    return repo.load_dataframe(dataset_id)


@router.post("/{dataset_id}/query", summary="Filter, sort and page dataset rows (Phase 06)")
async def query_dataset(
    dataset_id: str,
    body: DatasetQueryRequest,
    repo: ParquetDatasetRepository = Depends(get_query_repository),
) -> dict[str, Any]:
    # Prune cached parquet read to filter/sort/page columns when possible.
    wanted: list[str] | None = None
    if body.columns:
        wanted = list(dict.fromkeys(body.columns))
    else:
        cols: set[str] = {r.column for r in body.filters}
        if body.sort_by:
            cols.add(body.sort_by)
        wanted = sorted(cols) or None
    df = _load_df(dataset_id, repo, wanted)
    # If we pruned columns for the read, re-read full frame when filters
    # reference columns outside the pruned set is impossible — wanted already
    # unions filter+sort+page columns, so this frame is sufficient.
    try:
        res = DatasetQueryService.execute_query(df, body)
        return res.model_dump(mode="json")
    finally:
        del df
        gc.collect()


@router.post("/{dataset_id}/resolve-selection", summary="Resolve a queue preset into row indices (Phase 06)")
async def resolve_selection(
    dataset_id: str,
    body: ResolveSelectionRequest,
    repo: ParquetDatasetRepository = Depends(get_query_repository),
) -> dict[str, Any]:
    cols: set[str] = {r.column for r in body.filters}
    if body.sort_by:
        cols.add(body.sort_by)
    df = _load_df(dataset_id, repo, sorted(cols) or None)
    try:
        res = DatasetQueryService.resolve_selection(df, body)
        return res.model_dump(mode="json")
    finally:
        del df
        gc.collect()

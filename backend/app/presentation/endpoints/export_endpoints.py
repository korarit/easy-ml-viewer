"""Phase 16 — enriched-table export REST endpoints (CSV / XLSX / Parquet)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.application.export_orchestrator import (
    ExportEnrichedTableRequest,
    ExportOrchestrator,
)
from app.domains.export_reporting.services import EnrichedDatasetExportService

router = APIRouter(prefix="/exports", tags=["export-reporting"])

_CONTENT_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "parquet": "application/octet-stream",
}


def get_export_orchestrator() -> ExportOrchestrator:
    return ExportOrchestrator()


@router.post("/enriched-table", summary="Phase 16: export enriched predictions table")
async def export_enriched_table(
    body: ExportEnrichedTableRequest,
    orchestrator: ExportOrchestrator = Depends(get_export_orchestrator),
):
    data = await orchestrator.build_export_data(body)
    base_columns: list[str] = data["base_columns"]
    raw_rows: list[dict[str, Any]] = data["raw_rows"]
    enriched_rows: list[dict[str, Any]] = data["enriched_rows"]
    row_indices: list[int] = data["row_indices"]
    n = int(data["total_frames"])
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in body.dataset_id)[:40] or "export"

    if body.format == "csv":
        chunks = EnrichedDatasetExportService.stream_enriched_csv_chunks(
            base_columns, raw_rows, enriched_rows, row_indices
        )

        def _iter():
            yield from chunks

        return StreamingResponse(
            _iter(),
            media_type=_CONTENT_TYPES["csv"],
            headers={"Content-Disposition": f'attachment; filename="enriched-{safe}-n{n}.csv"'},
        )
    if body.format == "xlsx":
        blob = EnrichedDatasetExportService.build_xlsx_bytes(
            base_columns, raw_rows, enriched_rows, row_indices
        )

        def _iter_xlsx():
            yield blob

        return StreamingResponse(
            _iter_xlsx(),
            media_type=_CONTENT_TYPES["xlsx"],
            headers={"Content-Disposition": f'attachment; filename="enriched-{safe}-n{n}.xlsx"'},
        )
    blob = EnrichedDatasetExportService.build_parquet_bytes(
        base_columns, raw_rows, enriched_rows, row_indices
    )

    def _iter_parquet():
        yield blob

    return StreamingResponse(
        _iter_parquet(),
        media_type=_CONTENT_TYPES["parquet"],
        headers={"Content-Disposition": f'attachment; filename="enriched-{safe}-n{n}.parquet"'},
    )


@router.post("/enriched-table/preview", summary="Phase 16: preview first rows of an export")
async def preview_enriched_table(
    body: ExportEnrichedTableRequest,
    orchestrator: ExportOrchestrator = Depends(get_export_orchestrator),
) -> JSONResponse:
    data = await orchestrator.build_export_data(body)
    limit = 20
    header = EnrichedDatasetExportService.header_columns(data["base_columns"])
    rows: list[list[Any]] = []
    for t in range(min(limit, len(data["raw_rows"]))):
        rows.append(
            EnrichedDatasetExportService.row_to_list(
                data["base_columns"],
                data["raw_rows"][t],
                data["enriched_rows"][t],
                t,
                int(data["row_indices"][t]),
            )
        )
    return JSONResponse(
        {
            "header": header,
            "rows": rows,
            "total_frames": data["total_frames"],
            "is_classification": data["is_classification"],
            "format": body.format,
            "scope": body.scope,
        }
    )

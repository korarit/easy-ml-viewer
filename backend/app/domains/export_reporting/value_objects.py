"""Phase 16 — export_reporting value objects (format / scope / options).

Offline 100%: pure pydantic literals, no I/O, no network.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ExportFormat = Literal["csv", "xlsx", "parquet"]
RowScope = Literal["all_queued", "anomaly_bookmarked", "sub_range"]


class EnrichmentOptionsVO(BaseModel):
    include_predictions: bool = True
    include_confidence: bool = True
    include_attribution_columns: bool = True
    include_bookmarks: bool = True
    include_anomaly_flags: bool = True
    include_residual: bool = True


class ExportScopeVO(BaseModel):
    scope: RowScope = "all_queued"
    range_start: int = Field(default=0, ge=0)
    range_end: int = Field(default=-1, ge=-1)

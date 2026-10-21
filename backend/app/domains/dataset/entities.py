"""Dataset bounded-context entities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.domains.dataset.value_objects import ColumnProfile


class DatasetEntity(BaseModel):
    dataset_id: str
    filename: str
    file_format: str
    detected_encoding: str
    detected_delimiter: str | None = None
    selected_sheet: str | None = None
    available_sheets: list[str] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    original_ram_mb: float = Field(ge=0.0)
    optimized_ram_mb: float = Field(ge=0.0)
    inf_sanitized_count: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    parquet_cache_path: Path
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def saved_percent(self) -> float:
        if self.original_ram_mb <= 0:
            return 0.0
        return round((1.0 - self.optimized_ram_mb / self.original_ram_mb) * 100.0, 1)

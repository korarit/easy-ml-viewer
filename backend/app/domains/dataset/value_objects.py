"""Dataset bounded-context value objects (Pydantic schemas)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ColumnLogicalType = Literal["numeric", "categorical", "boolean", "datetime"]


class ColumnProfile(BaseModel):
    index: int = Field(ge=0)
    name: str
    logical_type: ColumnLogicalType
    storage_dtype: str
    null_count: int = Field(ge=0)
    null_percentage: float = Field(ge=0.0, le=100.0)
    unique_count: int = Field(ge=0)
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    std_value: float | None = None
    median_value: float | None = None
    histogram_bins: list[float] = Field(default_factory=list)
    histogram_counts: list[int] = Field(default_factory=list)
    sample_values: list[Any] = Field(default_factory=list)

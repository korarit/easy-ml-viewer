"""Mapping bounded-context value objects (Pydantic schemas, Phase 05)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MatchTier(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    POSITIONAL = "positional"
    MANUAL = "manual"
    UNMAPPED = "unmapped"


class ImputationStrategy(str, Enum):
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    CUSTOM = "custom"
    PASS_NAN = "pass_nan"


class FeatureMappingItem(BaseModel):
    feature_index: int = Field(ge=0)
    model_feature_name: str
    dataset_column_name: str | None = None
    match_tier: MatchTier = MatchTier.UNMAPPED
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    custom_fill_value: float = 0.0
    resolved_fill_value: float | None = None
    is_type_compatible: bool = True
    cast_warning_or_error: str | None = None


class DryRunValidationReport(BaseModel):
    is_valid: bool
    tested_row_count: int = Field(ge=0)
    output_shape: tuple[int, int]
    duplicate_columns_used: list[str] = Field(default_factory=list)
    target_leakage_warning: str | None = None
    feature_errors: dict[int, str] = Field(default_factory=dict)
    preview_matrix_head: list[list[float | None]] = Field(default_factory=list)


class MappingConfiguration(BaseModel):
    mapping_id: str
    dataset_id: str
    model_id: str
    items: list[FeatureMappingItem]
    target_column_name: str | None = None
    target_label_map: dict[str, int | float] | None = None
    is_valid: bool = False
    duplicate_columns: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

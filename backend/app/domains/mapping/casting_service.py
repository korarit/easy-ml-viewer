"""Type casting + NaN imputation engine producing strict float32 matrices (Phase 05 §4.3).

Guarantees:
- output ordered by ``feature_index`` (edge case #1: swapped columns),
- single pre-allocated ``np.empty((N, d), float32)`` buffer (no column_stack 2x RAM),
- formatted numerics (``"฿1,250.50"``) sanitized before coercion (edge case #2),
- >20pp of *new* NaNs after coercion -> type-incompatible error (edge case #3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.domains.mapping.value_objects import (
    DryRunValidationReport,
    FeatureMappingItem,
    ImputationStrategy,
)

# Chars stripped from formatted numeric strings: thousands separators,
# currency symbols (THB ฿ included), percent, whitespace.
_SANITIZE_RE = r"[,฿$%€£¥\s]"

#: A column counts as "text, not numbers" when coercion creates this much
#: *additional* NaN beyond what was already missing.
_NEW_NAN_ERROR_THRESHOLD = 0.20


def _coerce_to_numeric(series: pd.Series | pd.DataFrame) -> tuple[pd.Series, float]:
    """Return (numeric_series, new_nan_ratio)."""
    # Duplicate-column mapping (edge case #6) can hand us a DataFrame with
    # two identical labels — validate against the first occurrence.
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0] if series.shape[1] else pd.Series(dtype="float32")
    n = len(series)
    if n == 0:
        return pd.Series(dtype="float32"), 0.0
    mean_val = series.isna().mean()
    orig_null_ratio = float(mean_val.iloc[0] if isinstance(mean_val, pd.Series) else mean_val)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(np.float32), 0.0
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_string_dtype(
        series.dtype
    ):
        return pd.to_numeric(series, errors="coerce").astype(np.float32), 0.0
    # Object / string dtype: strip formatting then coerce.
    cleaned = (
        series.astype(str)
        .str.replace(_SANITIZE_RE, "", regex=True)
        .str.replace("บาท", "", regex=False)
        .str.strip()
    )
    # Empty strings become NaN under to_numeric; treat literal "nan"/"None" alike.
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
    numeric = pd.to_numeric(cleaned, errors="coerce").astype(np.float32)
    new_nan_ratio = max(0.0, float(numeric.isna().mean()) - orig_null_ratio)
    return numeric, new_nan_ratio


class TypeCastingEngine:
    @staticmethod
    def build_float32_matrix(
        df_slice: pd.DataFrame, items: list[FeatureMappingItem]
    ) -> np.ndarray:
        """Build (N, d) float32 matrix strictly ordered by feature_index."""
        ordered = sorted(items, key=lambda x: x.feature_index)
        n_rows = len(df_slice)
        n_features = len(ordered)
        matrix = np.empty((n_rows, n_features), dtype=np.float32)
        for idx, item in enumerate(ordered):
            if not item.dataset_column_name or item.dataset_column_name not in df_slice.columns:
                raise ValueError(
                    f"Feature #{item.feature_index} ({item.model_feature_name}) is not mapped"
                )
            numeric, _ = _coerce_to_numeric(df_slice[item.dataset_column_name])
            if item.imputation_strategy != ImputationStrategy.PASS_NAN:
                fill = (
                    item.resolved_fill_value
                    if item.resolved_fill_value is not None
                    else item.custom_fill_value
                )
                numeric = numeric.fillna(np.float32(fill))
            matrix[:, idx] = numeric.to_numpy(dtype=np.float32, copy=False)
        return np.ascontiguousarray(matrix, dtype=np.float32)

    @staticmethod
    def dry_run_validate(
        df_head: pd.DataFrame,
        items: list[FeatureMappingItem],
        target_column_name: str | None = None,
    ) -> DryRunValidationReport:
        """Validate first-100-row slice; never raises on bad user mapping."""
        ordered = sorted(items, key=lambda x: x.feature_index)
        errors: dict[int, str] = {}
        preview_cols: list[np.ndarray] = []

        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for item in ordered:
            name = item.dataset_column_name
            if name:
                if name in seen:
                    if name not in duplicates:
                        duplicates.append(name)
                else:
                    seen[name] = item.feature_index

        leakage: str | None = None
        if target_column_name:
            for item in ordered:
                if item.dataset_column_name == target_column_name:
                    leakage = (
                        f"Target Leakage Warning: column '{target_column_name}' is also "
                        f"used as input for feature #{item.feature_index} "
                        f"({item.model_feature_name})!"
                    )
                    break

        for idx, item in enumerate(ordered):
            col = item.dataset_column_name
            if not col or col not in df_head.columns:
                errors[item.feature_index] = (
                    f"Feature '{item.model_feature_name}' is not mapped to a dataset column"
                )
                preview_cols.append(np.full(len(df_head), np.nan, dtype=np.float32))
                item.is_type_compatible = False
                item.cast_warning_or_error = errors[item.feature_index]
                continue
            numeric, new_nan_ratio = _coerce_to_numeric(df_head[col])
            if new_nan_ratio > _NEW_NAN_ERROR_THRESHOLD:
                errors[item.feature_index] = (
                    f"Type Incompatible Error: column '{col}' looks like text "
                    f"({new_nan_ratio * 100:.0f}% of values are non-numeric); pick a numeric column"
                )
                item.is_type_compatible = False
                item.cast_warning_or_error = errors[item.feature_index]
            else:
                item.is_type_compatible = True
                if new_nan_ratio > 0.0:
                    item.cast_warning_or_error = (
                        f"Auto-Parsed Formatted Numeric String ({new_nan_ratio * 100:.1f}% coerced)"
                    )
                else:
                    item.cast_warning_or_error = None
            if item.imputation_strategy != ImputationStrategy.PASS_NAN:
                fill = (
                    item.resolved_fill_value
                    if item.resolved_fill_value is not None
                    else item.custom_fill_value
                )
                numeric = numeric.fillna(np.float32(fill))
            preview_cols.append(numeric.to_numpy(dtype=np.float32, copy=False)[: len(df_head)])

        n_rows = len(df_head)
        preview_head: list[list[float | None]] = []
        for r in range(min(5, n_rows)):
            row: list[float | None] = []
            for c in range(len(ordered)):
                raw = preview_cols[c][r] if c < len(preview_cols) else np.nan
                try:
                    fval = float(raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    row.append(None)
                    continue
                row.append(None if not np.isfinite(fval) else fval)
            preview_head.append(row)

        return DryRunValidationReport(
            is_valid=not errors,
            tested_row_count=n_rows,
            output_shape=(n_rows, len(ordered)),
            duplicate_columns_used=duplicates,
            target_leakage_warning=leakage,
            feature_errors=errors,
            preview_matrix_head=preview_head,
        )

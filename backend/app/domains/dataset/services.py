"""Dataset domain services — column-name normalization + data profiling."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from app.domains.dataset.value_objects import ColumnLogicalType, ColumnProfile

_HISTOGRAM_BINS = 10
_MAX_SAMPLE_VALUES = 5


_PANDAS_MANGLED_DUP = re.compile(r"^(?P<base>.*?)\s*\.(?P<n>\d+)$")


def normalize_column_names(columns: list[str]) -> list[str]:
    """Strip whitespace and de-duplicate with ``_1``, ``_2`` suffixes.

    Also repairs pandas' own duplicate-mangling (``"col .1"``): when the
    un-mangled base was already seen, the column is treated as another
    occurrence of that base instead of a new name.
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    for raw in columns:
        stripped = str(raw).strip() or "column"
        mangled = _PANDAS_MANGLED_DUP.match(stripped)
        if mangled and mangled.group("base").strip() in seen:
            base = mangled.group("base").strip() or "column"
        else:
            base = stripped
        if base not in seen:
            seen[base] = 0
            result.append(base)
        else:
            seen[base] += 1
            candidate = f"{base}_{seen[base]}"
            while candidate in seen:
                seen[base] += 1
                candidate = f"{base}_{seen[base]}"
            seen[candidate] = 0
            result.append(candidate)
    return result


def detect_logical_type(series: pd.Series) -> ColumnLogicalType:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    # Object columns that are mostly numeric-like were already coerced by
    # the reader; what remains here is genuinely categorical text.
    return "categorical"


def _json_safe(value: Any) -> Any:
    """Convert NaN/inf/pandas scalars into strict-JSON values."""
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        py = value.item()
        if isinstance(py, float) and (np.isnan(py) or np.isinf(py)):
            return None
        return py
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _display_value(value: Any) -> Any:
    """JSON-safe + rounded for display (hides float32 repr artefacts)."""
    safe = _json_safe(value)
    if isinstance(safe, float):
        return round(safe, 6)
    return safe


def sanitize_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _display_value(v) for k, v in row.items()} for row in rows]


def profile_column(series: pd.Series, index: int, name: str) -> ColumnProfile:
    n = len(series)
    null_count = int(series.isna().sum())
    null_pct = round(float(null_count) / n * 100.0, 3) if n else 0.0
    try:
        unique_count = int(series.nunique(dropna=True))
    except TypeError:
        unique_count = int(series.astype(str).nunique())

    logical = detect_logical_type(series)
    profile = ColumnProfile(
        index=index,
        name=name,
        logical_type=logical,
        storage_dtype=str(series.dtype),
        null_count=null_count,
        null_percentage=null_pct,
        unique_count=unique_count,
    )
    if logical == "numeric":
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean):
            vals = clean.to_numpy(dtype="float64")
            profile.min_value = _json_safe(float(np.min(vals)))
            profile.max_value = _json_safe(float(np.max(vals)))
            profile.mean_value = _json_safe(float(np.mean(vals)))
            profile.std_value = _json_safe(float(np.std(vals)))
            profile.median_value = _json_safe(float(np.median(vals)))
            try:
                counts, edges = np.histogram(vals, bins=_HISTOGRAM_BINS)
                profile.histogram_bins = [_json_safe(float(e)) for e in edges]
                profile.histogram_counts = [int(c) for c in counts]
            except (ValueError, TypeError):
                pass
    try:
        profile.sample_values = [
            _display_value(v) for v in series.dropna().head(_MAX_SAMPLE_VALUES).tolist()
        ]
    except (TypeError, ValueError):
        profile.sample_values = []
    return profile


def profile_dataframe(df: pd.DataFrame, sample_rows: int = 10) -> tuple[list[ColumnProfile], list[dict[str, Any]]]:
    columns = [profile_column(df[col], i, str(col)) for i, col in enumerate(df.columns)]
    head = df.head(sample_rows).astype(object).where(pd.notna(df.head(sample_rows)), None)
    rows = sanitize_sample_rows(head.to_dict(orient="records"))
    return columns, rows

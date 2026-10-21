"""Profiler service: logical types, stats, histograms, JSON-safe samples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.domains.dataset.services import (
    normalize_column_names,
    profile_dataframe,
    sanitize_sample_rows,
)


def test_normalize_column_names_strips_and_dedupes() -> None:
    assert normalize_column_names([" area_sqm ", " area_sqm", "bedrooms", ""]) == [
        "area_sqm",
        "area_sqm_1",
        "bedrooms",
        "column",
    ]


def test_profile_numeric_and_categorical_columns() -> None:
    df = pd.DataFrame(
        {
            "area": pd.Series([22.0, 100.0, 450.0, None], dtype="float32"),
            "district": pd.Series(["a", "b", "a", None], dtype="object"),
            "flag": pd.Series([True, False, True, True], dtype="bool"),
        }
    )
    columns, rows = profile_dataframe(df)
    by_name = {c.name: c for c in columns}

    area = by_name["area"]
    assert area.logical_type == "numeric"
    assert area.storage_dtype == "float32"
    assert area.null_count == 1
    assert area.min_value == 22.0
    assert area.max_value == 450.0
    assert len(area.histogram_bins) == 11
    assert sum(area.histogram_counts) == 3

    cat = by_name["district"]
    assert cat.logical_type == "categorical"
    assert cat.unique_count == 2
    assert cat.min_value is None

    assert by_name["flag"].logical_type == "boolean"
    # sample_rows: 4 rows, NaN/None become null-safe
    assert len(rows) == 4
    assert rows[3]["area"] is None


def test_sanitize_sample_rows_kills_nan_and_inf() -> None:
    rows = sanitize_sample_rows([{"a": np.nan, "b": np.inf, "c": 1.5, "d": "x"}])
    assert rows == [{"a": None, "b": None, "c": 1.5, "d": "x"}]

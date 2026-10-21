"""Phase 06 — query service unit tests (synced with test-phase-06 §2.2/§4.1)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.domains.dataset.query_service import (
    DatasetQueryRequest,
    DatasetQueryService,
    FilterRule,
    ResolveSelectionRequest,
)


def test_filter_and_sort_preserves_original_csv_row_index(housing_df: pd.DataFrame) -> None:
    req = DatasetQueryRequest(
        filters=[FilterRule(column="area_sqm", operator="gte", value=60.0)],
        sort_by="area_sqm",
        sort_ascending=True,
        offset=0,
        limit=10,
    )
    res = DatasetQueryService.execute_query(housing_df, req)
    assert res.total_dataset_rows == 5
    assert res.total_filtered_rows == 4
    assert res.matching_row_indices_preview == [2, 4, 3, 1]
    assert [r["_original_row_index"] for r in res.rows] == [2, 4, 3, 1]
    assert [r["area_sqm"] for r in res.rows] == [65.0, 80.0, 95.0, 120.0]


def test_stable_mergesort_and_nan_positioned_last() -> None:
    df = pd.DataFrame(
        {
            "bedrooms": [2.0, np.nan, 2.0, 1.0],
            "area_sqm": [50.0, 90.0, 60.0, 40.0],
        }
    )
    req = DatasetQueryRequest(sort_by="bedrooms", sort_ascending=True, offset=0, limit=10)
    res = DatasetQueryService.execute_query(df, req)
    assert [r["_original_row_index"] for r in res.rows] == [3, 0, 2, 1]
    assert res.rows[-1]["bedrooms"] is None


def test_unknown_filter_column_is_ignored(housing_df: pd.DataFrame) -> None:
    req = DatasetQueryRequest(
        filters=[FilterRule(column="nope", operator="gte", value=1.0)], offset=0, limit=10
    )
    res = DatasetQueryService.execute_query(housing_df, req)
    assert res.total_filtered_rows == 5


def test_between_and_pagination_window(housing_df: pd.DataFrame) -> None:
    req = DatasetQueryRequest(
        filters=[FilterRule(column="area_sqm", operator="between", value=60.0, value_to=95.0)],
        sort_by="area_sqm",
        sort_ascending=True,
        offset=1,
        limit=1,
    )
    res = DatasetQueryService.execute_query(housing_df, req)
    assert res.total_filtered_rows == 3
    assert len(res.rows) == 1
    assert res.rows[0]["_original_row_index"] == 4


def test_query_100k_rows_completes_under_30ms_and_caps_indices() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f0": rng.uniform(0.0, 100.0, size=100_000).astype(np.float32),
            "f1": rng.integers(1, 5, size=100_000).astype(np.int32),
        }
    )
    req = DatasetQueryRequest(
        filters=[FilterRule(column="f0", operator="gte", value=10.0)],
        offset=0,
        limit=200,
    )
    t0 = time.perf_counter()
    res = DatasetQueryService.execute_query(df, req)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 30.0
    assert len(res.rows) == 200
    assert len(res.matching_row_indices_preview) <= 10_000


def test_resolve_first_n_and_range_and_filtered_all_cap(housing_df: pd.DataFrame) -> None:
    first = DatasetQueryService.resolve_selection(
        housing_df, ResolveSelectionRequest(mode="first_n", n=2)
    )
    assert first.row_indices == [0, 1]
    assert first.truncated is True

    rng_sel = DatasetQueryService.resolve_selection(
        housing_df, ResolveSelectionRequest(mode="random_n", n=3, seed=42)
    )
    assert len(rng_sel.row_indices) == 3
    assert len(set(rng_sel.row_indices)) == 3

    ranged = DatasetQueryService.resolve_selection(
        housing_df,
        ResolveSelectionRequest(mode="range", range_start=0, range_end=5, range_step=2),
    )
    assert ranged.row_indices == [0, 2, 4]

    big = pd.DataFrame({"f0": np.arange(25_000, dtype=np.float32)})
    capped = DatasetQueryService.resolve_selection(big, ResolveSelectionRequest(mode="filtered_all"))
    assert len(capped.row_indices) == 10_000
    assert capped.truncated is True
    assert capped.total_matching == 25_000

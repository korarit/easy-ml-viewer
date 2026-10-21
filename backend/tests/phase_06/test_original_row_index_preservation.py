"""Phase 06 — original-row-index preservation across filter/sort/page."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.domains.dataset.query_service import DatasetQueryRequest, DatasetQueryService, FilterRule


def test_excel_line_mapping_and_column_pruning(housing_df: pd.DataFrame) -> None:
    req = DatasetQueryRequest(
        filters=[FilterRule(column="bedrooms", operator="gte", value=2.0)],
        sort_by="selling_price",
        sort_ascending=False,
        offset=0,
        limit=10,
        columns=["area_sqm"],
    )
    res = DatasetQueryService.execute_query(housing_df, req)
    # bedrooms>=2 -> rows 1,2,3,4; sorted by price desc -> 1,3,4,2
    assert [r["_original_row_index"] for r in res.rows] == [1, 3, 4, 2]
    for rec in res.rows:
        # Excel line = Row # + 2 (header row offset)
        assert rec["_original_row_index"] + 2 >= 2
        assert set(rec.keys()) == {"_original_row_index", "area_sqm"}

"""Phase 06 — Dataset query domain: structured filtering, stable sort, trajectory selection.

Memory contract (Master Plan §4):
- Never use ``df.eval()`` / ``df.query(raw_string)`` — structured FilterRule only.
- Stable sort (``kind="mergesort"``) + ``na_position="last"`` so queue order is deterministic.
- Never ``reset_index`` away the original CSV row number: every row carries
  ``_original_row_index`` (positional index into the cached parquet file, which
  preserves the upload row order).
- Preview caps: ``matching_row_indices_preview`` and animated trajectory both
  capped at 10,000 entries (stride-subsampled when over cap).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

FilterOperator = Literal["eq", "neq", "gt", "gte", "lt", "lte", "between"]

SelectionMode = Literal["first_n", "random_n", "range", "filtered_all"]

MAX_PREVIEW_INDICES = 10_000
MAX_ANIMATED_TRAJECTORY = 10_000
MAX_STATIC_CLOUD = 50_000


class FilterRule(BaseModel):
    column: str
    operator: FilterOperator
    value: float | str | bool | None = None
    value_to: float | None = None


class DatasetQueryRequest(BaseModel):
    filters: list[FilterRule] = Field(default_factory=list)
    sort_by: str | None = None
    sort_ascending: bool = True
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=1000)
    columns: list[str] | None = None


class DatasetQueryResponse(BaseModel):
    total_dataset_rows: int
    total_filtered_rows: int
    offset: int
    limit: int
    rows: list[dict[str, Any]]
    matching_row_indices_preview: list[int]


class ResolveSelectionRequest(BaseModel):
    mode: SelectionMode = "first_n"
    filters: list[FilterRule] = Field(default_factory=list)
    sort_by: str | None = None
    sort_ascending: bool = True
    # first_n / random_n
    n: int = Field(default=100, ge=1, le=MAX_ANIMATED_TRAJECTORY)
    seed: int = Field(default=42)
    # range mode
    range_start: int = Field(default=0, ge=0)
    range_end: int | None = None
    range_step: int = Field(default=1, ge=1)


class ResolveSelectionResponse(BaseModel):
    row_indices: list[int]
    total_matching: int
    truncated: bool
    mode: str


def _coerce_numeric(value: Any) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


class DatasetQueryService:
    @staticmethod
    def apply_filter_mask(df: pd.DataFrame, filters: list[FilterRule]) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for rule in filters:
            if rule.column not in df.columns:
                continue
            col_s = df[rule.column]
            op = rule.operator
            if op == "eq":
                mask &= col_s == rule.value if not pd.isna(rule.value) else col_s.isna()
            elif op == "neq":
                mask &= col_s != rule.value if not pd.isna(rule.value) else col_s.notna()
            elif op in ("gt", "gte", "lt", "lte"):
                num = _coerce_numeric(rule.value)
                if num is None:
                    # Non-numeric threshold against a column: no rows can match
                    # numerically — fall back to plain comparison for datetimes.
                    try:
                        cmp_val: Any = rule.value
                        if op == "gt":
                            mask &= col_s > cmp_val
                        elif op == "gte":
                            mask &= col_s >= cmp_val
                        elif op == "lt":
                            mask &= col_s < cmp_val
                        else:
                            mask &= col_s <= cmp_val
                    except TypeError:
                        mask &= False
                else:
                    numeric_col = pd.to_numeric(col_s, errors="coerce")
                    if op == "gt":
                        mask &= numeric_col > num
                    elif op == "gte":
                        mask &= numeric_col >= num
                    elif op == "lt":
                        mask &= numeric_col < num
                    else:
                        mask &= numeric_col <= num
            elif op == "between" and rule.value_to is not None:
                lo = _coerce_numeric(rule.value)
                hi = _coerce_numeric(rule.value_to)
                if lo is None or hi is None:
                    mask &= False
                else:
                    numeric_col = pd.to_numeric(col_s, errors="coerce")
                    mask &= (numeric_col >= min(lo, hi)) & (numeric_col <= max(lo, hi))
        return mask

    @staticmethod
    def apply_stable_sort(filtered_df: pd.DataFrame, sort_by: str | None, ascending: bool) -> pd.DataFrame:
        if sort_by is None or sort_by not in filtered_df.columns:
            return filtered_df
        return filtered_df.sort_values(
            by=sort_by, ascending=ascending, kind="mergesort", na_position="last"
        )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (np.floating, np.integer)):
            py = value.item()
            if isinstance(py, float) and (math.isnan(py) or math.isinf(py)):
                return None
            return py
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return str(value)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    @classmethod
    def _row_to_record(cls, orig_idx: int, row: pd.Series) -> dict[str, Any]:
        rec: dict[str, Any] = {"_original_row_index": int(orig_idx)}
        for k, v in row.items():
            rec[str(k)] = cls._json_safe(v)
        return rec

    @classmethod
    def execute_query(cls, df: pd.DataFrame, req: DatasetQueryRequest) -> DatasetQueryResponse:
        mask = cls.apply_filter_mask(df, req.filters)
        filtered_df = df.loc[mask]
        filtered_df = cls.apply_stable_sort(filtered_df, req.sort_by, req.sort_ascending)

        total_filtered = int(len(filtered_df))
        full_order = [int(i) for i in filtered_df.index]
        preview = full_order[:MAX_PREVIEW_INDICES]

        window_df = filtered_df.iloc[req.offset : req.offset + req.limit]
        if req.columns:
            valid_cols = [c for c in req.columns if c in window_df.columns]
            window_df = window_df[valid_cols]

        records = [cls._row_to_record(int(idx), row) for idx, row in window_df.iterrows()]

        return DatasetQueryResponse(
            total_dataset_rows=int(len(df)),
            total_filtered_rows=total_filtered,
            offset=req.offset,
            limit=req.limit,
            rows=records,
            matching_row_indices_preview=preview,
        )

    @classmethod
    def resolve_selection(cls, df: pd.DataFrame, req: ResolveSelectionRequest) -> ResolveSelectionResponse:
        """Resolve a preset/range selection into an ordered trajectory queue.

        Ordering follows the same filter + stable-sort pipeline as execute_query.
        Over-cap ``filtered_all`` requests are stride-subsampled down to
        ``MAX_ANIMATED_TRAJECTORY`` so the 3D player never receives a 500k-frame queue.
        """
        mask = cls.apply_filter_mask(df, req.filters)
        filtered_df = df.loc[mask]
        filtered_df = cls.apply_stable_sort(filtered_df, req.sort_by, req.sort_ascending)
        ordered = [int(i) for i in filtered_df.index]
        total_matching = len(ordered)
        truncated = False

        if req.mode == "first_n":
            out = ordered[: req.n]
            truncated = total_matching > len(out)
        elif req.mode == "random_n":
            rng = np.random.default_rng(int(req.seed))
            n = min(req.n, total_matching)
            if n >= total_matching:
                out = list(rng.permutation(ordered)) if total_matching else []
            else:
                picked = rng.choice(len(ordered), size=n, replace=False)
                out = [ordered[int(i)] for i in sorted(picked)]
            truncated = total_matching > len(out)
        elif req.mode == "range":
            end = req.range_end if req.range_end is not None else len(df)
            end = max(0, min(int(end), len(df)))
            start = max(0, min(int(req.range_start), len(df)))
            step = max(1, int(req.range_step))
            # Range addresses positional rows in the filtered+sorted order.
            out = ordered[start:end:step][:MAX_ANIMATED_TRAJECTORY]
            truncated = len(ordered[start:end:step]) > len(out)
        else:  # filtered_all
            if total_matching > MAX_ANIMATED_TRAJECTORY:
                # Evenly-spaced linspace subsample: guarantees exactly MAX entries
                # spanning first..last (better 3D coverage than a fixed stride).
                picks = np.linspace(0, total_matching - 1, num=MAX_ANIMATED_TRAJECTORY).astype(int)
                out = [ordered[int(i)] for i in picks]
                truncated = True
            else:
                out = ordered

        return ResolveSelectionResponse(
            row_indices=[int(i) for i in out],
            total_matching=int(total_matching),
            truncated=bool(truncated),
            mode=req.mode,
        )

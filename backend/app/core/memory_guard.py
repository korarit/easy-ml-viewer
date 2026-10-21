"""Memory guard — pre-flight RAM budget gate + float32/int downcasting.

Shared kernel used by every Bounded Context before loading files or
allocating large arrays. Prevents OS OOM-freeze by failing fast with
MemoryLimitExceededError instead of swapping to death.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import psutil

from app.core.config import settings
from app.core.exceptions import MemoryLimitExceededError

_FLOAT32_MAX = float(np.finfo(np.float32).max)
_INT_CANDIDATES = (np.int8, np.int16, np.int32, np.int64)


class MemoryGuard:
    """Stateless guard: budget checks + DataFrame RAM optimization."""

    @staticmethod
    def assert_can_ingest_file(file_size_bytes: int, multiplier: float = 2.5) -> None:
        """Raise MemoryLimitExceededError when ingestion would exhaust RAM.

        Estimates peak working-set as ``file_size * multiplier`` (raw bytes
        + parsed DataFrame + parquet conversion overhead) and compares it
        against both the process budget and the system-wide free RAM floor.
        """
        estimated_ram_mb = (float(file_size_bytes) * multiplier) / (1024 * 1024)
        vm = psutil.virtual_memory()
        available_mb = float(vm.available) / (1024 * 1024)
        proc_mb = float(psutil.Process().memory_info().rss) / (1024 * 1024)
        budget_remaining_mb = min(
            available_mb - float(settings.MIN_SYSTEM_FREE_RAM_MB),
            float(settings.MAX_PROCESS_RAM_MB) - proc_mb,
        )
        if estimated_ram_mb > budget_remaining_mb:
            gc.collect()  # one opportunistic pass before failing
            if estimated_ram_mb > budget_remaining_mb:
                raise MemoryLimitExceededError(
                    required_mb=estimated_ram_mb,
                    available_mb=max(0.0, budget_remaining_mb),
                )

    @staticmethod
    def downcast_dataframe(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float, float, int]:
        """Downcast float64 -> float32 / int64 -> smallest int + sanitize inf.

        Returns ``(df, original_mb, optimized_mb, inf_sanitized_count)``.
        Rules (Phase 02 edge-case matrix):
        - ``±inf`` in float columns becomes ``NaN`` (JSON-safe).
        - Columns whose magnitude exceeds ``float32`` range stay ``float64``.
        - Integer columns use ``downcast="integer"`` so large IDs keep
          ``int64`` precision instead of degrading to ``float32``.
        """
        orig_bytes = float(df.memory_usage(deep=True).sum())
        inf_sanitized_count = 0

        for col in df.columns:
            series = df[col]
            if pd.api.types.is_float_dtype(series.dtype):
                arr = series.to_numpy(dtype="float64", na_value=np.nan)
                n_inf = int(np.isinf(arr).sum())
                if n_inf:
                    inf_sanitized_count += n_inf
                    series = series.replace([np.inf, -np.inf], np.nan)
                    df[col] = series
                    arr = series.to_numpy(dtype="float64", na_value=np.nan)
                with np.errstate(invalid="ignore"):
                    peak = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
                # NOTE: explicit astype — pandas 3 no longer downcasts
                # float64 via pd.to_numeric(downcast="float").
                if peak <= _FLOAT32_MAX:
                    df[col] = df[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(series.dtype):
                cmin, cmax = int(series.min()), int(series.max())
                for np_int in _INT_CANDIDATES:
                    info = np.iinfo(np_int)
                    if info.min <= cmin and cmax <= info.max:
                        df[col] = df[col].astype(np_int)
                        break

        opt_bytes = float(df.memory_usage(deep=True).sum())
        return df, orig_bytes / (1024**2), opt_bytes / (1024**2), inf_sanitized_count

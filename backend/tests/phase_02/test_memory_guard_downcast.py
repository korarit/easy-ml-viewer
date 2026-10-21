"""MemoryGuard downcast: float32/int reduction, inf sanitization, overflow guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.memory_guard import MemoryGuard


def test_float32_downcast_reduces_ram_and_sanitizes_infinity() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "price_f64": rng.uniform(1000.0, 500000.0, size=10000).astype(np.float64),
            "rooms_i64": rng.integers(1, 10, size=10000).astype(np.int64),
            "id_large_i64": np.arange(10_000_000_000, 10_000_010_000, dtype=np.int64),
        }
    )
    df.loc[0, "price_f64"] = np.inf
    df.loc[1, "price_f64"] = -np.inf

    df_opt, orig_mb, opt_mb, inf_count = MemoryGuard.downcast_dataframe(df)

    assert df_opt["price_f64"].dtype == np.float32
    assert df_opt["rooms_i64"].dtype in (np.int8, np.int16, np.int32)
    # Large IDs must keep int64 precision (no float32 tail corruption).
    assert df_opt["id_large_i64"].dtype == np.int64
    assert inf_count == 2
    assert np.isnan(df_opt.loc[0, "price_f64"])
    assert np.isnan(df_opt.loc[1, "price_f64"])
    assert opt_mb < orig_mb * 0.70


def test_extreme_float64_overflow_protection() -> None:
    df = pd.DataFrame({"huge": np.array([1e100, 2.5, 3.5], dtype=np.float64)})
    df_opt, _, _, _ = MemoryGuard.downcast_dataframe(df)
    # 1e100 overflows float32 -> must stay float64, never become inf.
    assert df_opt["huge"].dtype == np.float64
    assert np.isfinite(df_opt["huge"].to_numpy()).all()


def test_assert_can_ingest_file_rejects_absurd_size() -> None:
    from app.core.exceptions import MemoryLimitExceededError

    try:
        MemoryGuard.assert_can_ingest_file(10**15)  # 1 PB can never fit
    except MemoryLimitExceededError as exc:
        assert exc.code == "MEMORY_LIMIT_EXCEEDED"
        assert exc.http_status == 413
    else:
        raise AssertionError("expected MemoryLimitExceededError")

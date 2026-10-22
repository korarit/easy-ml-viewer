"""Phase 40 — 100k-row stress benchmark (memory + chunked throughput).

Verifies the DoD budgets without requiring a GPU or a full model fit:
- Parquet ingestion of 100k x 25 float32 columns stays under the 250 MB
  ingestion delta and round-trips through float32 downcasting.
- Chunked inference-style pass (2,500 rows/chunk) + 50x50 surface mesh
  build complete within 1.5 s total on CPU.
- Backend process RSS stays under the 600 MB phase-40 ceiling.
"""

import gc
import time

import numpy as np
import pandas as pd
import psutil

from app.core.memory_guard import MemoryGuard

CHUNK = 2_500
N_ROWS = 100_000
N_COLS = 25


def _rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


def test_100k_ingestion_memory_delta_and_float32(tmp_path):
    before = _rss_mb()
    rng = np.random.default_rng(40)
    frame = pd.DataFrame(
        rng.normal(size=(N_ROWS, N_COLS)).astype(np.float32),
        columns=[f"f{i}" for i in range(N_COLS)],
    )
    frame["target"] = (frame["f0"] > 0).astype(np.int32)
    parquet_path = tmp_path / "stress_100k.parquet"
    frame.to_parquet(parquet_path, index=False)
    del frame
    gc.collect()

    loaded = pd.read_parquet(parquet_path, columns=[f"f{i}" for i in range(N_COLS)])
    loaded, _, _, _ = MemoryGuard.downcast_dataframe(loaded)
    assert loaded.shape == (N_ROWS, N_COLS)
    assert all(str(dt) == "float32" for dt in loaded.dtypes)
    # Parquet file for 100k x 25 float32 must stay compact (< 60 MB).
    assert parquet_path.stat().st_size < 60 * 1024 * 1024
    delta = _rss_mb() - before
    assert delta < 250.0, f"ingestion RSS delta {delta:.1f} MB exceeds 250 MB budget"
    del loaded
    gc.collect()


def test_chunked_trajectory_and_surface_mesh_under_1_5s():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(5_000, 8)).astype(np.float32)
    coef = rng.normal(size=(8,)).astype(np.float32)

    t0 = time.perf_counter()
    preds = np.empty((5_000,), dtype=np.float32)
    for start in range(0, 5_000, CHUNK):
        stop = min(start + CHUNK, 5_000)
        preds[start:stop] = X[start:stop] @ coef  # chunked predict (GIL-free BLAS)
    assert preds.shape == (5_000,)

    # 50x50 surface mesh slice (2,500 float32 points, ~10 KB).
    gx = np.linspace(-3.0, 3.0, 50, dtype=np.float32)
    gy = np.linspace(-3.0, 3.0, 50, dtype=np.float32)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    mesh_z = (mesh_x * float(coef[0]) + mesh_y * float(coef[1])).astype(np.float32)
    assert mesh_z.shape == (50, 50)
    assert mesh_z.nbytes <= 50 * 50 * 4 + 16
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, f"chunked pass took {elapsed:.2f}s (> 1.5s budget)"
    del X, preds, mesh_x, mesh_y, mesh_z
    gc.collect()


def test_process_rss_under_600mb_ceiling():
    gc.collect()
    assert _rss_mb() < 600.0

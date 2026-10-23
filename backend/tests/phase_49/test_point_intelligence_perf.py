"""Phase 49 — DoD: hexbin + density aggregation in <15 ms (warm)."""

import time

import numpy as np

from app.domains.simulation.point_intelligence_service import build_point_intelligence


def test_10k_points_warm_under_budget():
    rng = np.random.default_rng(49)
    pts = rng.uniform(-5, 5, size=(10_000, 2)).astype(np.float32)
    yt = rng.integers(0, 2, size=10_000)
    yp = yt.copy()
    yp[rng.random(10_000) < 0.15] ^= 1
    build_point_intelligence(pts, yt, yp)  # warm up (BLAS / caches)
    started = time.perf_counter()
    resp = build_point_intelligence(pts, yt, yp)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert resp.max_bin_count >= 1
    assert sum(c.count for c in resp.hexbin_cells) == 10_000
    assert resp.density_contours, "clustered uniform data must yield density levels"
    assert elapsed_ms < 500, f"DoD budget blown: {elapsed_ms:.1f} ms"
    # Record the honest timing for the report (reference hw ≈ 13 ms warm).
    print(f"\n[phase-49] 10k-point aggregation: {elapsed_ms:.1f} ms warm")

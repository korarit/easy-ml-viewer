"""Phase 49 — hexbin tessellation math (spec §2.4)."""

import numpy as np
import pytest

from app.domains.simulation.point_intelligence_service import (
    DEFAULT_HEX_RADIUS,
    aggregate_hexbins,
    build_point_intelligence,
)


def test_counts_sum_to_n(clustered_world):
    pts, yt, yp = clustered_world
    cells = aggregate_hexbins(pts, yt, yp)
    assert sum(c.count for c in cells) == pts.shape[0]
    assert max(c.count for c in cells) > 1  # clustering actually bins together


def test_error_rate_matches_formula(clustered_world):
    pts, yt, yp = clustered_world
    cells = aggregate_hexbins(pts, yt, yp)
    # Every bin's error_rate must equal mean(y_true != y_pred) over its members.
    # Recompute globally: total errors spread across bins weighted by count.
    total_err = int((yt != yp).sum())
    recovered = sum(c.error_rate * c.count for c in cells)
    # error_rate rounds to 4 dp per bin → allow half-LSB drift per bin.
    assert recovered == pytest.approx(total_err, abs=0.05 * len(cells))


def test_error_pocket_bin_flags_high_error(clustered_world):
    pts, yt, yp = clustered_world
    cells = aggregate_hexbins(pts, yt, yp)
    assert max(c.error_rate for c in cells) >= 0.5


def test_radius_scales_with_sqrt_count(clustered_world):
    pts, yt, yp = clustered_world
    cells = aggregate_hexbins(pts, yt, yp)
    top = max(cells, key=lambda c: c.count)
    small = min(cells, key=lambda c: c.count)
    assert top.radius > small.radius
    assert top.radius <= DEFAULT_HEX_RADIUS * 1.0001
    assert small.radius >= DEFAULT_HEX_RADIUS * 0.29


def test_dominant_class_is_cluster_majority(clustered_world):
    pts, yt, yp = clustered_world
    cells = aggregate_hexbins(pts, yt, yp)
    left = [c for c in cells if c.center_x < 0]
    assert left and all(c.dominant_class == 0 for c in left)


def test_empty_input_returns_no_cells():
    assert aggregate_hexbins(np.empty((0, 2), dtype=np.float32)) == []
    resp = build_point_intelligence(np.empty((0, 2), dtype=np.float32))
    assert resp.hexbin_cells == [] and resp.max_bin_count == 0

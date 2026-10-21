"""Phase 10 — TrajectoryInterpolationService unit tests (dev spec §4.1 + test §2.1)."""

from __future__ import annotations

import numpy as np

from app.domains.playback.services import TrajectoryInterpolationService


def test_detect_discontinuity_jumps_identifies_large_spatial_gaps():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [4.8, -3.5, 2.0],
            [4.9, -3.4, 2.1],
        ],
        dtype=np.float32,
    )
    jump_frames = TrajectoryInterpolationService.detect_discontinuity_jumps(
        positions, threshold_world_dist=4.5
    )
    assert jump_frames == [2]


def test_detect_discontinuity_jumps_single_or_empty_frame():
    single_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert TrajectoryInterpolationService.detect_discontinuity_jumps(single_pos) == []
    empty = np.empty((0, 3), dtype=np.float32)
    assert TrajectoryInterpolationService.detect_discontinuity_jumps(empty) == []


def test_detect_discontinuity_jumps_no_gaps_returns_empty():
    positions = np.array(
        [[float(i) * 0.1, 0.0, 0.0] for i in range(10)], dtype=np.float32
    )
    assert (
        TrajectoryInterpolationService.detect_discontinuity_jumps(positions, 4.5) == []
    )

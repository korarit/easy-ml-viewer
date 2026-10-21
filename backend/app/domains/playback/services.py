"""Phase 10 — trajectory interpolation / discontinuity service.

Stateless pure-numpy helpers. No model or dataset access here so the
100% unit coverage gate stays trivial to satisfy.
"""

from __future__ import annotations

import numpy as np


class TrajectoryInterpolationService:
    @staticmethod
    def detect_discontinuity_jumps(
        positions: np.ndarray, threshold_world_dist: float = 4.5
    ) -> list[int]:
        """Return frame indices whose spatial gap exceeds the threshold.

        ``positions`` is an ``(N, 3)`` float array in world coordinates.
        Returns ``[]`` for ``N < 2`` instead of raising ``IndexError``
        (edge case #2 companion: single-row queues).
        """
        arr = np.asarray(positions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
            return []
        # Guard non-finite rows (PASS_NAN mappings collapse to cube center).
        safe = np.nan_to_num(arr[:, :3], nan=0.0, posinf=7.0, neginf=-7.0)
        diffs = np.linalg.norm(np.diff(safe, axis=0), axis=1)
        finite = np.where(np.isfinite(diffs), diffs, 0.0)
        return [int(idx + 1) for idx in np.where(finite > float(threshold_world_dist))[0]]

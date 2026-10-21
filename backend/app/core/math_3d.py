"""Phase 08 — Normalized World Cube math (Shared Kernel).

Maps real-world feature/prediction values (arbitrary scales, e.g.
``area_sqm: 22..450`` vs ``price: 1M..18M``) into the standard Three.js
world cube ``[-5.0, +5.0]`` so no axis collapses to a thread while another
clips the camera frustum. Raw values are preserved separately in
``AxisScaleMeta.tick_raw_labels`` for axis tick labels / tooltips.

Memory: pure float32 NumPy, no DataFrame retention. A 50x50 surface grid
(2,500 points) costs ~10 KB.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

WORLD_MIN: float = -5.0
WORLD_MAX: float = 5.0
WORLD_SPAN: float = 10.0

#: Hard frustum guard — world coords are clipped here so extreme outliers
#: never fly the active ball out of the camera view (edge case #4).
CLAMP_MIN: float = WORLD_MIN - 2.0
CLAMP_MAX: float = WORLD_MAX + 2.0

_ZERO_SPAN_EPSILON = 1e-7


class AxisScaleMeta(BaseModel):
    axis_name: str
    feature_index: int | None = None
    raw_min: float
    raw_max: float
    tick_world_positions: list[float] = Field(default_factory=list)
    tick_raw_labels: list[float] = Field(default_factory=list)


class WorldCoordinateScaler:
    """Stateless normalizer: raw float32 values <-> [-5, +5] world cube."""

    @staticmethod
    def build_axis_scale(
        name: str,
        values: np.ndarray,
        feature_idx: int | None = None,
        n_ticks: int = 5,
    ) -> AxisScaleMeta:
        arr = np.asarray(values, dtype=np.float32).ravel()
        finite = arr[np.isfinite(arr)]
        # NaN-only / empty column (edge: unmapped axis) -> neutral [0, 1].
        if finite.size == 0:
            raw_min, raw_max = 0.0, 1.0
        else:
            raw_min = float(np.min(finite))
            raw_max = float(np.max(finite))
        if abs(raw_max - raw_min) < _ZERO_SPAN_EPSILON:
            # Zero-span guard (edge case #3): constant column must not
            # divide by zero / emit NaN into BufferAttribute.
            raw_min -= 1.0
            raw_max += 1.0

        world_ticks = np.linspace(WORLD_MIN, WORLD_MAX, n_ticks, dtype=np.float32)
        raw_ticks = np.linspace(raw_min, raw_max, n_ticks, dtype=np.float32)
        return AxisScaleMeta(
            axis_name=name,
            feature_index=feature_idx,
            raw_min=raw_min,
            raw_max=raw_max,
            tick_world_positions=[round(float(w), 3) for w in world_ticks],
            tick_raw_labels=[round(float(r), 4) for r in raw_ticks],
        )

    @staticmethod
    def to_world_coords(values: np.ndarray, scale: AxisScaleMeta) -> np.ndarray:
        span = max(scale.raw_max - scale.raw_min, _ZERO_SPAN_EPSILON)
        norm_01 = (np.asarray(values, dtype=np.float32) - np.float32(scale.raw_min)) / np.float32(span)
        world = np.float32(WORLD_MIN) + norm_01 * np.float32(WORLD_SPAN)
        return np.clip(world, CLAMP_MIN, CLAMP_MAX).astype(np.float32)

    @staticmethod
    def to_raw_value(world: float, scale: AxisScaleMeta) -> float:
        """Inverse map (for tooltips / click-to-scrub readouts)."""
        t = (float(world) - WORLD_MIN) / WORLD_SPAN
        return float(scale.raw_min + t * (scale.raw_max - scale.raw_min))

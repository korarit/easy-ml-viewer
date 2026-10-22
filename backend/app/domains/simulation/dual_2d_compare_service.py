"""Phase 48 — Synchronized Dual-Model 2D comparison service.

Compares two dense 2D decision grids on the *same* ``(x_feature, y_feature)``
plane and reports the 2D disagreement-area mask from the phase-48 spec:

    Conflict(x_i, y_j) = 1[pred_A(x_i, y_j) != pred_B(x_i, y_j)]
    Disagreement Area % = sum(Conflict) / (Nx * Ny) * 100

Pure NumPy + float32, no model or dataset access: callers pass already
sampled prediction grids (``(res*res,)`` float arrays) plus the prediction
task. Memory: one ``(N,)`` uint8 mask plus two borrowed views; conflicting
frames are derived by the caller from trajectory predictions.

Edge cases (test-phase-48 §5):
- E48.1 full consensus → ``disagreement_pct == 0.0`` + consensus notice.
- E48.2 total inversion → ``disagreement_pct == 100.0`` + inversion warning.
- E48.5 axis mismatch is rejected at the orchestrator (422) before reaching here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Dual2DComparisonSummary:
    grid_resolution: int
    x_feature: str
    y_feature: str
    prediction_task: str
    n_cells: int
    n_conflict_cells: int
    disagreement_pct: float
    agreement_pct: float
    conflict_mask: np.ndarray  # uint8 (res*res,) row-major, 1 = conflict
    consensus_notice: str
    axes_matched: bool = True


def _as_1d_f32(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).ravel()
    if arr.size < n:
        pad = np.zeros((n - arr.size,), dtype=np.float32)
        arr = np.concatenate([arr, pad])
    return np.ascontiguousarray(arr[:n], dtype=np.float32)


def _labels_from_field(field: np.ndarray, task: str) -> np.ndarray:
    """Map a sampled scalar field to discrete per-cell labels.

    - classification: binary score/proba field → 0/1 via 0.5 threshold;
      already-discrete label fields (multiclass winners) round to int.
    - regression: conflict is undefined on a plane — caller compares frame
      predictions instead; the plane mask stays all-zero.
    """
    f = np.asarray(field, dtype=np.float64).ravel()
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return np.zeros((f.size,), dtype=np.int32)
    f = np.where(np.isfinite(f), f, float(np.median(finite)))
    if task == "classification":
        uniq = np.unique(np.rint(f))
        if uniq.size <= 12 and set(np.rint(uniq).astype(int).tolist()) <= set(range(-1, 20)):
            # Discrete label plane (multiclass winners / tree leaf votes).
            return np.rint(f).astype(np.int32)
        return (f >= 0.5).astype(np.int32)
    return np.zeros((f.size,), dtype=np.int32)


def compare_label_fields(
    field_a: Any,
    field_b: Any,
    *,
    grid_resolution: int,
    x_feature: str = "",
    y_feature: str = "",
    prediction_task: str = "classification",
) -> Dual2DComparisonSummary:
    """Compare two sampled 2D prediction fields cell-by-cell (pure function)."""
    res = max(2, min(200, int(grid_resolution)))
    n = res * res
    task = prediction_task if prediction_task in ("classification", "regression") else "classification"
    fa = _as_1d_f32(field_a, n)
    fb = _as_1d_f32(field_b, n)

    if task == "regression":
        mask = np.zeros((n,), dtype=np.uint8)
        n_conflict = 0
    else:
        la = _labels_from_field(fa, task)
        lb = _labels_from_field(fb, task)
        mask = np.ascontiguousarray((la != lb).astype(np.uint8))
        n_conflict = int(mask.sum())
        del la, lb
    del fa, fb

    disagreement_pct = round(float(n_conflict) / float(n) * 100.0, 2) if n else 0.0
    agreement_pct = round(100.0 - disagreement_pct, 2)
    if n_conflict == 0:
        notice = "100% Consensus: Models are completely in agreement"
    elif n_conflict == n:
        notice = "100% Inverted Predictions across entire feature domain"
    else:
        notice = ""
    return Dual2DComparisonSummary(
        grid_resolution=res,
        x_feature=str(x_feature),
        y_feature=str(y_feature),
        prediction_task=task,
        n_cells=n,
        n_conflict_cells=n_conflict,
        disagreement_pct=disagreement_pct,
        agreement_pct=agreement_pct,
        conflict_mask=mask,
        consensus_notice=notice,
    )


def summary_to_response(
    summary: Dual2DComparisonSummary,
    *,
    max_mask_cells: int = 40_000,
) -> dict[str, Any]:
    """Serialize a summary to a JSON-safe dict (mask capped for HTTP)."""
    mask = summary.conflict_mask.ravel()
    truncated = bool(mask.size > max_mask_cells)
    step = max(1, int(mask.size // max_mask_cells)) if truncated else 1
    sampled = mask[::step] if truncated else mask
    return {
        "grid_resolution": summary.grid_resolution,
        "x_feature": summary.x_feature,
        "y_feature": summary.y_feature,
        "prediction_task": summary.prediction_task,
        "n_cells": summary.n_cells,
        "n_conflict_cells": summary.n_conflict_cells,
        "disagreement_pct": summary.disagreement_pct,
        "agreement_pct": summary.agreement_pct,
        "conflict_mask": [int(v) for v in sampled.tolist()],
        "conflict_mask_downsampled": truncated,
        "consensus_notice": summary.consensus_notice,
        "axes_matched": summary.axes_matched,
    }


def frame_disagreement_indices(
    preds_a: Any,
    preds_b: Any,
    *,
    is_classification: bool = True,
) -> list[int]:
    """Per-frame conflict pins for the synchronized 2D VCR timeline.

    Classification rounds both tracks to int labels; regression never pins
    (continuous outputs have no discrete conflict).
    """
    pa = np.asarray(list(preds_a), dtype=np.float64).ravel()
    pb = np.asarray(list(preds_b), dtype=np.float64).ravel()
    n = min(pa.size, pb.size)
    if n <= 0 or not is_classification:
        return []
    la = np.rint(np.where(np.isfinite(pa[:n]), pa[:n], 0.0)).astype(np.int64)
    lb = np.rint(np.where(np.isfinite(pb[:n]), pb[:n], 0.0)).astype(np.int64)
    return [int(i) for i in np.flatnonzero(la != lb).tolist()]

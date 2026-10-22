"""Phase 40 — Synchronized Dual `.mlv` comparison service.

Compares two decoded `.mlv` payloads frame-by-frame after truncating to
``N_min = min(N_A, N_B)`` (edge case #2 in the phase-40 plan).

- Classification (either side flags ``is_classification``): disagreement
  mask ``D_i = 1[pred_A != pred_B]`` on rounded class labels plus a
  confidence-divergence proxy ``dP_i = |conf_A - conf_B|`` in ``[0, 1]``.
  Full per-class probability vectors are not stored in `.mlv` columnar
  frames, so the TV-distance ``(1/2) Σ_c |P_A - P_B|`` is approximated by
  the confidence gap — documented in the response ``divergence_kind``.
- Regression (both sides continuous): ``Δy_i = |pred_A - pred_B|`` and
  ``D_i = 0`` everywhere (no discrete conflict).

Memory: operates on borrowed float32 views, allocates at most two
``(N_min,)`` float32/int32 columns; no frame-object lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class DualComparisonSummary:
    total_frames_a: int
    total_frames_b: int
    total_aligned_frames: int
    truncated: bool
    is_classification: bool
    divergence_kind: str
    disagreement_indices: np.ndarray  # uint32
    probability_divergence: np.ndarray  # float32 [0..1] per aligned frame
    absolute_error: np.ndarray  # float32 per aligned frame
    agreement_rate: float
    mean_divergence: float
    max_divergence: float


def _as_1d_f32(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).ravel()
    if arr.size < n:
        pad = np.zeros((n - arr.size,), dtype=np.float32)
        arr = np.concatenate([arr, pad])
    return np.ascontiguousarray(arr[:n], dtype=np.float32)


def compare_decoded_payloads(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> DualComparisonSummary:
    """Compare two ``decode_mlv_package`` payload dicts (pure function)."""
    header_a = dict(payload_a.get("header", {}))
    header_b = dict(payload_b.get("header", {}))
    frames_a = dict(payload_a.get("frames_track", {}))
    frames_b = dict(payload_b.get("frames_track", {}))

    n_a = int(header_a.get("total_frames", 0) or 0)
    n_b = int(header_b.get("total_frames", 0) or 0)
    # Fall back to actual column length when headers disagree with columns.
    len_a = int(np.asarray(frames_a.get("predictions", []), dtype=np.float32).ravel().size)
    len_b = int(np.asarray(frames_b.get("predictions", []), dtype=np.float32).ravel().size)
    if n_a <= 0:
        n_a = len_a
    if n_b <= 0:
        n_b = len_b
    n_min = min(n_a, n_b)

    if n_min <= 0:
        empty_f32 = np.zeros((0,), dtype=np.float32)
        empty_u32 = np.zeros((0,), dtype=np.uint32)
        return DualComparisonSummary(
            total_frames_a=int(n_a),
            total_frames_b=int(n_b),
            total_aligned_frames=0,
            truncated=bool(n_a != n_b),
            is_classification=bool(header_a.get("is_classification", False) or header_b.get("is_classification", False)),
            divergence_kind="confidence_gap",
            disagreement_indices=empty_u32,
            probability_divergence=empty_f32,
            absolute_error=empty_f32,
            agreement_rate=1.0,
            mean_divergence=0.0,
            max_divergence=0.0,
        )

    preds_a = _as_1d_f32(frames_a.get("predictions", []), n_min)
    preds_b = _as_1d_f32(frames_b.get("predictions", []), n_min)
    conf_a = _as_1d_f32(frames_a.get("confidences", []), n_min)
    conf_b = _as_1d_f32(frames_b.get("confidences", []), n_min)
    conf_a = np.clip(conf_a, 0.0, 1.0)
    conf_b = np.clip(conf_b, 0.0, 1.0)

    is_classification = bool(header_a.get("is_classification", False) or header_b.get("is_classification", False))
    abs_err = np.abs(preds_a - preds_b).astype(np.float32)
    if is_classification:
        labels_a = np.rint(preds_a).astype(np.int32)
        labels_b = np.rint(preds_b).astype(np.int32)
        mask = labels_a != labels_b
        divergence = np.abs(conf_a - conf_b).astype(np.float32)
        kind = "confidence_gap"
    else:
        mask = np.zeros((n_min,), dtype=bool)
        divergence = np.clip(abs_err, 0.0, 1.0).astype(np.float32) if False else abs_err
        kind = "absolute_error"

    disagreement_indices = np.ascontiguousarray(np.flatnonzero(mask), dtype=np.uint32)
    agreement_rate = float(1.0 - (disagreement_indices.size / n_min)) if n_min else 1.0
    mean_div = float(np.mean(divergence)) if divergence.size else 0.0
    max_div = float(np.max(divergence)) if divergence.size else 0.0

    result = DualComparisonSummary(
        total_frames_a=int(n_a),
        total_frames_b=int(n_b),
        total_aligned_frames=int(n_min),
        truncated=bool(n_a != n_b),
        is_classification=bool(is_classification),
        divergence_kind=kind,
        disagreement_indices=disagreement_indices,
        probability_divergence=np.ascontiguousarray(divergence, dtype=np.float32),
        absolute_error=np.ascontiguousarray(abs_err, dtype=np.float32),
        agreement_rate=agreement_rate,
        mean_divergence=mean_div,
        max_divergence=max_div,
    )
    del preds_a, preds_b, conf_a, conf_b
    return result


def summary_to_response(summary: DualComparisonSummary, max_indices: int = 10_000) -> dict[str, Any]:
    """Serialize a summary to a JSON-safe dict (indices capped for HTTP)."""
    idx = summary.disagreement_indices[:max(0, int(max_indices))]
    return {
        "total_frames_a": summary.total_frames_a,
        "total_frames_b": summary.total_frames_b,
        "total_aligned_frames": summary.total_aligned_frames,
        "truncated": summary.truncated,
        "truncation_notice": (
            f"Frame count mismatch: truncating comparison to first {summary.total_aligned_frames} frames"
            if summary.truncated
            else ""
        ),
        "is_classification": summary.is_classification,
        "divergence_kind": summary.divergence_kind,
        "disagreement_count": int(summary.disagreement_indices.size),
        "disagreement_indices": [int(v) for v in idx.tolist()],
        "disagreement_indices_truncated": bool(summary.disagreement_indices.size > idx.size),
        "probability_divergence": [float(v) for v in summary.probability_divergence.tolist()],
        "absolute_error": [float(v) for v in summary.absolute_error.tolist()],
        "agreement_rate": summary.agreement_rate,
        "mean_divergence": summary.mean_divergence,
        "max_divergence": summary.max_divergence,
    }

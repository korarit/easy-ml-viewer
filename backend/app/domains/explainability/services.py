"""Phase 12 — fast local feature-attribution service (no KernelExplainer).

Three paths (mirrors the dev blueprint §4.1):

- Linear family: exact SHAP ``phi_i = w_i * (x_i - E[x_i])`` — additivity
  holds to float32 precision by construction.
- Black-box family: single-batch marginal perturbation against the column
  median — exactly **2** ``predict_fn`` calls (1 for the live row + 1 for
  the ``(D, D)`` perturbed batch), then rescale so
  ``base + sum(contribs) == current_pred`` (additivity guard).
- Percentiles: per-feature ``np.searchsorted`` over the cached sorted
  reference columns — ``O(D log N)``.

``shap.KernelExplainer`` is intentionally never imported here (§5.1 ban).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class FastLocalAttributionService:
    """High-speed local feature contribution calculator."""

    @staticmethod
    def compute_linear_contributions(
        X_batch: np.ndarray,
        feature_means: np.ndarray,
        weights: np.ndarray,
        intercept: float,
    ) -> tuple[np.ndarray, float]:
        """Exact linear SHAP: ``phi_i = w_i * (x_i - E[x_i])``."""
        centered = X_batch.astype(np.float32) - feature_means.astype(np.float32)
        contribs = centered * weights.astype(np.float32).reshape(1, -1)
        base_val = float(intercept + np.dot(feature_means, weights))
        return contribs.astype(np.float32), base_val

    @staticmethod
    def compute_fast_perturbation_contributions(
        predict_fn: Callable[[np.ndarray], np.ndarray],
        x_row: np.ndarray,
        feature_medians: np.ndarray,
        base_prediction: float,
    ) -> np.ndarray:
        """Black-box marginal perturbation in a single batch call.

        Builds a ``(D, D)`` matrix where row ``i`` has feature ``i``
        replaced by its median, then rescales raw diffs so the sum equals
        ``current_pred - base_prediction``. When the row equals the median
        vector (``sum_raw ≈ 0``) returns all zeros — never divides by zero.
        """
        x_vec = np.asarray(x_row, dtype=np.float32).ravel()
        medians = np.asarray(feature_medians, dtype=np.float32).ravel()
        num_features = int(x_vec.shape[0])
        if num_features == 0:
            return np.empty((0,), dtype=np.float32)

        perturbed_batch = np.tile(x_vec, (num_features, 1)).astype(np.float32)
        for i in range(num_features):
            perturbed_batch[i, i] = medians[i]

        current_pred = float(predict_fn(x_vec.reshape(1, -1))[0])
        perturbed_preds = np.asarray(
            predict_fn(perturbed_batch), dtype=np.float32
        ).ravel()
        raw_diffs = (current_pred - perturbed_preds).astype(np.float32)

        total_diff = current_pred - float(base_prediction)
        sum_raw = float(np.sum(raw_diffs))
        if abs(sum_raw) > 1e-9:
            raw_diffs = (raw_diffs * (total_diff / sum_raw)).astype(np.float32)
        else:
            raw_diffs = np.zeros_like(raw_diffs)
        if not np.all(np.isfinite(raw_diffs)):
            raw_diffs = np.where(np.isfinite(raw_diffs), raw_diffs, 0.0).astype(np.float32)
        return raw_diffs.astype(np.float32)

    @staticmethod
    def compute_percentiles_for_row(
        x_row: np.ndarray,
        sorted_reference_columns: np.ndarray,
    ) -> np.ndarray:
        """Per-feature percentile (0..100) via binary search."""
        ref = np.asarray(sorted_reference_columns, dtype=np.float32)
        row = np.asarray(x_row, dtype=np.float32).ravel()
        if ref.size == 0:
            return np.full(row.shape[0], 50.0, dtype=np.float32)
        if ref.ndim == 1:
            ref = ref.reshape(-1, row.shape[0] if row.size else 1)
        n_samples, n_features = ref.shape
        if n_samples == 0:
            return np.full(n_features, 50.0, dtype=np.float32)
        pcts = np.empty(n_features, dtype=np.float32)
        for j in range(n_features):
            idx = np.searchsorted(ref[:, j], row[j] if j < row.size else 0.0, side="right")
            pcts[j] = (float(idx) / float(n_samples)) * 100.0
        return pcts.astype(np.float32)

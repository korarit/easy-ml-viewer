"""Phase 12 — FastLocalAttributionService unit tests (spec §2.1 + edge #1)."""

from __future__ import annotations

import numpy as np

from app.domains.explainability.services import FastLocalAttributionService


def test_compute_linear_contributions_satisfies_exact_additivity():
    """Exact linear SHAP (w_i * (x_i - mean_i)) + base == prediction."""
    weights = np.array([2.5, -1.5, 0.8], dtype=np.float32)
    intercept = 1.2
    X_batch = np.array(
        [
            [10.0, 4.0, 2.0],
            [20.0, 8.0, 5.0],
            [15.0, 6.0, 3.0],
        ],
        dtype=np.float32,
    )
    means = X_batch.mean(axis=0)

    contribs, base_val = FastLocalAttributionService.compute_linear_contributions(
        X_batch=X_batch,
        feature_means=means,
        weights=weights,
        intercept=intercept,
    )

    assert contribs.dtype == np.float32
    for row_idx in range(len(X_batch)):
        exact_pred = float(np.dot(X_batch[row_idx], weights) + intercept)
        reconstructed_pred = float(base_val + np.sum(contribs[row_idx]))
        assert np.isclose(reconstructed_pred, exact_pred, atol=1e-4)


def test_compute_fast_perturbation_contributions_single_batch_and_zero_sum_guard():
    """Perturbation uses exactly 2 predict calls and guards divide-by-zero."""
    call_sizes: list[int] = []

    def mock_nonlinear_predict(X_in: np.ndarray) -> np.ndarray:
        call_sizes.append(len(X_in))
        return (X_in[:, 0] ** 2 - 3.0 * X_in[:, 1] + 0.5 * X_in[:, 2]).astype(np.float32)

    medians = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    base_pred = float(mock_nonlinear_predict(medians.reshape(1, -1))[0])
    call_sizes.clear()

    x_row = np.array([5.0, -2.0, 6.0], dtype=np.float32)
    expected_row_pred = float(mock_nonlinear_predict(x_row.reshape(1, -1))[0])
    call_sizes.clear()

    contribs = FastLocalAttributionService.compute_fast_perturbation_contributions(
        predict_fn=mock_nonlinear_predict,
        x_row=x_row,
        feature_medians=medians,
        base_prediction=base_pred,
    )

    # Exactly 2 calls: sizes [1, D=3] — never one call per column.
    assert call_sizes == [1, 3]
    assert np.isclose(base_pred + float(np.sum(contribs)), expected_row_pred, atol=1e-4)

    # Edge case #1: x_row == medians exactly → zeros, no NaN / ZeroDivision.
    zero_contribs = FastLocalAttributionService.compute_fast_perturbation_contributions(
        predict_fn=mock_nonlinear_predict,
        x_row=medians.copy(),
        feature_medians=medians,
        base_prediction=base_pred,
    )
    assert not np.isnan(zero_contribs).any()
    assert np.all(zero_contribs == 0.0)


def test_compute_percentiles_for_row_binary_search_accuracy():
    """searchsorted percentiles match true rank positions."""
    sorted_cols = np.column_stack([
        np.linspace(1.0, 100.0, 100, dtype=np.float32),
        np.linspace(0.0, 10.0, 100, dtype=np.float32),
    ])
    row = np.array([50.0, 9.0], dtype=np.float32)
    pcts = FastLocalAttributionService.compute_percentiles_for_row(row, sorted_cols)

    assert pcts.dtype == np.float32
    assert np.isclose(pcts[0], 50.0, atol=1.0)
    assert np.isclose(pcts[1], 90.0, atol=1.0)


def test_fast_attribution_latency_and_no_kernel_explainer_import():
    """< 5 ms per frame on 50 features; KernelExplainer never imported."""
    import sys
    import time

    assert "shap" not in sys.modules

    rng = np.random.default_rng(123)
    X_50 = rng.normal(size=(100, 50)).astype(np.float32)
    medians = np.median(X_50, axis=0)
    weights = rng.normal(size=50).astype(np.float32)

    def fast_predict(batch: np.ndarray) -> np.ndarray:
        return batch @ weights

    t0 = time.perf_counter()
    _ = FastLocalAttributionService.compute_fast_perturbation_contributions(
        predict_fn=fast_predict,
        x_row=X_50[0],
        feature_medians=medians,
        base_prediction=float(medians @ weights),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 5.0
    assert "shap" not in sys.modules

"""Phase 07 — GlobalEvaluationService unit tests (pure numeric, no I/O)."""

from __future__ import annotations

import numpy as np

from app.domains.simulation.evaluation_service import (
    GlobalEvaluationService,
    _top_k_argpartition,
)


def test_top_k_argpartition_orders_by_severity():
    scores = np.array([1.0, 5.0, 3.0, 5.0, 0.5])
    top = _top_k_argpartition(scores, 2, largest=True)
    assert sorted(top.tolist()) == [1, 3]  # the two 5.0 entries
    bottom = _top_k_argpartition(scores, 1, largest=False)
    assert bottom.tolist() == [4]
    assert _top_k_argpartition(np.array([]), 5).size == 0


def test_regression_metrics_and_slices():
    rng = np.random.default_rng(7)
    y_true = rng.normal(5.0, 1.0, size=200)
    noise = rng.normal(0.0, 0.5, size=200)
    noise[10] = 6.0  # single extreme outlier -> worst[0]
    y_pred = y_true + noise
    report = GlobalEvaluationService.evaluate_regression(y_true, y_pred, top_k=50)
    assert report.evaluated_rows == 200
    assert -1.0 <= report.r2_score <= 1.0
    assert report.scale_mismatch_warning is None
    assert report.rmse > 0 and report.mae > 0
    assert len(report.residual_hist_bins) == 17
    assert sum(report.residual_hist_counts) == 200
    assert report.top_worst_abs_error_indices[0] == 10
    assert len(report.top_worst_abs_error_indices) == 50
    # Over-predicted (y_hat > y) means residual < 0: inject one.
    assert len(report.top_over_predicted_indices) == 50
    assert len(report.top_under_predicted_indices) == 50
    assert len(report.top_accurate_indices) == 50


def test_regression_nan_rows_filtered_and_scale_warning():
    y_true = np.array([1.0, float("nan"), 3.0, 4.0])
    y_pred = np.array([100.0, 200.0, 300.0, 400.0])  # wrong scale -> R2 << -1
    report = GlobalEvaluationService.evaluate_regression(y_true, y_pred, top_k=50)
    assert report.evaluated_rows == 3
    assert report.scale_mismatch_warning is not None
    assert all(i != 1 for i in report.top_worst_abs_error_indices)


def test_ood_constant_column_safe():
    # Second column is constant -> std pinned to 1.0, no crash, z == 0 there.
    X = np.column_stack(
        [np.linspace(0, 1, 60, dtype=np.float32), np.full(60, 3.0, dtype=np.float32)]
    )
    X[0, 0] = 50.0  # extreme on the varying column
    out = GlobalEvaluationService.compute_ood_outliers(X, top_k=50)
    assert out[0] == 0
    X_flat = np.full((20, 3), 1.0, dtype=np.float32)
    assert GlobalEvaluationService.compute_ood_outliers(X_flat) == []


def test_classification_cells_and_uncertainty():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # 2 misclassified
    proba = np.array(
        [[0.9, 0.1], [0.49, 0.51], [0.8, 0.2], [0.2, 0.8],
         [0.55, 0.45], [0.1, 0.9], [0.7, 0.3], [0.3, 0.7]]
    )
    report = GlobalEvaluationService.evaluate_classification(
        y_true, y_pred, ["Stay", "Churn"], y_proba=proba, top_k=50
    )
    assert report.evaluated_rows == 8
    assert report.correct_count == 6
    assert report.misclassified_count == 2
    assert len(report.cells) == 4
    fp = next(c for c in report.cells if c.actual_class_idx == 0 and c.predicted_class_idx == 1)
    assert fp.row_count == 1 and fp.sample_row_indices == [1]
    assert fp.percentage == round(100.0 * 1 / 4, 2)
    # Most uncertain: |p-0.5| smallest -> row 1 (0.51) first.
    assert report.most_uncertain_indices[0] == 1
    assert sorted(report.all_misclassified_indices) == [1, 4]
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0


def test_classification_empty_cell_click_safe():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])  # perfect: FP/FN cells empty
    report = GlobalEvaluationService.evaluate_classification(y_true, y_pred, ["A", "B"])
    assert report.misclassified_count == 0
    empties = [c for c in report.cells if c.row_count == 0]
    assert len(empties) == 2
    assert all(c.sample_row_indices == [] for c in empties)

"""Phase 07 — GlobalEvaluationService: pure numeric metrics over in-RAM vectors.

Memory contract (spec §5.1):
- No DataFrames here — callers stream chunked predictions and pass flat
  ``float32`` vectors plus the ``(N,)`` max-z vector.
- Top-K selection uses ``np.argpartition`` (O(N)), never a full ``argsort``.
- Histogram is fixed at 16 bins; confusion cells cap stored samples at 500.

All NaN guards: ``evaluate_regression`` drops NaN (y_true/y_pred) pairs and
returns original-position ``valid_indices``-mapped rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from app.domains.simulation.evaluation_value_objects import (
    ClassificationEvaluationReport,
    ConfusionMatrixCell,
    RegressionEvaluationReport,
)

HIST_BINS = 16
MAX_CELL_SAMPLES = 500
MAX_MISCLASSIFIED_STORED = 500


def _top_k_argpartition(scores: np.ndarray, top_k: int, largest: bool = True) -> np.ndarray:
    """Indices of the top-K scores via partial heap select (O(N)).

    Returns indices into ``scores`` sorted by severity (descending when
    ``largest=True``). Safe on empty input and ``top_k > N``.
    """
    n = int(scores.shape[0])
    if n == 0 or top_k <= 0:
        return np.empty((0,), dtype=np.int64)
    k = int(min(top_k, n))
    if largest:
        part = np.argpartition(-scores, kth=k - 1)[:k]
        order = np.argsort(-scores[part], kind="mergesort")
    else:
        part = np.argpartition(scores, kth=k - 1)[:k]
        order = np.argsort(scores[part], kind="mergesort")
    return part[order].astype(np.int64, copy=False)


def _finite_round(value: float, ndigits: int = 4) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return round(f, ndigits)


class GlobalEvaluationService:
    # -- OOD / statistical outliers -------------------------------------
    @staticmethod
    def compute_ood_outliers(X_float32: np.ndarray, top_k: int = 50) -> list[int]:
        """Rows whose max |z-score| across features exceeds 3.0.

        ``std < 1e-8`` columns are pinned to 1.0 (edge case #3: constant cols).
        Returns at most ``top_k`` original positions, worst-first.
        """
        if X_float32.size == 0 or X_float32.shape[0] == 0:
            return []
        x = np.asarray(X_float32, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs((x - mean) / std)
        max_z = np.nanmax(z, axis=1)
        max_z = np.where(np.isfinite(max_z), max_z, -np.inf)
        outlier_pos = np.where(max_z > 3.0)[0]
        if outlier_pos.size == 0:
            return []
        ranked = outlier_pos[np.argsort(-max_z[outlier_pos], kind="mergesort")][:top_k]
        return [int(i) for i in ranked]

    @staticmethod
    def max_z_scores(X_float32: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Per-row max |z| given global mean/std (chunk-friendly two-pass OOD)."""
        x = np.asarray(X_float32, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs((x - mean) / std)
        max_z = np.nanmax(z, axis=1)
        return np.where(np.isfinite(max_z), max_z, -np.inf)

    # -- regression ------------------------------------------------------
    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 50,
    ) -> RegressionEvaluationReport:
        yt_raw = np.asarray(y_true, dtype=np.float64).ravel()
        yp_raw = np.asarray(y_pred, dtype=np.float64).ravel()
        n = min(yt_raw.shape[0], yp_raw.shape[0])
        yt_raw = yt_raw[:n]
        yp_raw = yp_raw[:n]

        valid_mask = np.isfinite(yt_raw) & np.isfinite(yp_raw)
        valid_positions = np.where(valid_mask)[0]
        yt = yt_raw[valid_mask]
        yp = yp_raw[valid_mask]
        evaluated = int(yt.shape[0])

        if evaluated == 0:
            return RegressionEvaluationReport(
                r2_score=0.0,
                rmse=0.0,
                mae=0.0,
                max_abs_error=0.0,
                evaluated_rows=0,
                residual_hist_bins=[0.0] * (HIST_BINS + 1),
                residual_hist_counts=[0] * HIST_BINS,
            )

        residuals = yt - yp
        abs_err = np.abs(residuals)

        r2 = float(r2_score(yt, yp)) if evaluated > 1 else 0.0
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        max_err = float(np.max(abs_err))

        counts, bins = np.histogram(residuals, bins=HIST_BINS)

        worst = _top_k_argpartition(abs_err, top_k, largest=True)
        # y - y_hat < 0  =>  y_hat > y  (over-predicted): most negative first.
        over = np.argsort(residuals, kind="mergesort")[: min(top_k, evaluated)]
        # y - y_hat > 0  =>  y_hat < y  (under-predicted): most positive first.
        under = np.argsort(-residuals, kind="mergesort")[: min(top_k, evaluated)]
        best = _top_k_argpartition(abs_err, top_k, largest=False)

        warning: str | None = None
        if r2 < -1.0:
            warning = (
                "Scale Mismatch Warning: Actual target column scale differs significantly "
                "from model predictions. Verify target column selection."
            )

        return RegressionEvaluationReport(
            r2_score=_finite_round(r2),
            rmse=_finite_round(rmse),
            mae=_finite_round(mae),
            max_abs_error=_finite_round(max_err),
            evaluated_rows=evaluated,
            scale_mismatch_warning=warning,
            residual_hist_bins=[_finite_round(float(b)) for b in bins],
            residual_hist_counts=[int(c) for c in counts],
            top_worst_abs_error_indices=[int(valid_positions[i]) for i in worst],
            top_over_predicted_indices=[int(valid_positions[i]) for i in over],
            top_under_predicted_indices=[int(valid_positions[i]) for i in under],
            top_accurate_indices=[int(valid_positions[i]) for i in best],
        )

    # -- classification ----------------------------------------------------
    @staticmethod
    def evaluate_classification(
        y_true_idx: np.ndarray,
        y_pred_idx: np.ndarray,
        class_labels: list[str],
        y_proba: np.ndarray | None = None,
        top_k: int = 50,
    ) -> ClassificationEvaluationReport:
        yt = np.asarray(y_true_idx, dtype=np.int64).ravel()
        yp = np.asarray(y_pred_idx, dtype=np.int64).ravel()
        n = min(yt.shape[0], yp.shape[0])
        yt = yt[:n]
        yp = yp[:n]
        n_classes = len(class_labels)
        if n == 0 or n_classes == 0:
            return ClassificationEvaluationReport(
                accuracy=0.0,
                macro_precision=0.0,
                macro_recall=0.0,
                macro_f1=0.0,
                evaluated_rows=0,
                correct_count=0,
                misclassified_count=0,
                class_labels=list(class_labels),
            )

        labels_idx = list(range(n_classes))
        acc = float(accuracy_score(yt, yp))
        macro_p = float(precision_score(yt, yp, labels=labels_idx, average="macro", zero_division=0))
        macro_r = float(recall_score(yt, yp, labels=labels_idx, average="macro", zero_division=0))
        macro_f1 = float(f1_score(yt, yp, labels=labels_idx, average="macro", zero_division=0))
        cm = confusion_matrix(yt, yp, labels=labels_idx)

        cells: list[ConfusionMatrixCell] = []
        for a in range(n_classes):
            row_total = int(cm[a, :].sum())
            for p in range(n_classes):
                count = int(cm[a, p])
                mask = (yt == a) & (yp == p)
                sample_pos = np.where(mask)[0][:MAX_CELL_SAMPLES]
                pct = (100.0 * count / row_total) if row_total > 0 else 0.0
                cells.append(
                    ConfusionMatrixCell(
                        actual_class_idx=a,
                        predicted_class_idx=p,
                        actual_label=class_labels[a],
                        predicted_label=class_labels[p],
                        row_count=count,
                        percentage=_finite_round(pct, 2),
                        sample_row_indices=[int(i) for i in sample_pos],
                    )
                )

        mis_mask = yt != yp
        mis_pos = np.where(mis_mask)[0][:MAX_MISCLASSIFIED_STORED]

        uncertain: list[int] = []
        if y_proba is not None:
            proba = np.asarray(y_proba, dtype=np.float64)
            if proba.ndim == 2 and proba.shape[0] == n and proba.shape[1] >= 1:
                if proba.shape[1] == 2:
                    # Boundary uncertainty: |p(class1) - 0.5| ascending.
                    p1 = np.where(np.isfinite(proba[:, 1]), proba[:, 1], 0.5)
                    dist = np.abs(p1 - 0.5)
                    order = np.argsort(dist, kind="mergesort")[: min(top_k, n)]
                else:
                    # Multiclass: lowest max-probability (least confident) first.
                    with np.errstate(invalid="ignore"):
                        conf = np.nanmax(np.where(np.isfinite(proba), proba, -np.inf), axis=1)
                    conf = np.where(np.isfinite(conf), conf, np.inf)
                    order = np.argsort(conf, kind="mergesort")[: min(top_k, n)]
                uncertain = [int(i) for i in order]

        correct = int(n) - int(np.count_nonzero(mis_mask))
        return ClassificationEvaluationReport(
            accuracy=_finite_round(acc),
            macro_precision=_finite_round(macro_p),
            macro_recall=_finite_round(macro_r),
            macro_f1=_finite_round(macro_f1),
            evaluated_rows=int(n),
            correct_count=correct,
            misclassified_count=int(np.count_nonzero(mis_mask)),
            class_labels=list(class_labels),
            cells=cells,
            all_misclassified_indices=[int(i) for i in mis_pos],
            most_uncertain_indices=uncertain,
        )

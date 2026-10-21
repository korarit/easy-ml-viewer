"""Phase 11 — event scanner service (vectorized anomaly detection).

Detection is fully vectorized (boolean masks over the whole queue); the
Python loop only visits flagged frames to materialize ``AutoEventPinVO``
records. A realistic 20,000-row scan finishes in < 10 ms (§5.3 SLA).
"""

from __future__ import annotations

import numpy as np

from app.domains.timeline.value_objects import AutoEventPinVO, EventType

OUTLIER_Z_THRESHOLD = 3.2
LOW_CONF_LO = 0.45
LOW_CONF_HI = 0.55
# Edge case #4: a shuffled queue flips almost every frame — keep only the
# flips whose probability jump is decisive.
FLIP_RATE_THRESHOLD = 0.40
FLIP_DELTA_PROB_MIN = 0.5


class TimelineEventScannerService:
    @staticmethod
    def scan_events(
        X: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        actual_targets: np.ndarray | None,
        is_classification: bool,
        row_indices: list[int],
    ) -> list[AutoEventPinVO]:
        preds = np.asarray(predictions, dtype=np.float64).ravel()
        confs = np.asarray(confidences, dtype=np.float64).ravel()
        n_frames = int(preds.shape[0])
        if n_frames == 0:
            return []
        if confs.shape[0] != n_frames:
            confs = np.full((n_frames,), 0.85, dtype=np.float64)

        mat = np.asarray(X, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] != n_frames:
            raise ValueError(
                f"X must be an (N, D) matrix with N == len(predictions) ({n_frames})"
            )

        # NaN-normalized working copies (PASS_NAN models may emit NaNs).
        pred_filled = np.where(np.isfinite(preds), preds, 0.0)
        conf_filled = np.where(np.isfinite(confs), confs, 0.85)
        pred_round = np.round(pred_filled).astype(np.int64)

        actuals: np.ndarray | None = None
        if actual_targets is not None:
            actuals = np.asarray(actual_targets, dtype=np.float64).ravel()
            if actuals.shape[0] != n_frames:
                actuals = None

        # -- mask 1: misclassified (edge #2 — skipped when unlabeled) ----
        mis_mask = np.zeros(n_frames, dtype=bool)
        reg_mask = np.zeros(n_frames, dtype=bool)
        residual_threshold = 0.0
        if actuals is not None:
            act_finite = np.isfinite(actuals)
            if is_classification:
                act_round = np.round(np.where(act_finite, actuals, 0.0)).astype(np.int64)
                mis_mask = act_finite & (pred_round != act_round)
            else:
                with np.errstate(invalid="ignore"):
                    residuals = np.abs(preds - actuals)
                    valid = np.isfinite(residuals)
                if np.count_nonzero(valid):
                    with np.errstate(invalid="ignore"):
                        r_mean = float(np.nanmean(residuals[valid]))
                        r_std = float(np.nanstd(residuals[valid]))
                    residual_threshold = r_mean + 2.0 * (r_std if np.isfinite(r_std) else 0.0)
                    reg_mask = valid & (residuals > residual_threshold)

        # -- mask 2: decision flips + edge #4 storm filter ---------------
        flip_mask = np.zeros(n_frames, dtype=bool)
        if is_classification and n_frames > 1:
            changed = pred_round[1:] != pred_round[:-1]
            flip_count = int(np.count_nonzero(changed))
            flip_rate = flip_count / (n_frames - 1)
            finite_conf = confs[np.isfinite(confs)]
            conf_spread = (
                float(np.max(finite_conf) - np.min(finite_conf))
                if finite_conf.size
                else 0.0
            )
            # Guard: probability-free classifiers report a constant fallback
            # confidence — deltas would all be 0 and the filter would nuke
            # every flip, so only filter when confidences actually vary.
            if flip_rate > FLIP_RATE_THRESHOLD and conf_spread > 1e-9:
                delta = np.abs(conf_filled[1:] - conf_filled[:-1])
                flip_mask[1:] = changed & (delta > FLIP_DELTA_PROB_MIN)
            else:
                flip_mask[1:] = changed

        # -- mask 3: low confidence near the boundary ---------------------
        if is_classification:
            low_mask = (conf_filled >= LOW_CONF_LO) & (conf_filled <= LOW_CONF_HI)
        else:
            low_mask = np.zeros(n_frames, dtype=bool)

        # -- mask 4: input outliers via vectorized Z-score ----------------
        # Zero-variance columns collapse to std 1.0 instead of dividing by 0.
        mat64 = mat.astype(np.float64, copy=False)
        with np.errstate(invalid="ignore"):
            col_mean = np.nanmean(mat64, axis=0)
        col_std = np.nanstd(mat64, axis=0)
        col_std = np.where(np.isfinite(col_std) & (col_std >= 1e-9), col_std, 1.0)
        denom = np.where(np.isfinite(col_mean), col_mean, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z_scores = np.abs((mat64 - denom) / col_std)
        with np.errstate(invalid="ignore"):
            max_z = np.nanmax(np.where(np.isfinite(z_scores), z_scores, 0.0), axis=1)
        max_z = np.where(np.isfinite(max_z), max_z, 0.0)
        out_mask = max_z > OUTLIER_Z_THRESHOLD

        # -- materialize VOs for flagged frames only ----------------------
        flagged = np.where(mis_mask | reg_mask | flip_mask | low_mask | out_mask)[0]
        events: list[AutoEventPinVO] = []
        for i in (int(v) for v in flagged):
            row_id = int(row_indices[i]) if i < len(row_indices) else i
            pred_i = float(pred_filled[i])
            conf_i = float(conf_filled[i])

            if mis_mask[i] and actuals is not None:
                actual_i = float(actuals[i])
                events.append(
                    AutoEventPinVO(
                        frame_index=i,
                        dataset_row_index=row_id,
                        event_type="misclassified",
                        severity=1.0,
                        title=f"ทายผิดที่เฟรม #{i}",
                        detail=f"Pred: {int(round(pred_i))} vs Actual: {int(round(actual_i))}",
                    )
                )
            if reg_mask[i] and actuals is not None:
                actual_i = float(actuals[i])
                err_val = abs(float(pred_i - actual_i))
                events.append(
                    AutoEventPinVO(
                        frame_index=i,
                        dataset_row_index=row_id,
                        event_type="misclassified",
                        severity=0.9,
                        title=f"คลาดเคลื่อนสูงที่เฟรม #{i}",
                        detail=f"|y - y_hat| = {err_val:.4f}",
                    )
                )
            if flip_mask[i]:
                prev_i = float(pred_filled[i - 1])
                events.append(
                    AutoEventPinVO(
                        frame_index=i,
                        dataset_row_index=row_id,
                        event_type="decision_flip",
                        severity=0.75,
                        title=f"เปลี่ยนคำตัดสิน ({int(round(prev_i))} → {int(round(pred_i))})",
                        detail=f"สลับคลาสจากเฟรม #{i - 1}",
                    )
                )
            if low_mask[i]:
                events.append(
                    AutoEventPinVO(
                        frame_index=i,
                        dataset_row_index=row_id,
                        event_type="low_confidence",
                        severity=0.6,
                        title=f"ความมั่นใจก้ำกึ่ง ({conf_i * 100:.1f}%)",
                        detail="อยู่ใกล้เส้นแบ่งเขตการตัดสินใจ (Decision Boundary)",
                    )
                )
            if out_mask[i]:
                events.append(
                    AutoEventPinVO(
                        frame_index=i,
                        dataset_row_index=row_id,
                        event_type="input_outlier",
                        severity=0.7,
                        title=f"ค่า Input สุดโต่ง (Z={float(max_z[i]):.1f}σ)",
                        detail="ค่า Feature ต้นทางเบี่ยงเบนจากค่าเฉลี่ยเกิน 3.2 SD",
                    )
                )

        # Deterministic order: frame asc, then severity desc.
        order: dict[EventType, int] = {
            "misclassified": 0,
            "decision_flip": 1,
            "low_confidence": 2,
            "input_outlier": 3,
        }
        events.sort(key=lambda e: (e.frame_index, -e.severity, order[e.event_type]))
        return events

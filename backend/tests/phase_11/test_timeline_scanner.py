"""Phase 11 — TimelineEventScannerService unit tests (test spec §2.1)."""

from __future__ import annotations

import numpy as np

from app.domains.timeline.services import TimelineEventScannerService


def _base_matrix(n: int = 26, d: int = 2, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (np.ones((n, d), dtype=np.float32) + rng.normal(0, 0.05, size=(n, d)).astype(np.float32))


def test_scan_events_detects_all_four_anomaly_categories():
    X = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [1.0, 1.0],
            [100.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    X_full = np.vstack([X, np.ones((20, 2), dtype=np.float32)])
    n_total = len(X_full)

    predictions = np.zeros(n_total, dtype=np.float32)
    predictions[2] = 1.0

    confidences = np.full(n_total, 0.95, dtype=np.float32)
    confidences[3] = 0.49

    actual_targets = np.zeros(n_total, dtype=np.float32)
    row_indices = list(range(n_total))

    events = TimelineEventScannerService.scan_events(
        X=X_full,
        predictions=predictions,
        confidences=confidences,
        actual_targets=actual_targets,
        is_classification=True,
        row_indices=row_indices,
    )

    event_types_by_frame = {(e.frame_index, e.event_type) for e in events}
    assert (2, "misclassified") in event_types_by_frame
    assert (2, "decision_flip") in event_types_by_frame
    assert (3, "low_confidence") in event_types_by_frame
    assert (4, "input_outlier") in event_types_by_frame


def test_scan_events_unlabeled_dataset_skips_misclassified_gracefully():
    X = np.random.randn(20, 3).astype(np.float32)
    preds = np.array([0 if i < 10 else 1 for i in range(20)], dtype=np.float32)
    confs = np.full(20, 0.85, dtype=np.float32)

    events = TimelineEventScannerService.scan_events(
        X=X,
        predictions=preds,
        confidences=confs,
        actual_targets=None,
        is_classification=True,
        row_indices=list(range(20)),
    )

    assert all(e.event_type != "misclassified" for e in events)
    assert any(e.event_type == "decision_flip" and e.frame_index == 10 for e in events)


def test_scan_events_empty_queue_returns_empty():
    events = TimelineEventScannerService.scan_events(
        X=np.empty((0, 3), dtype=np.float32),
        predictions=np.empty((0,), dtype=np.float32),
        confidences=np.empty((0,), dtype=np.float32),
        actual_targets=None,
        is_classification=True,
        row_indices=[],
    )
    assert events == []


def test_scan_events_flip_storm_keeps_only_decisive_jumps():
    # Edge case #4: alternating classes flip almost every frame.
    n = 40
    X = np.ones((n, 2), dtype=np.float32)
    preds = np.array([i % 2 for i in range(n)], dtype=np.float32)
    # Gentle probability wobble — no decisive jump anywhere.
    confs = np.array([0.51 + (0.02 if i % 2 else -0.02) for i in range(n)], dtype=np.float64)

    events = TimelineEventScannerService.scan_events(
        X=X,
        predictions=preds,
        confidences=confs,
        actual_targets=None,
        is_classification=True,
        row_indices=list(range(n)),
    )
    flips = [e for e in events if e.event_type == "decision_flip"]
    assert flips == []

    # One decisive spike survives the storm (both adjacent flips jump).
    confs = np.full(n, 0.49, dtype=np.float64)
    confs[20] = 1.0
    events = TimelineEventScannerService.scan_events(
        X=X,
        predictions=preds,
        confidences=confs,
        actual_targets=None,
        is_classification=True,
        row_indices=list(range(n)),
    )
    flips = [e for e in events if e.event_type == "decision_flip"]
    assert [e.frame_index for e in flips] == [20, 21]


def test_scan_events_constant_confidence_disables_flip_filter():
    # Probability-free classifiers report flat fallback confidences —
    # the storm filter must not nuke every flip (spread guard).
    n = 30
    X = np.ones((n, 2), dtype=np.float32)
    preds = np.array([i % 2 for i in range(n)], dtype=np.float32)
    confs = np.full(n, 1.0, dtype=np.float64)

    events = TimelineEventScannerService.scan_events(
        X=X,
        predictions=preds,
        confidences=confs,
        actual_targets=None,
        is_classification=True,
        row_indices=list(range(n)),
    )
    flips = [e for e in events if e.event_type == "decision_flip"]
    assert len(flips) == n - 1


def test_scan_events_regression_high_residual():
    rng = np.random.default_rng(3)
    n = 60
    X = (rng.normal(0, 1, size=(n, 2))).astype(np.float32)
    preds = np.zeros(n, dtype=np.float64)
    actuals = np.zeros(n, dtype=np.float64)
    actuals[10] = 50.0  # single huge residual: mean+2σ gate must catch it

    events = TimelineEventScannerService.scan_events(
        X=X,
        predictions=preds,
        confidences=np.full(n, 0.9),
        actual_targets=actuals,
        is_classification=False,
        row_indices=list(range(n)),
    )
    by_frame = {(e.frame_index, e.event_type) for e in events}
    assert (10, "misclassified") in by_frame
    assert all(e.event_type != "decision_flip" for e in events)

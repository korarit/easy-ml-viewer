"""Phase 48 — dual-2D disagreement-area math (spec §2.1, E48.1/E48.2)."""

import numpy as np

from app.domains.simulation.dual_2d_compare_service import (
    compare_label_fields,
    frame_disagreement_indices,
    summary_to_response,
)


def test_full_consensus_reports_zero_pct_with_notice():
    field = np.linspace(0.0, 1.0, 144, dtype=np.float32)
    summary = compare_label_fields(field, field.copy(), grid_resolution=12,
                                   x_feature="income", y_feature="debt_ratio")
    assert summary.n_cells == 144
    assert summary.n_conflict_cells == 0
    assert summary.disagreement_pct == 0.0
    assert summary.agreement_pct == 100.0
    assert "Consensus" in summary.consensus_notice
    body = summary_to_response(summary)
    assert body["disagreement_pct"] == 0.0
    assert body["conflict_mask"] == [0] * 144


def test_total_inversion_reports_hundred_pct_with_warning():
    low = np.zeros((100,), dtype=np.float32)
    high = np.ones((100,), dtype=np.float32)
    summary = compare_label_fields(low, high, grid_resolution=10)
    assert summary.n_conflict_cells == 100
    assert summary.disagreement_pct == 100.0
    assert "Inverted" in summary.consensus_notice


def test_partial_disagreement_pct_matches_formula():
    # 0.5-thresholded labels: first half agrees, second half conflicts.
    a = np.concatenate([np.zeros(50, dtype=np.float32), np.zeros(50, dtype=np.float32)])
    b = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)])
    summary = compare_label_fields(a, b, grid_resolution=10)
    assert summary.n_conflict_cells == 50
    assert summary.disagreement_pct == 50.0
    assert summary.conflict_mask.dtype == np.uint8
    body = summary_to_response(summary)
    assert body["n_conflict_cells"] == 50
    assert sum(body["conflict_mask"]) == 50


def test_regression_plane_has_no_conflicts_but_frames_pin_nothing():
    a = np.linspace(0.0, 5.0, 64, dtype=np.float32)
    b = a + 1.0
    summary = compare_label_fields(a, b, grid_resolution=8, prediction_task="regression")
    assert summary.n_conflict_cells == 0
    assert frame_disagreement_indices([1.0, 2.0], [1.5, 2.0], is_classification=False) == []


def test_frame_pins_mark_only_mismatched_labels():
    assert frame_disagreement_indices([0, 1, 1, 0], [0, 0, 1, 1]) == [1, 3]
    assert frame_disagreement_indices([0, 1], [0, 1]) == []

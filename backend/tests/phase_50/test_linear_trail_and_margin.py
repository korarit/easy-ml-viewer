"""Phase 50 — trail alignment (A), margin rings (B), hexbin flags (D)."""

import numpy as np

from app.domains.model_engine.adapters.linear.linear_point_intelligence import (
    MARGIN_LEVEL,
    error_distance_correlation,
    hexbin_boundary_flags,
    logistic_ring_thickness,
    margin_violation_mask,
    residual_stem_lengths,
    trail_alignment_signs,
    z_scores,
)


def test_trail_signs_follow_weight_dot_step():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    assert trail_alignment_signs(pts, 1.0, 0.0) == [1, 0, 0]
    assert trail_alignment_signs(pts, -1.0, 0.0) == [-1, 0, 0]
    assert trail_alignment_signs(pts, 0.0, 1.0) == [0, 1, 0]
    assert trail_alignment_signs(np.zeros((1, 2)), 1.0, 1.0) == []


def test_margin_mask_boundary_inclusive():
    z = np.array([0.0, 1.0, -1.0, 1.0001, float("nan")])
    mask = margin_violation_mask(z)
    assert mask.tolist() == [True, True, True, False, False]
    assert MARGIN_LEVEL == 1.0


def test_ring_thickness_peaks_at_uncertainty():
    t = logistic_ring_thickness(np.array([0.5, 0.7, 0.9, 0.0, 1.0]))
    assert t[0] == min(t[0], 0.08) and t[0] == 0.08  # max at P=0.5
    assert t[-1] == t[-2] == 0.02  # thinnest at extremes
    assert all(0.02 <= v <= 0.08 for v in t)


def test_residual_stems_normalize_and_direct():
    lengths, signs = residual_stem_lengths(np.array([2.0, -4.0, 0.0]))
    assert max(lengths) == 1.5  # peak maps to max_world
    assert signs.tolist() == [1.0, -1.0, 0.0]
    assert lengths[0] == lengths[1] / 2.0
    l0, s0 = residual_stem_lengths(np.zeros(3))
    assert (l0 == 0).all() and (s0 == 0).all()


def test_hexbin_flags_only_line_crossers():
    # Line x=0 (w=(1,0), b=0): cells at x=±5 untouched, cell at x=0 flagged.
    flags = hexbin_boundary_flags(
        np.array([-5.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0]),
        np.array([0.45, 0.45, 0.45]), 1.0, 0.0, 0.0,
    )
    assert flags == [False, True, False]
    # Degenerate weights flag nothing.
    assert hexbin_boundary_flags(
        np.array([0.0]), np.array([0.0]), np.array([0.45]), 0.0, 0.0, 1.0
    ) == [False]


def test_error_gradient_negative_for_separable_data():
    d = np.array([0.1, 0.4, 1.0, 2.0, 3.0])
    e = np.array([0.9, 0.6, 0.3, 0.1, 0.0])
    assert error_distance_correlation(d, e) < -0.9
    assert np.isnan(error_distance_correlation(np.ones(4), e[:4]))
    assert np.isnan(error_distance_correlation(np.array([1.0, 2.0]), np.array([1.0, 0.0])))


def test_z_scores_match_plane():
    pts = np.array([[1.0, 2.0]])
    assert z_scores(pts, 3.0, 4.0, -5.0)[0] == 3.0 + 8.0 - 5.0
    assert z_scores(np.empty((0, 2)), 1.0, 1.0, 0.0).size == 0

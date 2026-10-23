"""Phase 49 — mechanisms A/B/C pure math (spec §2.1–§2.3)."""

import math

import numpy as np
import pytest

from app.domains.simulation.point_intelligence_service import (
    COMET_ALPHA_MAX,
    COMET_ALPHA_MIN,
    GHOST_ALPHA,
    GHOST_RADIUS,
    RING_CONTRAST_WIDTH,
    RING_INNER_RADIUS,
    RING_OUTER_RADIUS,
    RING_THICKNESS,
    comet_alpha,
    comet_stroke_width,
    comet_trail_alphas,
    contrast_stroke,
    errors_only_indices,
    residual_spoke_endpoint,
)


def test_comet_head_is_full_opacity_tail_decays():
    alphas = comet_trail_alphas(11, 5, 4)
    assert alphas[5] == pytest.approx(COMET_ALPHA_MAX)
    assert alphas[4] < alphas[5] and alphas[3] < alphas[4]
    # Monotone decay away from the head on the past side.
    past = alphas[1:6]
    assert past == sorted(past)


def test_comet_alpha_formula_spot_check():
    sigma = 4 / 2.5
    expected = COMET_ALPHA_MIN + (COMET_ALPHA_MAX - COMET_ALPHA_MIN) * math.exp(
        -(2.0**2) / (2.0 * sigma**2)
    )
    assert comet_alpha(3, 5, 4) == pytest.approx(expected)


def test_out_of_window_frames_are_ghosts():
    alphas = comet_trail_alphas(10, 5, 2)
    assert alphas[0] == GHOST_ALPHA and alphas[9] == GHOST_ALPHA
    assert GHOST_RADIUS == pytest.approx(0.06)


def test_stroke_width_grows_toward_head():
    w_tail = comet_stroke_width(1, 5, 4)
    w_head = comet_stroke_width(5, 5, 4)
    assert w_head > w_tail
    assert w_head == pytest.approx(0.03 * 3.0)


def test_dual_ring_geometry_constants():
    assert (RING_INNER_RADIUS, RING_OUTER_RADIUS, RING_THICKNESS) == (0.08, 0.14, 0.04)
    assert RING_CONTRAST_WIDTH == pytest.approx(0.025)
    assert contrast_stroke(True) == "#0b0f14"
    assert contrast_stroke(False) == "#ffffff"


def test_errors_only_classification():
    idx = errors_only_indices(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))
    assert idx == [1, 3]
    assert errors_only_indices(np.array([0, 1]), np.array([0, 1])) == []


def test_errors_only_regression_eps():
    idx = errors_only_indices(None, None, residuals=np.array([0.1, 0.9, -0.6]), eps=0.5)
    assert idx == [1, 2]


def test_residual_spoke_scales_with_error():
    small = residual_spoke_endpoint(1.0, 2.0, 0.2)
    big = residual_spoke_endpoint(1.0, 2.0, 2.0)
    assert small[:2] == [1.0, 2.0] == big[:2]
    len_small = math.hypot(small[2] - 1.0, small[3] - 2.0)
    len_big = math.hypot(big[2] - 1.0, big[3] - 2.0)
    assert len_big > len_small > 0
    # Non-finite residuals collapse to a dot (no NaN reaches the SVG).
    dot = residual_spoke_endpoint(0.0, 0.0, float("nan"))
    assert dot[2:] == [0.0, 0.0]

"""Phase 55 — input loss-gradient spokes (C) and data-desert hexbins (D).

DoD mapping: spoke length follows ``||∇_x L||`` with the adversarial unit
direction (spec §2.3); desert badges fire exactly where the hexbin is empty
yet the network predicts with ``max P > 0.85`` (spec §2.4, DoD #3), with
parity against the real MLP input-gradient path (``input_x_gradient``).
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

from app.domains.model_engine.adapters.neural_onnx.mlp_adapter import MLPAdapter
from app.domains.model_engine.adapters.neural_onnx.neural_point_intelligence import (
    desert_hazard,
    gradient_spoke_lengths,
    gradient_unit_vectors,
    hidden_halo_tone,
)


def _blobs():
    X, y = make_blobs(
        n_samples=80, centers=[[0.0, 0.0], [2.5, 2.5]], n_features=2,
        cluster_std=1.2, random_state=7,
    )
    return X.astype(np.float64), y


def test_gradient_spoke_lengths_peak_normalize():
    lengths, ok = gradient_spoke_lengths(np.array([1.0, -2.0, 0.0, np.nan]))
    assert lengths.tolist() == pytest.approx([0.75, 1.5, 0.0, 0.0])
    assert ok.tolist() == [True, True, False, False]
    lz, okz = gradient_spoke_lengths(np.zeros(3))
    assert (lz == 0.0).all() and not okz.any()  # converged → no spokes


def test_gradient_unit_vectors_are_directions():
    units = gradient_unit_vectors(np.array([[3.0, 4.0], [0.0, 0.0], [np.nan, 1.0]]))
    assert units[0].tolist() == pytest.approx([0.6, 0.8])
    assert units[1].tolist() == [0.0, 0.0]  # zero gradient → no spoke
    assert units[2].tolist() == [0.0, 0.0]  # non-finite → no spoke
    assert gradient_unit_vectors(np.empty((0, 2))).shape == (0, 2)


def test_mlp_input_gradient_parity_with_attribution_path():
    X, y = _blobs()
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=3).fit(X, y)
    adapter = MLPAdapter(model=mlp)
    attrib = np.asarray(
        adapter.compute_input_gradient_attribution(X[:20].astype(np.float32)), dtype=np.float64
    )
    assert attrib.shape == (20, 2)
    assert np.all(np.isfinite(attrib))
    # Gradient norms vary across rows — the spokes have somewhere to point.
    norms = np.hypot(attrib[:, 0], attrib[:, 1])
    assert norms.max() > norms.min()
    lengths, _ = gradient_spoke_lengths(norms)
    assert lengths.max() == pytest.approx(1.5)  # peak norm → world cap
    units = gradient_unit_vectors(attrib[:5])
    assert np.allclose(np.hypot(units[0, 0], units[0, 1]), 1.0)


def test_desert_hazard_flags_empty_overconfident_cells():
    hazard, is_desert = desert_hazard(
        np.array([0, 5, 0, 0]), np.array([0.94, 0.99, 0.5, 0.86])
    )
    assert hazard.tolist() == pytest.approx([0.94, 0.0, 0.5, 0.86])
    assert is_desert.tolist() == [True, False, False, True]  # > 0.85 + empty
    # Non-finite proba yields NaN hazard and no badge.
    h, d = desert_hazard(np.array([0]), np.array([np.nan]))
    assert np.isnan(h[0]) and not d[0]
    he, de = desert_hazard(np.array([]), np.array([]))
    assert he.size == 0 and de.size == 0


def test_desert_threshold_is_exact():
    _, is_desert = desert_hazard(np.array([0, 0]), np.array([0.85, 0.8501]))
    assert is_desert.tolist() == [False, True]  # strictly greater than 0.85


def test_hidden_halo_tone_tracks_activation_energy():
    tone = hidden_halo_tone(np.array([[1.0, -2.0], [0.0, 0.0]]))
    assert tone.tolist() == pytest.approx([1.5, 0.0])
    assert np.isnan(hidden_halo_tone(np.array([[np.nan]]))[0])
    assert hidden_halo_tone(np.empty((0, 3))).size == 0

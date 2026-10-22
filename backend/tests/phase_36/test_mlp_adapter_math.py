"""Phase 36 — MLP exact forward parity, Top-32 cap, Dead ReLU, attribution tests."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.neural_onnx.mlp_adapter import (
    MAX_RENDERED_NEURONS_PER_LAYER,
    MAX_SYNAPSES_PER_LAYER_PAIR,
    MLPAdapter,
    MLPClassifierAdapter,
    MLPRegressorAdapter,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import GridSpec3D


def _spec(**over: object) -> GridSpec3D:
    base: dict[str, object] = {
        "axis_x_feature_index": 0,
        "axis_y_feature_index": 1,
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "resolution": 20,
        "fixed_feature_medians": {},
    }
    base.update(over)
    return GridSpec3D(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("activation", ["relu", "tanh", "logistic", "identity"])
def test_mlp_forward_pass_exact_parity_with_sklearn(activation: str) -> None:
    """compute_forward_trace() output layer matches predict_proba (< 1e-5)."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = (X[:, 0] * 1.5 - X[:, 1] > 0).astype(int)
    mlp = MLPClassifier(
        hidden_layer_sizes=(24, 16), activation=activation, max_iter=30, random_state=42
    )
    mlp.fit(X, y)
    adapter = MLPAdapter(model=mlp)
    trace = adapter.compute_forward_trace(X)
    sklearn_probs = mlp.predict_proba(X).astype(np.float32)
    np.testing.assert_allclose(trace.output_activations, sklearn_probs, rtol=1e-5, atol=1e-5)


def test_top32_neuron_downsampling_and_dead_relu_detection() -> None:
    """128-node hidden layer caps at 32 nodes; forced dead neurons detected."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0.1, 1.0, size=(50, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=50)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 48), activation="relu", max_iter=10, random_state=7
    )
    mlp.fit(X, y)
    # Force 3 dead ReLU neurons in the first hidden layer.
    mlp.coefs_[0][:, [3, 17, 99]] = -5.0
    mlp.intercepts_[0][[3, 17, 99]] = -100.0
    adapter = MLPAdapter(model=mlp)
    trace = adapter.compute_forward_trace(X)
    # Hidden layers (indices 1, 2) downsampled to Top-32.
    assert trace.layer_activations[1].shape == (50, 32)
    assert trace.layer_activations[2].shape == (50, 32)
    assert len(trace.rendered_node_indices[1]) == MAX_RENDERED_NEURONS_PER_LAYER
    for edges in trace.synapse_edges_per_layer:
        assert len(edges) <= MAX_SYNAPSES_PER_LAYER_PAIR
    # Full-width dead masks preserve original indices.
    assert trace.dead_neuron_masks[1].shape == (128,)
    dead_indices = set(np.where(trace.dead_neuron_masks[1])[0].tolist())
    assert {3, 17, 99}.issubset(dead_indices)
    assert trace.layer_telemetry[1]["dead_neurons"] >= 3


def test_input_gradient_attribution_matches_finite_difference() -> None:
    """Analytical Input x Gradient matches numeric gradient of the forward map."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(12, 5)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.2 > 0).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="tanh", max_iter=80, random_state=11)
    mlp.fit(X, y)
    adapter = MLPAdapter(model=mlp)
    analytic = adapter.compute_input_gradient_attribution(X)

    # Exact self-consistency: numeric gradient of the float64 forward map
    # (no float32 rounding) must match the chain-rule Jacobian (< 1e-4).
    from app.domains.model_engine.adapters.neural_onnx import mlp_adapter as _m

    coefs = [np.asarray(c, dtype=np.float64) for c in mlp.coefs_]
    intercepts = [np.asarray(b, dtype=np.float64).ravel() for b in mlp.intercepts_]

    def forward_pos(row: np.ndarray) -> float:
        acts, _ = _m._forward_full(
            row.reshape(1, -1), coefs, intercepts, "tanh", "logistic"
        )
        return float(np.asarray(acts[-1]).ravel()[0])

    eps = 1e-7
    row = X[0].astype(np.float64)
    for j in range(5):
        rp, rm = row.copy(), row.copy()
        rp[j] += eps
        rm[j] -= eps
        numeric = (forward_pos(rp) - forward_pos(rm)) / (2 * eps) * row[j]
        assert analytic[0, j] == pytest.approx(numeric, rel=1e-4, abs=1e-6)

    # Rough end-to-end agreement with sklearn predict_proba (float32 path).
    def proba_pos(row: np.ndarray) -> float:
        return float(mlp.predict_proba(row.reshape(1, -1).astype(np.float32))[0, 1])

    eps32 = 1e-3
    for j in range(5):
        rp, rm = row.copy(), row.copy()
        rp[j] += eps32
        rm[j] -= eps32
        numeric32 = (proba_pos(rp) - proba_pos(rm)) / (2 * eps32) * row[j]
        assert analytic[0, j] == pytest.approx(numeric32, rel=5e-2, abs=5e-4)


def test_regressor_parity_and_waterfall_identity() -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    model = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", max_iter=60, random_state=13)
    model.fit(X, y)
    adapter = MLPRegressorAdapter()
    trace = adapter.compute_forward_trace(X, estimator=model)
    np.testing.assert_allclose(
        np.asarray(trace.output_activations).ravel(),
        np.asarray(model.predict(X), dtype=np.float32).ravel(),
        rtol=1e-5,
        atol=1e-5,
    )
    traces = adapter.trace_internal_flow(model, X, [0, 1, 2], ["a", "b", "c", "d"])
    preds = np.asarray(model.predict(X[:3]), dtype=np.float64)
    for i, tr in enumerate(traces):
        np.testing.assert_allclose(
            tr.base_value + sum(tr.feature_contributions.values()),
            float(preds[i]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert tr.extra_mechanics["is_regressor"] is True
        assert tr.extra_mechanics["model_family"] == "mlp"


def test_single_vs_deep_hidden_layouts_and_overflow_guard() -> None:
    """1-hidden vs 4-hidden layouts both trace; extreme inputs stay finite."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(30, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    single = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", max_iter=20, random_state=17)
    single.fit(X, y)
    deep = MLPClassifier(
        hidden_layer_sizes=(32, 16, 12, 8), activation="relu", max_iter=20, random_state=17
    )
    deep.fit(X, y)
    assert MLPAdapter(model=single).compute_forward_trace(X).layer_dimensions == [5, 16, 1]
    deep_trace = MLPAdapter(model=deep).compute_forward_trace(X)
    assert deep_trace.layer_dimensions == [5, 32, 16, 12, 8, 1]
    assert len(deep_trace.synapse_edges_per_layer) == 5

    extreme = np.full((10, 5), 1e4, dtype=np.float32)
    y_ext = np.array([1] * 5 + [0] * 5)
    X_ext = np.vstack([np.ones((5, 5)), np.zeros((5, 5))]).astype(np.float32)
    ov = MLPClassifier(hidden_layer_sizes=(8,), activation="logistic", max_iter=5, random_state=0)
    with np.errstate(all="ignore"):
        ov.fit(X_ext, y_ext)
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        overflow_trace = MLPAdapter(model=ov).compute_forward_trace(extreme)
    assert np.isfinite(np.asarray(overflow_trace.output_activations)).all()
    assert not [w for w in record if "overflow" in str(w.message).lower()]


def test_registry_pipeline_wrapper_and_schema() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = MLPClassifier(hidden_layer_sizes=(12,), activation="relu", max_iter=20, random_state=19)
    model.fit(X, y)
    assert ModelRegistry.resolve_adapter(model).algorithm_key == "mlp"
    reg = MLPRegressor(hidden_layer_sizes=(12,), activation="relu", max_iter=20, random_state=19)
    reg.fit(X, X[:, 0] * 2.0)
    assert ModelRegistry.resolve_adapter(reg).algorithm_key == "mlp_regressor"
    pipe = Pipeline(
        [("scaler", StandardScaler()),
         ("mlp", MLPClassifier(hidden_layer_sizes=(12,), activation="relu", max_iter=20, random_state=19))]
    )
    pipe.fit(X, y)
    assert ModelRegistry.resolve_adapter(pipe).algorithm_key == "mlp"
    schema = UniversalSchemaDetector.extract_schema(model)
    assert schema.n_features_in == 3
    assert UniversalSchemaDetector.extract_schema(reg).n_features_in == 3
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    assert len(MLPClassifierAdapter().trace_internal_flow(
        joblib.load(buf), X, [0, 1], ["a", "b", "c"])) == 2


def test_surface_preview_shape_and_overlays() -> None:
    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = MLPClassifier(hidden_layer_sizes=(12, 8), activation="relu", max_iter=30, random_state=23)
    model.fit(X, y)
    mesh = MLPClassifierAdapter().compute_3d_surface(model, _spec(), n_features=3)
    assert mesh.resolution == 20
    assert len(mesh.z_values) == 400
    ov = mesh.extra_overlays
    assert ov["model_family"] == "mlp"
    assert ov["layer_dimensions"] == [3, 12, 8, 1]
    assert ov["activation"] == "relu"
    assert all(0.0 <= v <= 1.0 for v in mesh.z_values)
    big = MLPClassifierAdapter().compute_3d_surface(model, _spec(resolution=200), n_features=3)
    assert big.resolution == 100

    Xr = rng.normal(size=(60, 3)).astype(np.float32)
    yr = (Xr[:, 0] * 2.0 - Xr[:, 1]).astype(np.float32)
    reg = MLPRegressor(hidden_layer_sizes=(12,), activation="tanh", max_iter=30, random_state=23)
    reg.fit(Xr, yr)
    rmesh = MLPRegressorAdapter().compute_3d_surface(reg, _spec(), n_features=3)
    assert rmesh.extra_overlays["is_regressor"] is True
    assert len(rmesh.extra_overlays["regression_z_range"]) == 2

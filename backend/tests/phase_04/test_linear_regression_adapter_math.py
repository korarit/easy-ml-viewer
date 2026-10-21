"""Phase 04 — LinearRegressionAdapter math identity tests.

Gate: base_value + sum(contributions) == .predict(X) for every row (< 1e-4).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.adapters.linear.linear_regression_adapter import (
    LinearRegressionAdapter,
)


def test_linear_trace_contributions_sum_exact_to_prediction(
    house_model: LinearRegression,
    house_X: np.ndarray,
    house_y: np.ndarray,
    feature_names: list[str],
) -> None:
    adapter = LinearRegressionAdapter()
    rows = house_X[:3]
    batch_res = adapter.predict_batch(house_model, rows)
    traces = adapter.trace_internal_flow(
        model_obj=house_model,
        X_float32=rows,
        row_indices=[0, 1, 2],
        feature_names=feature_names,
    )

    for idx, trace in enumerate(traces):
        expected_pred = batch_res.predictions[idx]
        reconstructed = trace.base_value + sum(trace.feature_contributions.values())
        assert np.isclose(reconstructed, expected_pred, rtol=1e-5, atol=1.0)
        assert np.isclose(trace.final_prediction, expected_pred, rtol=1e-5, atol=1.0)

    # Fitted on noise-free data: weights recover ground truth (rtol 1e-3).
    assert np.isclose(traces[0].feature_contributions["area_sqm"], 42500.0 * float(rows[0, 0]), rtol=1e-3)
    assert np.isclose(
        traces[0].feature_contributions["distance_bts_km"], -185000.0 * float(rows[0, 2]), rtol=1e-3
    )


def test_linear_adapter_with_standard_scaler_pipeline() -> None:
    rng = np.random.default_rng(123)
    X = rng.uniform(10.0, 200.0, size=(50, 3)).astype(np.float32)
    y = (500.0 + 12.0 * X[:, 0] - 8.5 * X[:, 1] + 3.2 * X[:, 2]).astype(np.float32)

    pipe = Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())]).fit(X, y)
    adapter = LinearRegressionAdapter()
    traces = adapter.trace_internal_flow(pipe, X[:5], [0, 1, 2, 3, 4], ["f0", "f1", "f2"])
    preds = pipe.predict(X[:5])

    for i, tr in enumerate(traces):
        total = tr.base_value + sum(tr.feature_contributions.values())
        assert np.isclose(total, preds[i], rtol=1e-4)


def test_multi_target_and_no_intercept_linear_regression() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    Y_multi = np.column_stack([2.0 * X[:, 0] + X[:, 1], -1.0 * X[:, 0] + 3.0 * X[:, 1]]).astype(
        np.float32
    )

    model_multi = LinearRegression(fit_intercept=False).fit(X, Y_multi)
    adapter = LinearRegressionAdapter()

    weights = adapter.extract_feature_weights(model_multi, n_features=2)
    assert len(weights) == 2
    traces = adapter.trace_internal_flow(model_multi, X[:1], [0], ["x0", "x1"])
    assert traces[0].base_value == 0.0
    # Identity still holds for target 0.
    pred0 = float(np.asarray(model_multi.predict(X[:1]))[0, 0])
    assert np.isclose(traces[0].final_prediction, pred0, rtol=1e-4)
    assert np.isclose(
        traces[0].base_value + sum(traces[0].feature_contributions.values()), pred0, rtol=1e-4
    )

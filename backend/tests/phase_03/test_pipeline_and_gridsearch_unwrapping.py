"""Container unwrapping: Pipeline steps and GridSearchCV best estimators."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector


def test_unwrap_pipeline_and_grid_search_cv() -> None:
    rng = np.random.default_rng(7)
    X = (rng.random((20, 3)) * 10).astype(np.float32)
    y = (rng.random(20) * 5).astype(np.float32)
    pipe = Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())])
    grid = GridSearchCV(pipe, param_grid={"ols__fit_intercept": [True, False]}, cv=2)
    grid.fit(X, y)

    _, final_est, steps = UniversalSchemaDetector.unwrap_model_container(grid)
    assert isinstance(final_est, LinearRegression)
    assert steps == ["StandardScaler", "LinearRegression"]
    adapter = ModelRegistry.resolve_adapter(final_est)
    assert adapter.algorithm_key == "linear_regression"


def test_unwrap_plain_estimator_returns_single_step() -> None:
    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    model = LinearRegression().fit(X, y)
    predict_obj, final_est, steps = UniversalSchemaDetector.unwrap_model_container(model)
    assert predict_obj is model
    assert final_est is model
    assert steps == ["LinearRegression"]

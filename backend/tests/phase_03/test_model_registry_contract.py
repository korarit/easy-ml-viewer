"""Registry contract: decorator registration, resolution, open-closed extension."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from app.core.exceptions import ModelLoadError
from app.domains.model_engine.base_adapter import BaseModelAdapter
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.value_objects import (
    GridSpec3D,
    InternalFlowRowTrace,
    PredictionBatchResult,
    SurfaceMesh3DData,
    TaskType,
    Visual3DType,
)


def _fit_linear() -> LinearRegression:
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return LinearRegression().fit(X, y)


def test_resolve_builtin_linear_regression() -> None:
    adapter = ModelRegistry.resolve_adapter(_fit_linear())
    assert adapter.algorithm_key == "linear_regression"
    assert adapter.task_type == TaskType.REGRESSION
    assert adapter.visual_3d_type == Visual3DType.FLAT_HYPERPLANE


def test_unsupported_model_lists_supported_algorithms() -> None:
    with pytest.raises(ModelLoadError) as exc_info:
        ModelRegistry.resolve_adapter(object())
    assert exc_info.value.code == "UNSUPPORTED_MODEL_TYPE"
    assert "linear_regression" in exc_info.value.message


def test_open_closed_dummy_adapter_registration(isolated_registry) -> None:
    @isolated_registry.register("dummy_probe")
    class DummyAdapter(BaseModelAdapter):
        algorithm_key = "dummy_probe"
        display_name = "Dummy Probe"
        family_name = "Test Family"
        task_type = TaskType.CLUSTERING
        visual_3d_type = Visual3DType.CLUSTER_CENTROID_VORONOI

        @classmethod
        def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
            return type(estimator).__name__ == "DummyEstimator"

        def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
            return [None] * n_features

        def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
            return PredictionBatchResult(predictions=[])

        def trace_internal_flow(self, model_obj: Any, X_float32: np.ndarray, row_indices: list[int], feature_names: list[str]) -> list[InternalFlowRowTrace]:
            return []

        def compute_3d_surface(self, model_obj: Any, grid_spec: GridSpec3D, n_features: int) -> SurfaceMesh3DData:
            return SurfaceMesh3DData(
                visual_type=Visual3DType.CLUSTER_CENTROID_VORONOI,
                resolution=2,
                x_grid=[0.0, 1.0],
                y_grid=[0.0, 1.0],
                z_values=[0.0, 0.0, 0.0, 0.0],
            )

    class DummyEstimator:
        pass

    # New family resolves with zero changes to detector or endpoints.
    adapter = isolated_registry.resolve_adapter(DummyEstimator())
    assert adapter.algorithm_key == "dummy_probe"
    assert any(a["algorithm_key"] == "dummy_probe" for a in isolated_registry.supported_algorithms())

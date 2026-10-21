"""Standard contract every model adapter must implement (Open-Closed core)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.domains.model_engine.value_objects import (
    FeatureRequirement,
    GridSpec3D,
    InternalFlowRowTrace,
    PredictionBatchResult,
    SurfaceMesh3DData,
    TaskType,
    Visual3DType,
)


class BaseModelAdapter(ABC):
    algorithm_key: str = ""
    display_name: str = ""
    family_name: str = ""
    task_type: TaskType = TaskType.REGRESSION
    visual_3d_type: Visual3DType = Visual3DType.FLAT_HYPERPLANE
    # Phase 05 edge case #5: most sklearn estimators (LinearRegression, SVC,
    # MLP, ...) raise on NaN input — adapters that natively tolerate NaN
    # (e.g. HistGradientBoosting, XGBoost) override this to True.
    supports_nan_input: bool = False

    @classmethod
    @abstractmethod
    def can_handle(cls, estimator: Any, file_ext: str = ".joblib") -> bool:
        """Return True when this adapter supports the given estimator object."""

    @abstractmethod
    def extract_feature_weights(self, estimator: Any, n_features: int) -> list[float | None]:
        """Per-index coefficients / importances (None when not applicable)."""

    @abstractmethod
    def predict_batch(self, model_obj: Any, X_float32: np.ndarray) -> PredictionBatchResult:
        """Vectorized prediction over a float32 matrix."""

    @abstractmethod
    def trace_internal_flow(
        self,
        model_obj: Any,
        X_float32: np.ndarray,
        row_indices: list[int],
        feature_names: list[str],
    ) -> list[InternalFlowRowTrace]:
        """Per-row internal decision mechanics (weights, paths, distances...)."""

    @abstractmethod
    def compute_3d_surface(
        self, model_obj: Any, grid_spec: GridSpec3D, n_features: int
    ) -> SurfaceMesh3DData:
        """3D decision-surface coordinates over a resolution x resolution grid."""

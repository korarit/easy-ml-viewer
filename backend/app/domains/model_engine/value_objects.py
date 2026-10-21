"""Model-engine bounded-context value objects (Pydantic schemas)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


class Visual3DType(str, Enum):
    FLAT_HYPERPLANE = "flat_hyperplane"
    PROBABILITY_SIGMOID_SURFACE = "probability_sigmoid_surface"
    STEPPED_AXIS_BLOCKS = "stepped_axis_blocks"
    ENSEMBLE_SMOOTH_TERRAIN = "ensemble_smooth_terrain"
    VORONOI_KNN_PROXIMITY = "voronoi_knn_proximity"
    SVM_MARGIN_TUBE = "svm_margin_tube"
    GAUSSIAN_ELLIPSOID_DOMES = "gaussian_ellipsoid_domes"
    NEURAL_MORPHING_MANIFOLD = "neural_morphing_manifold"
    CLUSTER_CENTROID_VORONOI = "cluster_centroid_voronoi"
    ANOMALY_ISOLATION_DEPTH = "anomaly_isolation_depth"


class FeatureRequirement(BaseModel):
    index: int = Field(ge=0)
    name: str
    expected_dtype: str = "float32"
    is_auto_generated_name: bool = False
    learned_weight_or_importance: float | None = None


class GridSpec3D(BaseModel):
    axis_x_feature_index: int = Field(ge=0)
    axis_y_feature_index: int = Field(ge=0)
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: int = Field(default=50, ge=2, le=200)
    fixed_feature_medians: dict[int, float] = Field(default_factory=dict)


class PredictionBatchResult(BaseModel):
    predictions: list[float | int | str]
    probabilities: list[list[float]] | None = None
    confidences: list[float] | None = None


class InternalFlowRowTrace(BaseModel):
    row_index: int = Field(ge=0)
    base_value: float
    final_prediction: float | int | str
    feature_contributions: dict[str, float]
    extra_mechanics: dict[str, Any] = Field(default_factory=dict)


class SurfaceMesh3DData(BaseModel):
    visual_type: Visual3DType
    resolution: int = Field(ge=2, le=200)
    x_grid: list[float]
    y_grid: list[float]
    z_values: list[float]
    extra_overlays: dict[str, Any] = Field(default_factory=dict)

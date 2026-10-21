"""Phase 07 — Global evaluation value objects (Pydantic schemas).

Backend returns only pre-aggregated payloads: histogram bins (16), per-cell
sample indices (cap 500), top-K index lists. Payload stays < 25 KB.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegressionEvaluationReport(BaseModel):
    r2_score: float
    rmse: float
    mae: float
    max_abs_error: float
    evaluated_rows: int = Field(ge=0)
    scale_mismatch_warning: str | None = None
    residual_hist_bins: list[float]
    residual_hist_counts: list[int]
    top_worst_abs_error_indices: list[int] = Field(default_factory=list)
    top_over_predicted_indices: list[int] = Field(default_factory=list)
    top_under_predicted_indices: list[int] = Field(default_factory=list)
    top_accurate_indices: list[int] = Field(default_factory=list)


class ConfusionMatrixCell(BaseModel):
    actual_class_idx: int = Field(ge=0)
    predicted_class_idx: int = Field(ge=0)
    actual_label: str
    predicted_label: str
    row_count: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)
    sample_row_indices: list[int] = Field(default_factory=list)


class ClassificationEvaluationReport(BaseModel):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    evaluated_rows: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    misclassified_count: int = Field(ge=0)
    class_labels: list[str]
    cells: list[ConfusionMatrixCell] = Field(default_factory=list)
    all_misclassified_indices: list[int] = Field(default_factory=list)
    most_uncertain_indices: list[int] = Field(default_factory=list)


class PredictionDistribution(BaseModel):
    pred_min: float
    pred_max: float
    pred_mean: float
    pred_std: float


class GlobalEvaluationSummary(BaseModel):
    task_type: str
    evaluated_row_count: int = Field(ge=0)
    ood_outlier_indices: list[int] = Field(default_factory=list)
    ood_outlier_count: int = Field(ge=0)
    needs_label_mapping: bool = False
    label_mapping_hint: str | None = None
    raw_target_labels: list[str] = Field(default_factory=list)
    expected_classes: list[str] = Field(default_factory=list)
    prediction_distribution: PredictionDistribution | None = None
    regression: RegressionEvaluationReport | None = None
    classification: ClassificationEvaluationReport | None = None

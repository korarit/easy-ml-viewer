/** Phase 07 — typed client for the global evaluation endpoint. */
import { apiFetch } from '../../../core/api/client';

export interface RegressionReport {
  r2_score: number;
  rmse: number;
  mae: number;
  max_abs_error: number;
  evaluated_rows: number;
  scale_mismatch_warning: string | null;
  residual_hist_bins: number[];
  residual_hist_counts: number[];
  top_worst_abs_error_indices: number[];
  top_over_predicted_indices: number[];
  top_under_predicted_indices: number[];
  top_accurate_indices: number[];
}

export interface ConfusionCell {
  actual_class_idx: number;
  predicted_class_idx: number;
  actual_label: string;
  predicted_label: string;
  row_count: number;
  percentage: number;
  sample_row_indices: number[];
}

export interface ClassificationReport {
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  evaluated_rows: number;
  correct_count: number;
  misclassified_count: number;
  class_labels: string[];
  cells: ConfusionCell[];
  all_misclassified_indices: number[];
  most_uncertain_indices: number[];
}

export interface GlobalEvaluationSummary {
  task_type: string;
  evaluated_row_count: number;
  ood_outlier_indices: number[];
  ood_outlier_count: number;
  needs_label_mapping: boolean;
  label_mapping_hint: string | null;
  raw_target_labels: string[];
  expected_classes: string[];
  prediction_distribution: {
    pred_min: number;
    pred_max: number;
    pred_mean: number;
    pred_std: number;
  } | null;
  regression: RegressionReport | null;
  classification: ClassificationReport | null;
}

export async function fetchComputeEvaluation(body: {
  dataset_id: string;
  model_id: string;
  mapping_id: string;
  top_k?: number;
  target_label_map?: Record<string, number | string> | null;
}): Promise<GlobalEvaluationSummary> {
  return apiFetch<GlobalEvaluationSummary>('/api/v1/evaluations/compute', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: body.dataset_id,
      model_id: body.model_id,
      mapping_id: body.mapping_id,
      top_k: body.top_k ?? 50,
      target_label_map: body.target_label_map ?? null,
    }),
  });
}

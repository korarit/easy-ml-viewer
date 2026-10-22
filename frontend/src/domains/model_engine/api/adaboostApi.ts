/** Phase 30 — typed client for adaboost staged-cascade trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface AdaBoostTopLearner {
  stage: number;
  weight: number;
  error: number;
  feature_idx: number;
  feature_name: string | null;
  threshold: number;
  delta: number;
  cumulative: number;
}

export interface AdaBoostTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  classes: string[];
  class_labels: Array<string | number>;
  model_family: string;
  n_estimators_actual: number;
  n_estimators_param: number;
  early_terminated: boolean;
  base_value_stage0: number;
  sampled_stage_indices: number[];
  staged_trajectory: number[];
  top_influential_learners: AdaBoostTopLearner[];
  crossed_threshold_stage: number | null;
  learning_rate: number | null;
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface AdaBoostRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: AdaBoostTraceMechanics;
}

export interface AdaBoostStumpSplit {
  stage: number;
  feature_idx: number;
  threshold: number;
  weight: number;
  error: number;
}

export interface AdaBoostSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  n_estimators_actual?: number;
  estimator_weights?: number[];
  estimator_errors?: number[];
  stump_splits?: AdaBoostStumpSplit[];
  feature_importances?: number[];
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface AdaBoostSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface AdaBoostSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: AdaBoostSurfaceOverlays;
  telemetry: AdaBoostSurfaceTelemetry;
}

export async function fetchAdaBoostTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: AdaBoostRowTrace[] }> {
  return apiFetch<{ traces: AdaBoostRowTrace[] }>(
    `/api/v1/models/${encodeURIComponent(modelId)}/trace-preview`,
    {
      method: 'POST',
      body: JSON.stringify({
        rows: body.rows,
        row_indices: body.row_indices,
        feature_names: body.feature_names,
        target_idx: body.target_idx ?? 0,
      }),
    },
  );
}

export async function fetchAdaBoostSurfacePreview(
  modelId: string,
  body: {
    axis_x_feature_index: number;
    axis_y_feature_index: number;
    x_min: number;
    x_max: number;
    y_min: number;
    y_max: number;
    resolution: number;
    fixed_feature_medians: Record<number, number>;
    target_idx?: number;
  },
): Promise<AdaBoostSurfacePreview> {
  return apiFetch<AdaBoostSurfacePreview>(
    `/api/v1/models/${encodeURIComponent(modelId)}/surface-preview`,
    {
      method: 'POST',
      body: JSON.stringify({
        axis_x_feature_index: body.axis_x_feature_index,
        axis_y_feature_index: body.axis_y_feature_index,
        x_min: body.x_min,
        x_max: body.x_max,
        y_min: body.y_min,
        y_max: body.y_max,
        resolution: body.resolution,
        fixed_feature_medians: body.fixed_feature_medians,
        target_idx: body.target_idx ?? 0,
      }),
    },
  );
}

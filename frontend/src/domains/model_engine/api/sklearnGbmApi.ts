/** Phase 31 — typed client for sklearn GBM staged-residual trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface SklearnGbmTopStage {
  stage: number;
  leaf_index: number;
  delta_raw: number;
  cumulative_raw: number;
  cumulative_pred: number | null;
}

export interface SklearnGbmTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  loss: string;
  classes: string[];
  class_labels: Array<string | number>;
  model_family: string;
  learning_rate: number;
  n_stages_total: number;
  n_stages_traced: number;
  negated_target_zero_view: boolean;
  base_raw_score: number;
  final_raw_score: number;
  sampled_stage_indices: number[];
  staged_predictions: number[];
  stage_raw_deltas_sample: number[];
  leaf_indices_per_stage: number[];
  top_residual_stages: SklearnGbmTopStage[];
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface SklearnGbmRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: SklearnGbmTraceMechanics;
}

export interface SklearnGbmSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  loss?: string;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  learning_rate?: number;
  n_stages_total?: number;
  n_estimators_actual?: number;
  feature_importances?: number[];
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface SklearnGbmSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface SklearnGbmSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: SklearnGbmSurfaceOverlays;
  telemetry: SklearnGbmSurfaceTelemetry;
}

export async function fetchSklearnGbmTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: SklearnGbmRowTrace[] }> {
  return apiFetch<{ traces: SklearnGbmRowTrace[] }>(
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

export async function fetchSklearnGbmSurfacePreview(
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
): Promise<SklearnGbmSurfacePreview> {
  return apiFetch<SklearnGbmSurfacePreview>(
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

/** Phase 32 — typed client for HistGBM quantized-bin trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface HistGbmTopStage {
  stage: number;
  leaf_index: number;
  delta_raw: number;
  cumulative_raw: number;
  cumulative_pred: number | null;
  nan_routed_feature_idx: number | null;
}

export interface HistGbmTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  loss: string;
  link_function: string;
  classes: string[];
  class_labels: Array<string | number>;
  model_family: string;
  learning_rate: number;
  n_iter_total: number;
  n_iter_traced: number;
  negated_target_zero_view: boolean;
  base_raw_score: number;
  final_raw_score: number;
  sampled_stage_indices: number[];
  staged_predictions: number[];
  stage_raw_deltas_sample: number[];
  leaf_indices_per_stage: number[];
  top_residual_stages: HistGbmTopStage[];
  has_missing_values: boolean;
  monotonic_cst: number[];
  categorical_feature_idx: number[];
  gain_feature_importances: number[];
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface HistGbmRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: HistGbmTraceMechanics;
}

export interface HistGbmSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  loss?: string;
  link_function?: string;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  learning_rate?: number;
  n_iter_total?: number;
  n_estimators_actual?: number;
  gain_feature_importances?: number[];
  bin_thresholds?: Record<string, number[]>;
  axis_bin_thresholds?: { x: number[]; y: number[] };
  monotonic_cst?: number[];
  categorical_feature_idx?: number[];
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface HistGbmSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface HistGbmSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: HistGbmSurfaceOverlays;
  telemetry: HistGbmSurfaceTelemetry;
}

export async function fetchHistGbmTracePreview(
  modelId: string,
  body: {
    rows: (number | string)[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: HistGbmRowTrace[] }> {
  return apiFetch<{ traces: HistGbmRowTrace[] }>(
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

export async function fetchHistGbmSurfacePreview(
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
): Promise<HistGbmSurfacePreview> {
  return apiFetch<HistGbmSurfacePreview>(
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

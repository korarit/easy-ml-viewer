/** Phase 19 — typed client for ridge/lasso trace/surface previews (same REST shape as linear/logistic). */
import { apiFetch } from '../../../core/api/client';

export interface RegularizedTraceMechanics {
  weights: number[];
  intercept: number;
  target_idx: number;
  target_class_idx?: number;
  penalty_type: string;
  alpha: number;
  active_weight_count: number;
  active_indices: number[];
  zeroed_indices: number[];
  l1_norm: number;
  l2_norm: number;
  sparsity_ratio: number;
  is_classifier: boolean;
  is_pseudo_proba: boolean;
  is_binary?: boolean;
  classes?: string[];
  class_labels?: Array<string | number>;
  decision_scores?: number[];
  probabilities?: number[];
  predicted_class?: string | number;
  winning_class_idx?: number;
  decision_threshold?: number;
  criterion?: string | null;
  transformed_inputs?: number[];
}

export interface RegularizedRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: RegularizedTraceMechanics;
}

export interface RegularizedSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface RegularizedSurfaceOverlays {
  wx?: number;
  wy?: number;
  is_x_pruned?: boolean;
  is_y_pruned?: boolean;
  zero_influence_axes?: boolean;
  intercept_offset?: number;
  base_intercept?: number;
  weight_row?: number[];
  penalty_type?: string;
  alpha?: number;
  target_idx?: number;
  resolved_target_idx?: number;
  resolved_target_class_idx?: number;
  boundary_threshold?: number | null;
  decision_threshold?: number;
  gpu_formula_valid?: boolean;
  is_classifier?: boolean;
  is_pseudo_proba?: boolean;
  is_binary?: boolean;
  classes?: string[];
  active_indices?: number[];
  zeroed_indices?: number[];
  active_weight_count?: number;
  l1_norm?: number;
  l2_norm?: number;
  sparsity_ratio?: number;
}

export interface RegularizedSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: RegularizedSurfaceOverlays;
  telemetry: RegularizedSurfaceTelemetry;
}

export async function fetchRegularizedTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: RegularizedRowTrace[] }> {
  return apiFetch<{ traces: RegularizedRowTrace[] }>(
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

export async function fetchRegularizedSurfacePreview(
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
): Promise<RegularizedSurfacePreview> {
  return apiFetch<RegularizedSurfacePreview>(
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

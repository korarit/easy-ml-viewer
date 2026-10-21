/** Phase 20 — typed client for elastic-net trace/surface previews (same REST shape as regularized). */
import { apiFetch } from '../../../core/api/client';

export interface ElasticNetTraceMechanics {
  weights: number[];
  intercept: number;
  target_idx: number;
  target_class_idx?: number;
  active_task_idx: number;
  n_tasks: number;
  is_multitask: boolean;
  is_classifier: boolean;
  is_pseudo_proba: boolean;
  alpha: number;
  l1_ratio: number;
  lambda_1: number;
  lambda_2: number;
  l1_term: number;
  l2_term: number;
  l1_penalty_burden: number;
  l2_penalty_burden: number;
  total_regularization: number;
  penalty_shares: number[];
  penalty_type: string;
  active_weight_count: number;
  active_indices: number[];
  zeroed_indices: number[];
  joint_zeroed_indices: number[];
  l1_norm: number;
  l2_norm: number;
  sparsity_ratio: number;
  n_iter: number;
  max_iter: number;
  convergence_warning: boolean;
  positive_only: boolean;
}

export interface ElasticNetRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: ElasticNetTraceMechanics;
}

export interface ElasticNetSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface ElasticNetSurfaceOverlays {
  wx?: number;
  wy?: number;
  is_x_pruned?: boolean;
  is_y_pruned?: boolean;
  zero_influence_axes?: boolean;
  intercept_offset?: number;
  base_intercept?: number;
  weight_row?: number[];
  alpha?: number;
  l1_ratio?: number;
  lambda_1?: number;
  lambda_2?: number;
  l1_penalty_burden?: number;
  l2_penalty_burden?: number;
  total_regularization?: number;
  penalty_shares?: number[];
  penalty_type?: string;
  target_idx?: number;
  active_task_idx?: number;
  resolved_target_idx?: number;
  n_tasks?: number;
  is_multitask?: boolean;
  positive_only?: boolean;
  n_iter?: number;
  max_iter?: number;
  convergence_warning?: boolean;
  boundary_threshold?: number | null;
  gpu_formula_valid?: boolean;
  is_classifier?: boolean;
  active_indices?: number[];
  zeroed_indices?: number[];
  joint_zeroed_indices?: number[];
  active_weight_count?: number;
  l1_norm?: number;
  l2_norm?: number;
  sparsity_ratio?: number;
}

export interface ElasticNetSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: ElasticNetSurfaceOverlays;
  telemetry: ElasticNetSurfaceTelemetry;
}

export async function fetchElasticNetTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: ElasticNetRowTrace[] }> {
  return apiFetch<{ traces: ElasticNetRowTrace[] }>(
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

export async function fetchElasticNetSurfacePreview(
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
): Promise<ElasticNetSurfacePreview> {
  return apiFetch<ElasticNetSurfacePreview>(
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

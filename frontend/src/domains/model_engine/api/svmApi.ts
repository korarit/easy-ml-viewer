/** Phase 23 — typed client for SVM kernel-saddle trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface SvmKernelForce {
  sv_position: number;
  support_index: number;
  kernel_value: number;
  force: number;
}

export interface SvmSupportDisplay {
  coordinates: number[][];
  dual_weights: number[];
  support_indices: number[];
  display_positions: number[];
  importance: number[];
  total_support_vectors: number;
  displayed_count: number;
}

export interface SvmTraceMechanics {
  kernel: string;
  gamma: number | string;
  C: number | null;
  nu: number | null;
  degree: number;
  coef0: number;
  decision_function_shape: string;
  has_native_proba: boolean;
  n_support_per_class: number[];
  total_support_vectors: number;
  is_linear_model: boolean;
  epsilon: number;
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_binary?: boolean;
  is_ovo?: boolean;
  is_oneclass?: boolean;
  is_pseudo_proba?: boolean;
  classes?: string[];
  class_labels?: Array<string | number>;
  decision_scores?: number[];
  raw_score?: number;
  geometric_gap?: number;
  inside_margin?: boolean;
  margin_status?: string;
  margin_width?: number;
  intercept?: number;
  probabilities?: number[];
  predicted_class?: string | number;
  predicted_label?: string;
  platt_conflict?: boolean;
  kernel_forces_top: SvmKernelForce[];
  kernel_force_sum: number;
  weight_row?: number[] | null;
  support_vectors_display: SvmSupportDisplay;
}

export interface SvmRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: SvmTraceMechanics;
}

export interface SvmSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface SvmSurfaceOverlays {
  kernel?: string;
  gamma?: number | string;
  C?: number | null;
  nu?: number | null;
  degree?: number;
  coef0?: number;
  decision_function_shape?: string;
  has_native_proba?: boolean;
  n_support_per_class?: number[];
  total_support_vectors?: number;
  is_linear_model?: boolean;
  epsilon?: number;
  target_idx?: number;
  resolved_target_class_idx?: number;
  resolved_target_idx?: number;
  classes?: string[];
  surface_value?: string;
  surface_proba_column?: number | null;
  boundary_threshold?: number | null;
  margin_levels?: number[];
  is_classifier?: boolean;
  is_binary?: boolean;
  is_ovo?: boolean;
  is_oneclass?: boolean;
}

export interface SvmSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: SvmSurfaceOverlays;
  telemetry: SvmSurfaceTelemetry;
}

export async function fetchSvmTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: SvmRowTrace[] }> {
  return apiFetch<{ traces: SvmRowTrace[] }>(
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

export async function fetchSvmSurfacePreview(
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
): Promise<SvmSurfacePreview> {
  return apiFetch<SvmSurfacePreview>(
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

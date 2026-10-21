/** Phase 21 — typed client for SGD trace/surface previews (same REST shape as elastic-net). */
import { apiFetch } from '../../../core/api/client';

export interface SgdTraceMechanics {
  weights: number[];
  intercept: number;
  target_idx: number;
  target_class_idx?: number;
  loss: string;
  loss_regime: string;
  penalty: string;
  alpha: number;
  l1_ratio: number;
  epsilon: number;
  learning_rate: string;
  eta0: number;
  n_iter: number;
  total_weight_updates_t: number;
  is_classifier: boolean;
  is_binary?: boolean;
  is_pseudo_proba: boolean;
  has_native_proba?: boolean;
  classes?: string[];
  class_labels?: Array<string | number>;
  decision_scores?: number[];
  raw_score?: number;
  logit?: number;
  geometric_distance?: number;
  geometric_contributions?: number[];
  w_norm?: number;
  weight_l2_norm?: number;
  inside_margin?: boolean;
  margin_status?: string;
  margin_width?: number;
  probabilities?: number[];
  predicted_class?: string | number;
  winning_class_idx?: number;
  decision_threshold?: number;
}

export interface SgdRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: SgdTraceMechanics;
}

export interface SgdSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface SgdSurfaceOverlays {
  wx?: number;
  wy?: number;
  intercept_offset?: number;
  base_intercept?: number;
  weight_row?: number[];
  loss?: string;
  loss_regime?: string;
  penalty?: string;
  alpha?: number;
  l1_ratio?: number;
  epsilon?: number;
  learning_rate?: string;
  eta0?: number;
  n_iter?: number;
  total_weight_updates_t?: number;
  is_binary?: boolean;
  classes?: string[];
  target_idx?: number;
  resolved_target_class_idx?: number;
  resolved_target_idx?: number;
  has_native_proba?: boolean;
  boundary_threshold?: number | null;
  decision_threshold?: number;
  margin_width?: number;
  corridor_half_width_score?: number;
  corridor_half_width_geometric?: number;
  w_norm?: number;
  weight_l2_norm?: number;
  gpu_formula_valid?: boolean;
  is_classifier?: boolean;
}

export interface SgdSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: SgdSurfaceOverlays;
  telemetry: SgdSurfaceTelemetry;
}

export async function fetchSgdTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: SgdRowTrace[] }> {
  return apiFetch<{ traces: SgdRowTrace[] }>(
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

export async function fetchSgdSurfacePreview(
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
): Promise<SgdSurfacePreview> {
  return apiFetch<SgdSurfacePreview>(
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

/** Phase 18 — typed client for logistic trace/surface previews (same REST shape as linear). */
import { apiFetch } from '../../../core/api/client';

export interface LogisticTraceMechanics {
  classes: string[];
  class_labels: Array<string | number>;
  is_binary: boolean;
  decision_scores: number[];
  logit: number;
  probabilities: number[];
  predicted_class: string | number;
  target_class_idx: number;
  winning_class_idx: number;
  weights: number[];
  intercept: number;
  decision_threshold: number;
  transformed_inputs?: number[];
}

export interface LogisticRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: string | number;
  actual_target: string | number | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: LogisticTraceMechanics;
}

export interface LogisticSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface LogisticSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: {
    is_binary?: boolean;
    classes?: string[];
    resolved_target_class_idx?: number;
    boundary_threshold?: number;
    coef_2d?: [number, number];
    intercept_offset?: number;
    weight_row?: number[];
    base_intercept?: number;
    gpu_formula_valid?: boolean;
    zero_influence_axes?: boolean;
  };
  telemetry: LogisticSurfaceTelemetry;
}

export async function fetchLogisticTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_class_idx?: number;
  },
): Promise<{ traces: LogisticRowTrace[] }> {
  return apiFetch<{ traces: LogisticRowTrace[] }>(
    `/api/v1/models/${encodeURIComponent(modelId)}/trace-preview`,
    {
      method: 'POST',
      body: JSON.stringify({
        rows: body.rows,
        row_indices: body.row_indices,
        feature_names: body.feature_names,
        target_idx: body.target_class_idx ?? 0,
      }),
    },
  );
}

export async function fetchLogisticSurfacePreview(
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
    target_class_idx?: number;
  },
): Promise<LogisticSurfacePreview> {
  return apiFetch<LogisticSurfacePreview>(
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
        target_idx: body.target_class_idx ?? 0,
      }),
    },
  );
}

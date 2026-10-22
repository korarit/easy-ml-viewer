/** Phase 38 — typed client for ONNX classical (ai.onnx.ml) trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface OnnxInputTensor {
  name: string;
  shape: Array<number | null>;
  dtype: string;
}

export interface OnnxTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  algorithm: string;
  model_family: string;
  operator_family: string;
  detected_operator: string;
  detected_operator_family: string;
  domain_badge: string;
  node_chain: string[];
  input_signature: OnnxInputTensor[];
  has_zipmap: boolean;
  zipmap_bypass_active: boolean;
  opset_version: number;
  classes: string[];
  class_labels: Array<string | number>;
  probabilities?: number[];
  predicted_class?: string | number;
  predicted_value?: number;
  winning_class_idx?: number;
  resolved_class_idx?: number;
  attribution_method: string;
  post_transform?: string;
  linear_same_space?: boolean;
  tree_same_space?: boolean;
  non_numeric_columns?: string[];
  n_trees?: number;
  n_nodes?: number;
  decision_path?: Array<{
    tree_id: number;
    node_id: number;
    feature_idx: number;
    threshold: number;
    mode: string;
    went_true: boolean;
  }>;
  decision_path_truncated?: boolean;
  n_path_steps?: number;
  active_leaf_count?: number;
  leaf_trace?: Array<{ tree_id: number; node_id: number; weight: number }>;
  kernel_type?: string;
  kernel_params?: number[];
  n_support_vectors?: number;
  support_vector_preview?: number[][];
  coefficients?: number[];
  intercept?: number;
  base_raw_score: number;
  final_raw_score: number;
  has_missing_values: boolean;
  top_shap_features: Array<{ rank: number; feature_idx: number; feature_name: string; shap_value: number }>;
}

export interface OnnxClassicalRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: OnnxTraceMechanics;
}

export interface OnnxClassicalSurfaceOverlays {
  model_family?: string;
  operator_family?: string;
  detected_operator?: string;
  detected_operator_family?: string;
  domain_badge?: string;
  node_chain?: string[];
  input_signature?: OnnxInputTensor[];
  has_zipmap?: boolean;
  zipmap_bypass_active?: boolean;
  opset_version?: number;
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  num_class?: number;
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
  n_trees?: number;
  n_nodes?: number;
  post_transform?: string;
  split_thresholds?: { axis_x_thresholds: number[]; axis_y_thresholds: number[] };
  weight_shape?: number[];
  coefficients_preview?: number[];
  intercept?: number;
  multi_class?: string;
  kernel_type?: string;
  kernel_params?: number[];
  n_support_vectors?: number;
}

export interface OnnxClassicalSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface OnnxClassicalSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: OnnxClassicalSurfaceOverlays;
  telemetry: OnnxClassicalSurfaceTelemetry;
}

export async function fetchOnnxClassicalTracePreview(
  modelId: string,
  body: {
    rows: Array<Array<number | string | null>>;
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: OnnxClassicalRowTrace[] }> {
  return apiFetch<{ traces: OnnxClassicalRowTrace[] }>(
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

export async function fetchOnnxClassicalSurfacePreview(
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
): Promise<OnnxClassicalSurfacePreview> {
  return apiFetch<OnnxClassicalSurfacePreview>(
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

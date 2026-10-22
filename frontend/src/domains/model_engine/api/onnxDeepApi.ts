/** Phase 39 — typed client for ONNX deep-network probe trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface OnnxDeepInputTensor {
  name: string;
  shape: Array<number | null>;
  dtype: string;
}

export interface OnnxDeepLayerTelemetry {
  layer_name: string;
  width: number;
  rendered_neurons: number;
  mean_activation: number;
  max_activation: number;
  sparsity: number;
  rendered_indices: number[];
  rendered_means: number[];
}

export interface OnnxDeepTraceMechanics {
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
  input_signature: OnnxDeepInputTensor[];
  opset_version: number;
  attribution_method: string;
  output_transform: string;
  provider: string;
  providers_available: string[];
  batch_mode: string;
  static_batch_size: number | null;
  probed_layers: string[];
  n_probed_layers: number;
  penultimate_layer: string | null;
  penultimate_width: number;
  injection_active: boolean;
  layer_telemetry: OnnxDeepLayerTelemetry[];
  latent_coords_3d: [number, number, number];
  classes: Array<string | number>;
  class_labels: Array<string | number>;
  probabilities?: number[];
  predicted_class?: string | number;
  predicted_value?: number;
  winning_class_idx?: number;
  resolved_class_idx?: number;
  base_raw_score: number;
  final_raw_score: number;
  has_missing_values: boolean;
  top_shap_features: Array<{ rank: number; feature_idx: number; feature_name: string; shap_value: number }>;
}

export interface OnnxDeepRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: OnnxDeepTraceMechanics;
}

export interface OnnxDeepSurfaceOverlays {
  model_family?: string;
  operator_family?: string;
  detected_operator?: string;
  detected_operator_family?: string;
  domain_badge?: string;
  node_chain?: string[];
  input_signature?: OnnxDeepInputTensor[];
  opset_version?: number;
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: Array<string | number>;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  num_class?: number;
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
  output_transform?: string;
  provider?: string;
  batch_mode?: string;
  static_batch_size?: number | null;
  probed_layers?: string[];
  n_probed_layers?: number;
  penultimate_layer?: string | null;
  injection_active?: boolean;
  latent_manifold_sample?: number[][];
  latent_sample_stride?: number;
}

export interface OnnxDeepSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface OnnxDeepSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: OnnxDeepSurfaceOverlays;
  telemetry: OnnxDeepSurfaceTelemetry;
}

export async function fetchOnnxDeepTracePreview(
  modelId: string,
  body: {
    rows: Array<Array<number | string | null>>;
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: OnnxDeepRowTrace[] }> {
  return apiFetch<{ traces: OnnxDeepRowTrace[] }>(
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

export async function fetchOnnxDeepSurfacePreview(
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
): Promise<OnnxDeepSurfacePreview> {
  return apiFetch<OnnxDeepSurfacePreview>(
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

/** Phase 36 — typed client for MLP synapse-brain trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface MlpSynapseEdge {
  fromLayer: number;
  u: number;
  v: number;
  flux: number;
}

export interface MlpLayerTelemetry {
  layer_index: number;
  total_neurons: number;
  rendered_neurons: number;
  active_neurons: number;
  dead_neurons: number;
  dead_ratio_mean: number;
  max_activation: number;
}

export interface MlpTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  activation: string;
  out_activation: string;
  hidden_layer_sizes: number[];
  layer_dimensions: number[];
  n_layers: number;
  n_hidden_layers: number;
  classes: string[];
  class_labels: Array<string | number>;
  num_class: number;
  model_family: string;
  rendered_node_indices: number[][];
  layer_activations_sample: number[][];
  dead_neuron_masks: boolean[][];
  dead_ratios: number[][];
  dead_counts_per_layer: number[];
  total_dead_neurons: number;
  layer_telemetry: MlpLayerTelemetry[];
  synapse_edges: MlpSynapseEdge[];
  synapse_edges_per_layer_counts: number[];
  max_synapses_per_layer_pair: number;
  max_rendered_neurons_per_layer: number;
  probabilities: number[];
  predicted_class: string | number;
  linked_prediction: number | null;
  winning_class_idx: number;
  resolved_class_idx: number;
  base_raw_score: number;
  final_raw_score: number;
  has_missing_values: boolean;
  top_shap_features: Array<{ rank: number; feature_idx: number; feature_name: string; shap_value: number }>;
  attribution_method: string;
}

export interface MlpRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: MlpTraceMechanics;
}

export interface MlpSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  activation?: string;
  out_activation?: string;
  hidden_layer_sizes?: number[];
  layer_dimensions?: number[];
  n_layers?: number;
  loss?: string;
  final_loss?: number;
  n_iter?: number;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  num_class?: number;
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface MlpSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface MlpSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: MlpSurfaceOverlays;
  telemetry: MlpSurfaceTelemetry;
}

export async function fetchMlpTracePreview(
  modelId: string,
  body: {
    rows: Array<Array<number | string>>;
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: MlpRowTrace[] }> {
  return apiFetch<{ traces: MlpRowTrace[] }>(
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

export async function fetchMlpSurfacePreview(
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
): Promise<MlpSurfacePreview> {
  return apiFetch<MlpSurfacePreview>(
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

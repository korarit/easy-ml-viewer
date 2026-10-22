/** Phase 28 — typed client for random-forest consensus trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface ForestPathStep {
  node_id: number;
  next_node_id: number;
  feature_idx: number;
  feature_name: string;
  threshold: number;
  went_left: boolean;
  delta: number;
}

export interface ForestTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  classes: string[];
  class_labels: Array<string | number>;
  model_family: string;
  n_estimators_total: number;
  n_estimators_exported: number;
  consensus_ratio: number;
  inter_tree_mean: number;
  inter_tree_std: number;
  tree_predictions_sample: number[];
  leaf_indices_sample: number[];
  mean_path_length: number;
  first_tree_active_path: ForestPathStep[];
  first_tree_leaf_id: number;
  oob_score: number | null;
  n_jobs: number | null;
  max_depth: number | null;
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface ForestRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: ForestTraceMechanics;
}

export interface ForestSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  n_estimators_total?: number;
  n_estimators_exported?: number;
  first_tree_topology?: { node_count: number; max_depth: number; n_leaves: number };
  mean_max_depth?: number;
  feature_importances?: number[];
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface ForestSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface ForestSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: ForestSurfaceOverlays;
  telemetry: ForestSurfaceTelemetry;
}

export async function fetchForestTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: ForestRowTrace[] }> {
  return apiFetch<{ traces: ForestRowTrace[] }>(
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

export async function fetchForestSurfacePreview(
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
): Promise<ForestSurfacePreview> {
  return apiFetch<ForestSurfacePreview>(
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

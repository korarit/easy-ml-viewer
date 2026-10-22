/** Phase 34 — typed client for LightGBM leaf-wise TreeSHAP trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface LgbmTopShapFeature {
  rank: number;
  feature_idx: number;
  feature_name: string;
  shap_value: number;
}

export interface LgbmSplitLine {
  round: number;
  feature_idx: number;
  gain: number;
  count: number;
}

export interface LgbmDumpNode {
  id: number;
  leaf_index?: number;
  leaf_value?: number;
  leaf_count?: number;
  split_index?: number;
  split_feature_idx?: number;
  threshold?: number | null;
  categories?: number[];
  decision_type?: string;
  default_left?: boolean;
  gain?: number;
  count?: number;
  depth?: number;
  yes?: number;
  no?: number;
}

export interface LgbmTreeTopology {
  round: number;
  num_leaves: number;
  shrinkage: number;
  nodes: LgbmDumpNode[];
  leaf_depths: Record<number, number>;
}

export interface LgbmDepthStat {
  round: number;
  min_leaf_depth: number;
  max_leaf_depth: number;
  num_leaves: number;
}

export interface LgbmTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  objective: string;
  num_class: number;
  classes: string[];
  class_labels: Array<string | number>;
  model_family: string;
  learning_rate: number;
  n_iterations: number;
  n_boost_rounds: number;
  n_rounds_total: number;
  n_trees_total: number;
  num_leaves: number;
  negated_target_zero_view: boolean;
  has_missing_values: boolean;
  base_raw_score: number;
  final_raw_score: number;
  sampled_iteration_indices: number[];
  staged_trajectory: number[];
  leaf_indices_sample: number[];
  active_leaf_depth: number | null;
  top_shap_features: LgbmTopShapFeature[];
  split_lines_sample: LgbmSplitLine[];
  categorical_feature_idx: number[];
  feature_importances: Record<string, number[]>;
  tree_topology_rounds: number[];
  tree_depth_stats: LgbmDepthStat[];
  probabilities: number[];
  predicted_class: string | number;
  linked_prediction: number | null;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface LgbmRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: LgbmTraceMechanics;
}

export interface LgbmSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  objective?: string;
  num_class?: number;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  learning_rate?: number;
  n_iterations?: number;
  n_boost_rounds?: number;
  n_rounds_total?: number;
  n_trees_total?: number;
  feature_importances?: Record<string, number[]>;
  tree_topology_sample?: LgbmTreeTopology[];
  tree_topology_rounds?: number[];
  tree_depth_stats?: LgbmDepthStat[];
  split_lines_sample?: LgbmSplitLine[];
  categorical_feature_idx?: number[];
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface LgbmSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface LgbmSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: LgbmSurfaceOverlays;
  telemetry: LgbmSurfaceTelemetry;
}

export async function fetchLgbmTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: LgbmRowTrace[] }> {
  return apiFetch<{ traces: LgbmRowTrace[] }>(
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

export async function fetchLgbmSurfacePreview(
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
): Promise<LgbmSurfacePreview> {
  return apiFetch<LgbmSurfacePreview>(
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

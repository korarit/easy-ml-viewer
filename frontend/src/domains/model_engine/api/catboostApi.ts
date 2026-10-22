/** Phase 35 — typed client for CatBoost oblivious-tree SHAP trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface CatboostTopShapFeature {
  rank: number;
  feature_idx: number;
  feature_name: string;
  shap_value: number;
}

export interface CatboostLeafBadge {
  tree: number;
  leaf: number;
  bitmask: string;
}

export interface CatboostSplitLine {
  round: number;
  depth: number;
  feature_idx: number;
  gain: number;
  kind: string;
  threshold: number | null;
  category: string | null;
}

export interface CatboostObliviousLevel {
  depth: number;
  feature_name: string;
  feature_idx: number;
  kind: string;
  threshold: number | null;
  category: string | null;
  gain: number;
}

export interface CatboostTreeTopology {
  round: number;
  depth: number;
  num_leaves: number;
  levels: CatboostObliviousLevel[];
  leaf_values: number[];
  leaf_bitmasks: string[];
}

export interface CatboostTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  loss_function: string;
  objective: string;
  num_class: number;
  classes: string[];
  class_labels: Array<string | number>;
  model_family: string;
  learning_rate: number;
  depth: number;
  n_iterations: number;
  n_boost_rounds: number;
  n_rounds_total: number;
  n_trees_total: number;
  num_leaves: number;
  negated_target_zero_view: boolean;
  has_missing_values: boolean;
  native_categorical_features: string[];
  categorical_feature_idx: number[];
  top_feature_is_categorical: boolean;
  ctr_inspector: boolean;
  base_raw_score: number;
  final_raw_score: number;
  sampled_iteration_indices: number[];
  staged_trajectory: number[];
  leaf_indices_sample: number[];
  leaf_bitmask_badges: CatboostLeafBadge[];
  active_leaf_depth: number | null;
  top_shap_features: CatboostTopShapFeature[];
  split_lines_sample: CatboostSplitLine[];
  feature_importances: Record<string, number[]>;
  tree_topology_rounds: number[];
  oblivious_tree_topology: CatboostTreeTopology[];
  probabilities: number[];
  predicted_class: string | number;
  linked_prediction: number | null;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface CatboostRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: CatboostTraceMechanics;
}

export interface CatboostSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  loss_function?: string;
  objective?: string;
  num_class?: number;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  learning_rate?: number;
  depth?: number;
  n_iterations?: number;
  n_boost_rounds?: number;
  n_rounds_total?: number;
  n_trees_total?: number;
  feature_importances?: Record<string, number[]>;
  tree_topology_sample?: CatboostTreeTopology[];
  tree_topology_rounds?: number[];
  oblivious_tree_topology?: CatboostTreeTopology[];
  split_lines_sample?: CatboostSplitLine[];
  categorical_feature_idx?: number[];
  categorical_ticks_x?: string[];
  categorical_ticks_y?: string[];
  axis_x_is_categorical?: boolean;
  axis_y_is_categorical?: boolean;
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface CatboostSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface CatboostSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: CatboostSurfaceOverlays;
  telemetry: CatboostSurfaceTelemetry;
}

export async function fetchCatboostTracePreview(
  modelId: string,
  body: {
    rows: Array<Array<number | string>>;
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: CatboostRowTrace[] }> {
  return apiFetch<{ traces: CatboostRowTrace[] }>(
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

export async function fetchCatboostSurfacePreview(
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
): Promise<CatboostSurfacePreview> {
  return apiFetch<CatboostSurfacePreview>(
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

/** Phase 27 — typed client for decision-tree voxel trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface TreePathStep {
  node_id: number;
  next_node_id: number;
  feature_idx: number;
  feature_name: string;
  threshold: number;
  went_left: boolean;
  delta: number;
  impurity: number;
}

export interface TreeTopology {
  node_count: number;
  max_depth: number;
  n_leaves: number;
  exported_nodes: number;
  truncated: boolean;
  children_left: number[];
  children_right: number[];
  feature: number[];
  threshold: number[];
  impurity: number[];
  n_node_samples: number[];
  node_expectation: number[];
  depth: number[];
}

export interface TreeTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  classes: string[];
  class_labels: Array<string | number>;
  calibrated_wrapper: boolean;
  multi_output_traced_index: number | null;
  is_multi_output: boolean;
  criterion: string;
  max_depth: number;
  node_count: number;
  n_leaves: number;
  leaf_id: number;
  leaf_impurity: number;
  leaf_n_samples: number;
  leaf_expectation: number;
  active_path: TreePathStep[];
  active_path_truncated: boolean;
  path_node_ids: number[];
  tree_topology: TreeTopology;
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface TreeRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: TreeTraceMechanics;
}

export interface TreeSplitLine {
  node_id: number;
  threshold: number;
  depth: number;
  impurity: number;
  n_node_samples: number;
}

export interface TreeSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface TreeSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  calibrated_wrapper?: boolean;
  criterion?: string;
  depth_cap?: number;
  tree_topology?: TreeTopology;
  split_lines_x?: TreeSplitLine[];
  split_lines_y?: TreeSplitLine[];
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface TreeSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: TreeSurfaceOverlays;
  telemetry: TreeSurfaceTelemetry;
}

export async function fetchTreeTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: TreeRowTrace[] }> {
  return apiFetch<{ traces: TreeRowTrace[] }>(
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

export async function fetchTreeSurfacePreview(
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
): Promise<TreeSurfacePreview> {
  return apiFetch<TreeSurfacePreview>(
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

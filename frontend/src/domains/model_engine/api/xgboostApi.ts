/** Phase 33 — typed client for XGBoost TreeSHAP leaf-flow trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface XgbTopShapFeature {
  rank: number;
  feature_idx: number;
  feature_name: string;
  shap_value: number;
}

export interface XgbSplitLine {
  round: number;
  feature_idx: number;
  gain: number;
  cover: number;
}

export interface XgbDumpNode {
  id: number;
  leaf?: number;
  cover?: number;
  split_feature_idx?: number;
  split_feature?: string;
  threshold?: number | null;
  categories?: number[];
  yes?: number;
  no?: number;
  missing?: number;
  gain?: number;
}

export interface XgbTreeTopology {
  round: number;
  nodes: XgbDumpNode[];
}

export interface XgbTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_regressor: boolean;
  is_binary: boolean;
  algorithm: string;
  objective: string;
  booster_type: string;
  classes: string[];
  class_labels: Array<string | number>;
  model_family: string;
  learning_rate: number;
  n_boost_rounds: number;
  n_rounds_total: number;
  n_parallel_trees: number;
  negated_target_zero_view: boolean;
  has_missing_values: boolean;
  base_raw_score: number;
  final_raw_score: number;
  sampled_round_indices: number[];
  staged_trajectory: number[];
  leaf_indices_sample: number[];
  top_shap_features: XgbTopShapFeature[];
  split_lines_sample: XgbSplitLine[];
  categorical_feature_idx: number[];
  feature_importances: Record<string, number[]>;
  tree_topology_rounds: number[];
  probabilities: number[];
  predicted_class: string | number;
  linked_prediction: number | null;
  winning_class_idx: number;
  resolved_class_idx: number;
}

export interface XgbRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: XgbTraceMechanics;
}

export interface XgbSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  algorithm?: string;
  objective?: string;
  booster_type?: string;
  model_family?: string;
  is_classifier?: boolean;
  is_regressor?: boolean;
  is_binary?: boolean;
  learning_rate?: number;
  n_boost_rounds?: number;
  n_rounds_total?: number;
  feature_importances?: Record<string, number[]>;
  tree_topology_sample?: XgbTreeTopology[];
  tree_topology_rounds?: number[];
  split_lines_sample?: XgbSplitLine[];
  categorical_feature_idx?: number[];
  regression_z_range?: [number, number];
  boundary_threshold?: number | null;
}

export interface XgbSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface XgbSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: XgbSurfaceOverlays;
  telemetry: XgbSurfaceTelemetry;
}

export async function fetchXgbTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: XgbRowTrace[] }> {
  return apiFetch<{ traces: XgbRowTrace[] }>(
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

export async function fetchXgbSurfacePreview(
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
): Promise<XgbSurfacePreview> {
  return apiFetch<XgbSurfacePreview>(
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

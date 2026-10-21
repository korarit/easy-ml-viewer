/** Phase 22 — typed client for KNN radar trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface KnnVoteEntry {
  label: string;
  count: number;
  weight_share: number;
}

export interface KnnReferenceCloud {
  original_indices: number[];
  coordinates: number[][];
  targets: Array<number | string | null>;
  index_map: Record<string, number>;
  total_reference_points: number;
  mandatory_count: number;
}

export interface KnnTraceMechanics {
  n_neighbors: number;
  n_neighbors_effective: number;
  radius: number | null;
  weights: string;
  algorithm: string;
  leaf_size: number;
  effective_metric: string;
  p: number;
  radar_geometry_shape: 'sphere' | 'octahedron' | 'cube';
  radar_radius: number;
  neighbor_distances: number[];
  neighbor_original_indices: number[];
  neighbor_local_indices: number[];
  neighbor_labels?: Array<string | number>;
  neighbor_targets?: number[];
  neighbor_weights: number[];
  vote_tally?: KnnVoteEntry[];
  weighted_neighbor_mean?: number;
  display_capped: boolean;
  spider_display_cap: number;
  is_outlier: boolean;
  outlier_label?: number | string | null;
  is_radius: boolean;
  is_classifier: boolean;
  classes?: string[];
  class_labels?: Array<string | number>;
  has_native_proba?: boolean;
  probabilities?: number[];
  predicted_class?: string | number;
  total_reference_points: number;
  mandatory_count: number;
  reference_cloud: KnnReferenceCloud;
  target_idx: number;
  target_class_idx: number;
  active_target_idx?: number;
  n_targets?: number;
  is_multioutput?: boolean;
}

export interface KnnRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: KnnTraceMechanics;
}

export interface KnnSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface KnnSurfaceOverlays {
  n_neighbors?: number;
  radius?: number | null;
  weights?: string;
  algorithm?: string;
  leaf_size?: number;
  effective_metric?: string;
  p?: number;
  radar_geometry_shape?: 'sphere' | 'octahedron' | 'cube';
  total_reference_points?: number;
  is_radius?: boolean;
  target_idx?: number;
  resolved_target_class_idx?: number;
  resolved_target_idx?: number;
  active_target_idx?: number;
  n_targets?: number;
  is_multioutput?: boolean;
  classes?: string[];
  boundary_threshold?: number | null;
  is_classifier?: boolean;
}

export interface KnnSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: KnnSurfaceOverlays;
  telemetry: KnnSurfaceTelemetry;
}

export async function fetchKnnTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: KnnRowTrace[] }> {
  return apiFetch<{ traces: KnnRowTrace[] }>(
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

export async function fetchKnnSurfacePreview(
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
): Promise<KnnSurfacePreview> {
  return apiFetch<KnnSurfacePreview>(
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

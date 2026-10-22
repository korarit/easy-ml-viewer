/** Phase 26 — typed client for LDA/QDA ellipsoid trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export type DiscriminantKind = 'lda' | 'qda';

export interface Ellipsoid2DSpec {
  class_idx: number;
  center_xy: [number, number];
  radii_1sigma: [number, number];
  rotation_angle_rad: number;
  eigenvectors_2d: number[][];
}

export interface DiscriminantTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_binary: boolean;
  discriminant: DiscriminantKind;
  algorithm: string;
  classes: string[];
  class_labels: Array<string | number>;
  priors: number[];
  log_priors: number[];
  means: number[][];
  log_determinants: number[];
  solver: string;
  reg_param: number;
  store_covariance: boolean;
  is_covariance_approximated: boolean;
  decision_scores: number[];
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
  mahalanobis_distances: number[];
  euclidean_distances: number[];
  paradox_detected: boolean;
  nearest_mahalanobis_class: number;
  nearest_euclidean_class: number;
}

export interface DiscriminantRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: DiscriminantTraceMechanics;
}

export interface DiscriminantSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface DiscriminantSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  discriminant?: DiscriminantKind;
  algorithm?: string;
  solver?: string;
  reg_param?: number;
  is_covariance_approximated?: boolean;
  ellipsoids_2d?: Ellipsoid2DSpec[];
  means_2d?: number[][];
  boundary_threshold?: number | null;
  is_classifier?: boolean;
  is_binary?: boolean;
}

export interface DiscriminantSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: DiscriminantSurfaceOverlays;
  telemetry: DiscriminantSurfaceTelemetry;
}

export async function fetchDiscriminantTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: DiscriminantRowTrace[] }> {
  return apiFetch<{ traces: DiscriminantRowTrace[] }>(
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

export async function fetchDiscriminantSurfacePreview(
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
): Promise<DiscriminantSurfacePreview> {
  return apiFetch<DiscriminantSurfacePreview>(
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

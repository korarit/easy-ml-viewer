/** Phase 24 — typed client for GaussianNB bell-hills trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface GnbTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_binary: boolean;
  classes: string[];
  class_labels: Array<string | number>;
  class_priors: number[];
  class_counts: number[];
  log_priors: number[];
  has_native_proba: boolean;
  is_pseudo_proba: boolean;
  joint_log_probs: number[];
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
  var_smoothing: number;
  epsilon: number;
  winning_class_z_scores: number[];
  per_class_min_z: number[];
  min_z_score: number;
  is_low_density_outlier: boolean;
  outlier_threshold_sigma: number;
}

export interface GnbRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: GnbTraceMechanics;
}

export interface GnbSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface GnbSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  class_priors?: number[];
  bell_hills_per_class?: Record<string, number[]>;
  hill_class_order?: number[];
  hills_truncated_to_top5?: boolean;
  peaks_2d?: number[][];
  stds_2d?: number[][];
  peak_densities?: Record<string, number>;
  var_smoothing?: number;
  boundary_threshold?: number | null;
  is_classifier?: boolean;
  is_binary?: boolean;
}

export interface GnbSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: GnbSurfaceOverlays;
  telemetry: GnbSurfaceTelemetry;
}

export async function fetchGnbTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: GnbRowTrace[] }> {
  return apiFetch<{ traces: GnbRowTrace[] }>(
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

export async function fetchGnbSurfacePreview(
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
): Promise<GnbSurfacePreview> {
  return apiFetch<GnbSurfacePreview>(
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

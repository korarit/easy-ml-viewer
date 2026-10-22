/** Phase 25 — typed client for discrete-NB tower trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export type DiscreteNbVariant = 'multinomial' | 'bernoulli' | 'complement' | 'categorical';

export interface DiscreteNbTraceMechanics {
  target_idx: number;
  target_class_idx: number;
  is_classifier: boolean;
  is_binary: boolean;
  nb_variant: DiscreteNbVariant;
  algorithm: string;
  classes: string[];
  class_labels: Array<string | number>;
  class_counts: number[];
  log_priors: number[];
  class_log_prior: number[];
  alpha: number;
  binarize: number | null;
  requires_non_negative_input: boolean;
  non_negative_clamped: boolean;
  joint_log_probs: number[];
  probabilities: number[];
  predicted_class: string | number;
  winning_class_idx: number;
  resolved_class_idx: number;
  presence_flags: Record<string, 'present' | 'absent' | 'unknown'>;
  unseen_category_clamped: boolean;
  contributions_pruned: boolean;
  pruned_feature_count: number;
  pruned_remainder_sum: number;
  tower_heights: number[];
}

export interface DiscreteNbRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: DiscreteNbTraceMechanics;
}

export interface DiscreteNbSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface DiscreteNbSurfaceOverlays {
  target_idx?: number;
  resolved_target_class_idx?: number;
  classes?: string[];
  nb_variant?: DiscreteNbVariant;
  algorithm?: string;
  alpha?: number;
  binarize?: number | null;
  clamped_non_negative?: boolean;
  non_negative_domain?: boolean;
  boundary_threshold?: number | null;
  is_classifier?: boolean;
  is_binary?: boolean;
}

export interface DiscreteNbSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: DiscreteNbSurfaceOverlays;
  telemetry: DiscreteNbSurfaceTelemetry;
}

export async function fetchDiscreteNbTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: DiscreteNbRowTrace[] }> {
  return apiFetch<{ traces: DiscreteNbRowTrace[] }>(
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

export async function fetchDiscreteNbSurfacePreview(
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
): Promise<DiscreteNbSurfacePreview> {
  return apiFetch<DiscreteNbSurfacePreview>(
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

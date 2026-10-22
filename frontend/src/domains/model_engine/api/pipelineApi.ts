/** Phase 37 — typed client for sklearn Pipeline tunnel trace/surface previews. */
import { apiFetch } from '../../../core/api/client';

export interface PipelineStepSummary {
  step_index: number;
  step_name: string;
  kind: string;
  n_in: number;
  n_out: number;
  input_columns: string[];
  output_columns: string[];
}

export interface PipelineTraceMechanics {
  model_family: string;
  delegated_estimator: string;
  inner_model_family?: string;
  pipeline_steps: PipelineStepSummary[];
  pipeline_step_names: string[];
  pipeline_n_steps: number;
  raw_feature_names: string[];
  transformed_feature_names: string[];
  n_transformed_features: number;
  feature_lineage: {
    raw_names: string[];
    transformed_names: string[];
    mapping: Record<string, string[]>;
  };
  raw_row_values: Record<string, string | number | null>;
  transformed_row_top16: Record<string, number>;
  imputed_columns: string[];
  has_imputed_values: boolean;
  oov_columns: string[];
  has_oov_category: boolean;
  aggregated_raw_contributions?: Record<string, number>;
  waterfall_modes: string[];
  probabilities?: number[];
  predicted_class?: string | number;
  target_idx?: number;
  is_classifier?: boolean;
}

export interface PipelineRowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number | string;
  actual_target: number | string | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: PipelineTraceMechanics;
}

export interface PipelineSurfaceOverlays {
  model_family?: string;
  delegated_estimator?: string;
  delegated_visual_type?: string;
  coordinate_space?: string;
  coordinate_space_toggle?: string[];
  raw_feature_names?: string[];
  raw_axis_x?: string;
  raw_axis_y?: string;
  n_transformed_features?: number;
  feature_lineage?: Record<string, string[]>;
  pipeline_step_names?: string[];
  pipeline_n_steps?: number;
  categorical_raw_columns?: string[];
  is_classifier?: boolean;
}

export interface PipelineSurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface PipelineSurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  extra_overlays?: PipelineSurfaceOverlays;
  telemetry: PipelineSurfaceTelemetry;
}

export async function fetchPipelineTracePreview(
  modelId: string,
  body: {
    rows: Array<Array<number | string | null>>;
    row_indices: number[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: PipelineRowTrace[] }> {
  return apiFetch<{ traces: PipelineRowTrace[] }>(
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

export async function fetchPipelineSurfacePreview(
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
): Promise<PipelineSurfacePreview> {
  return apiFetch<PipelineSurfacePreview>(
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

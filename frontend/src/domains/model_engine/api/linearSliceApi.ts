/** Phase 04 — typed client for trace-preview / surface-preview endpoints. */
import { apiFetch } from '../../../core/api/client';

export interface RowTrace {
  row_index: number;
  base_value: number;
  final_prediction: number;
  actual_target: number | null;
  residual_error: number | null;
  feature_contributions: Record<string, number>;
  extra_mechanics: {
    transformed_inputs?: number[];
    weights?: number[];
    target_idx?: number;
  };
}

export interface SurfaceTelemetry {
  grid_points: number;
  resolution: number;
  z_min: number;
  z_max: number;
  compute_ms: number;
}

export interface SurfacePreview {
  visual_type: string;
  resolution: number;
  x_grid: number[];
  y_grid: number[];
  z_values: number[];
  telemetry: SurfaceTelemetry;
}

export async function fetchTracePreview(
  modelId: string,
  body: {
    rows: number[][];
    row_indices: number[];
    actual_targets?: (number | null)[];
    feature_names?: string[];
    target_idx?: number;
  },
): Promise<{ traces: RowTrace[] }> {
  return apiFetch<{ traces: RowTrace[] }>(
    `/api/v1/models/${encodeURIComponent(modelId)}/trace-preview`,
    { method: 'POST', body: JSON.stringify(body) },
  );
}

export async function fetchSurfacePreview(
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
): Promise<SurfacePreview> {
  return apiFetch<SurfacePreview>(
    `/api/v1/models/${encodeURIComponent(modelId)}/surface-preview`,
    { method: 'POST', body: JSON.stringify(body) },
  );
}

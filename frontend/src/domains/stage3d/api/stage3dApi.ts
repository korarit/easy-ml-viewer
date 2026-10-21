/** Phase 08 — typed client for the 3D stage initialize-scene endpoint. */
import { apiFetch } from '../../../core/api/client';

export interface AxisScaleMeta {
  axis_name: string;
  feature_index: number | null;
  raw_min: number;
  raw_max: number;
  tick_world_positions: number[];
  tick_raw_labels: number[];
}

export interface TrajectoryPoint3D {
  frame_index: number;
  original_row_index: number;
  world_x: number;
  world_y: number;
  world_z_pred: number;
  world_z_actual: number | null;
  raw_x: number;
  raw_y: number;
  raw_z_pred: number;
  raw_z_actual: number | null;
  residual_error: number | null;
}

export interface InitializeSceneResponse {
  axes: { x: AxisScaleMeta; y: AxisScaleMeta; z: AxisScaleMeta };
  surface: { visual_type: string; resolution: number; world_vertices_xyz: number[] };
  trajectory_points: TrajectoryPoint3D[];
  dataset_row_count: number;
  task_type: string;
}

export interface InitializeSceneParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  axisXFeatureIndex: number;
  axisYFeatureIndex: number;
  selectedRowIndices: number[];
  gridResolution?: number;
}

export async function fetchInitializeScene(
  params: InitializeSceneParams,
  signal?: AbortSignal,
): Promise<InitializeSceneResponse> {
  return apiFetch<InitializeSceneResponse>('/api/v1/stage3d/initialize-scene', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      axis_x_feature_index: params.axisXFeatureIndex,
      axis_y_feature_index: params.axisYFeatureIndex,
      selected_row_indices: params.selectedRowIndices,
      grid_resolution: params.gridResolution ?? 50,
    }),
    signal,
  });
}

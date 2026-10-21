/** Phase 09 — typed client for POST /api/v1/projection/surface-mesh. */
import { apiFetch } from '../../../core/api/client';
import type {
  LodLevel,
  PinningMode,
  ProjectionMode,
  SurfaceMeshPayload,
  ZMode,
} from '../types/projection.types';

interface SurfaceMeshResponse {
  grid_resolution: number;
  x_feature: string;
  y_feature: string;
  x_range: [number, number];
  y_range: [number, number];
  z_range: [number, number];
  z_heights_world: number[];
  raw_predictions: number[];
  point_positions_world: Array<[number, number, number]>;
  point_row_indices: number[];
  trajectory_points: SurfaceMeshPayload['trajectoryPoints'];
  pca_variance_ratio: [number, number, number] | null;
  pca_loadings: number[][] | null;
  resolved_z_mode: ZMode;
  has_constant_axis_warning: boolean;
  computation_time_ms: number;
  dataset_row_count: number;
}

export interface FetchSurfaceParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  mode: ProjectionMode;
  xFeature: string;
  yFeature: string;
  zMode: ZMode;
  targetClassIndex: number;
  pinningMode: PinningMode;
  lodLevel: LodLevel;
  currentFrameIndex: number | null;
  useRobustRange: boolean;
  selectedRowIndices: number[];
}

export async function fetchSurfaceMesh(
  params: FetchSurfaceParams,
  signal?: AbortSignal,
): Promise<SurfaceMeshPayload> {
  const data = await apiFetch<SurfaceMeshResponse>('/api/v1/projection/surface-mesh', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      mode: params.mode,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      z_mode: params.zMode,
      target_class_index: params.targetClassIndex,
      pinning_mode: params.pinningMode,
      lod_level: params.lodLevel,
      current_frame_index: params.currentFrameIndex,
      use_robust_range: params.useRobustRange,
      selected_row_indices: params.selectedRowIndices,
    }),
    signal,
  });
  return {
    gridResolution: data.grid_resolution,
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    xRange: data.x_range,
    yRange: data.y_range,
    zRange: data.z_range,
    zHeightsWorld: new Float32Array(data.z_heights_world),
    rawPredictions: new Float32Array(data.raw_predictions),
    pointPositionsWorld: data.point_positions_world,
    pointRowIndices: data.point_row_indices,
    trajectoryPoints: data.trajectory_points,
    pcaVarianceRatio: data.pca_variance_ratio,
    pcaLoadings: data.pca_loadings,
    resolvedZMode: data.resolved_z_mode,
    hasConstantAxisWarning: data.has_constant_axis_warning,
    computationTimeMs: data.computation_time_ms,
    datasetRowCount: data.dataset_row_count,
  };
}

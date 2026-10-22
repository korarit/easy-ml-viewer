/** Phase 09 — typed client for POST /api/v1/projection/surface-mesh. */
import { apiFetch } from '../../../core/api/client';
import type {
  Contour2DPayload,
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

interface Contour2DResponse {
  grid_resolution: number;
  x_feature: string;
  y_feature: string;
  x_range_raw: [number, number];
  y_range_raw: [number, number];
  prediction_task: 'classification' | 'regression';
  classes: Array<string | number>;
  heatmap_min: number;
  heatmap_max: number;
  heatmap_quantized: number[];
  contours: Array<{
    threshold: number;
    class_index: number;
    kind: 'decision' | 'confidence' | 'elevation';
    line_segments: Array<[number, number, number, number]>;
  }>;
  response_curve_1d: { x_values_raw: number[]; y_pred: number[] } | null;
  linear_2d: {
    family: string;
    task: 'classification' | 'regression';
    x_feature: string;
    y_feature: string;
    w1: number;
    w2: number;
    b_eff: number;
    decision_threshold: number;
    normal_vector: [number, number];
    corridor_width: number | null;
    margin_level: number | null;
    is_x_zeroed: boolean;
    is_y_zeroed: boolean;
    both_zeroed: boolean;
    sparsity_ratio: number;
    l1_ratio: number | null;
    alpha: number | null;
    penalty: string | null;
    slope_w1: number | null;
    x_sweep_intercept: number | null;
    formula: string;
    is_multiclass: boolean;
    n_classes: number;
    lines: Array<{
      kind: 'decision' | 'margin_pos' | 'margin_neg' | 'confidence' | 'pairwise' | 'slope';
      level: number;
      segment_world: [number, number, number, number] | null;
      segment_raw: [number, number, number, number] | null;
      label: string;
      class_a: number | null;
      class_b: number | null;
    }>;
  } | null;
  computation_time_ms: number;
}

export interface FetchContourParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  xFeature: string;
  yFeature: string;
  pinningMode: PinningMode;
  currentFrameIndex: number | null;
  useRobustRange: boolean;
  gridResolution: number;
  selectedRowIndices: number[];
}

/** Phase 41 — typed client for POST /api/v1/projection/contour-2d. */
export async function fetchContour2D(
  params: FetchContourParams,
  signal?: AbortSignal,
): Promise<Contour2DPayload> {
  const data = await apiFetch<Contour2DResponse>('/api/v1/projection/contour-2d', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      pinning_mode: params.pinningMode,
      current_frame_index: params.currentFrameIndex,
      use_robust_range: params.useRobustRange,
      grid_resolution: params.gridResolution,
      selected_row_indices: params.selectedRowIndices,
    }),
    signal,
  });
  return {
    gridResolution: data.grid_resolution,
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    xRangeRaw: data.x_range_raw,
    yRangeRaw: data.y_range_raw,
    predictionTask: data.prediction_task,
    classes: data.classes,
    heatmapMin: data.heatmap_min,
    heatmapMax: data.heatmap_max,
    heatmapQuantized: Uint8Array.from(data.heatmap_quantized),
    contours: data.contours.map((c) => ({
      threshold: c.threshold,
      classIndex: c.class_index,
      kind: c.kind,
      lineSegments: c.line_segments,
    })),
    responseCurve1D: data.response_curve_1d
      ? { xValuesRaw: data.response_curve_1d.x_values_raw, yPred: data.response_curve_1d.y_pred }
      : null,
    linear2d: data.linear_2d
      ? {
          family: data.linear_2d.family,
          task: data.linear_2d.task,
          xFeature: data.linear_2d.x_feature,
          yFeature: data.linear_2d.y_feature,
          w1: data.linear_2d.w1,
          w2: data.linear_2d.w2,
          bEff: data.linear_2d.b_eff,
          decisionThreshold: data.linear_2d.decision_threshold,
          normalVector: data.linear_2d.normal_vector,
          corridorWidth: data.linear_2d.corridor_width,
          marginLevel: data.linear_2d.margin_level,
          isXZeroed: data.linear_2d.is_x_zeroed,
          isYZeroed: data.linear_2d.is_y_zeroed,
          bothZeroed: data.linear_2d.both_zeroed,
          sparsityRatio: data.linear_2d.sparsity_ratio,
          l1Ratio: data.linear_2d.l1_ratio,
          alpha: data.linear_2d.alpha,
          penalty: data.linear_2d.penalty,
          slopeW1: data.linear_2d.slope_w1,
          xSweepIntercept: data.linear_2d.x_sweep_intercept,
          formula: data.linear_2d.formula,
          isMulticlass: data.linear_2d.is_multiclass,
          nClasses: data.linear_2d.n_classes,
          lines: data.linear_2d.lines.map((l) => ({
            kind: l.kind,
            level: l.level,
            segmentWorld: l.segment_world,
            segmentRaw: l.segment_raw,
            label: l.label,
            classA: l.class_a,
            classB: l.class_b,
          })),
        }
      : null,
    computationTimeMs: data.computation_time_ms,
  };
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

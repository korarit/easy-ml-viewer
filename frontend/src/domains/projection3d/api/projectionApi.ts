/** Phase 09 — typed client for POST /api/v1/projection/surface-mesh. */
import { apiFetch } from '../../../core/api/client';
import type {
  Contour2DPayload,
  KnnFrame2DPayload,
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
    kind: 'decision' | 'confidence' | 'elevation' | 'margin';
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
  svm_2d: {
    model_type: string;
    kernel: string;
    gamma: number | null;
    c: number | null;
    degree: number;
    coef0: number;
    is_regressor: boolean;
    is_binary: boolean;
    n_classes: number;
    n_support_total: number;
    is_linear_no_sv: boolean;
    support_vectors_2d: Array<{
      train_index: number;
      x_world: number;
      y_world: number;
      dual_coef: number;
      halo_scale: number;
      status: 'on_margin' | 'violates_margin' | 'support';
    }>;
    halos_truncated: boolean;
  } | null;
  knn_2d: {
    is_knn_model: boolean;
    model_type: string;
    k: number;
    weights_mode: string;
    metric: string;
    is_radius_model: boolean;
    radius: number | null;
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
    svm2d: data.svm_2d
      ? {
          modelType: data.svm_2d.model_type,
          kernel: data.svm_2d.kernel,
          gamma: data.svm_2d.gamma,
          c: data.svm_2d.c,
          degree: data.svm_2d.degree,
          coef0: data.svm_2d.coef0,
          isRegressor: data.svm_2d.is_regressor,
          isBinary: data.svm_2d.is_binary,
          nClasses: data.svm_2d.n_classes,
          nSupportTotal: data.svm_2d.n_support_total,
          isLinearNoSv: data.svm_2d.is_linear_no_sv,
          halos: data.svm_2d.support_vectors_2d.map((h) => ({
            trainIndex: h.train_index,
            xWorld: h.x_world,
            yWorld: h.y_world,
            dualCoef: h.dual_coef,
            haloScale: h.halo_scale,
            status: h.status,
          })),
          halosTruncated: data.svm_2d.halos_truncated,
        }
      : null,
    knn2d: data.knn_2d
      ? {
          isKnnModel: data.knn_2d.is_knn_model,
          modelType: data.knn_2d.model_type,
          k: data.knn_2d.k,
          weightsMode: data.knn_2d.weights_mode,
          metric: data.knn_2d.metric,
          isRadiusModel: data.knn_2d.is_radius_model,
          radius: data.knn_2d.radius,
        }
      : null,
    computationTimeMs: data.computation_time_ms,
  };
}

export interface FetchKnnFrameParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  xFeature: string;
  yFeature: string;
  rowIndex: number;
}

interface KnnFrame2DResponse {
  x_feature: string;
  y_feature: string;
  row_index: number;
  query_world: [number, number];
  radar_radius_raw: number;
  radar_radius_world: [number, number];
  neighbors: Array<{
    train_index: number;
    x_world: number;
    y_world: number;
    distance_full: number;
    distance_2d: number;
    vote_weight: number;
    rank: number;
    label: string | number | null;
  }>;
  votes: Record<string, number>;
  predicted: string | number | null;
  k: number;
  weights_mode: string;
  metric: string;
  is_radius_model: boolean;
  outlier: boolean;
  display_capped: boolean;
  total_neighbors: number;
  computation_time_ms: number;
}

/** Phase 43 — typed client for POST /api/v1/projection/knn-frame-2d. */
export async function fetchKnnFrame2D(
  params: FetchKnnFrameParams,
  signal?: AbortSignal,
): Promise<KnnFrame2DPayload> {
  const data = await apiFetch<KnnFrame2DResponse>('/api/v1/projection/knn-frame-2d', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      row_index: params.rowIndex,
    }),
    signal,
  });
  return {
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    rowIndex: data.row_index,
    queryWorld: data.query_world,
    radarRadiusRaw: data.radar_radius_raw,
    radarRadiusWorld: data.radar_radius_world,
    neighbors: data.neighbors.map((n) => ({
      trainIndex: n.train_index,
      xWorld: n.x_world,
      yWorld: n.y_world,
      distanceFull: n.distance_full,
      distance2d: n.distance_2d,
      voteWeight: n.vote_weight,
      rank: n.rank,
      label: n.label,
    })),
    votes: data.votes,
    predicted: data.predicted,
    k: data.k,
    weightsMode: data.weights_mode,
    metric: data.metric,
    isRadiusModel: data.is_radius_model,
    outlier: data.outlier,
    displayCapped: data.display_capped,
    totalNeighbors: data.total_neighbors,
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

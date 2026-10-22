/** Phase 09 — projection3d shared types (mirrors SurfaceMesh3DEntity). */

export type ProjectionMode = 'axis_explorer' | 'pca_3d';
export type LodLevel = 'low' | 'standard' | 'high';
export type PinningMode = 'median' | 'mean' | 'zero' | 'current_row';
export type ZMode = 'prediction' | 'probability' | 'residual_error' | 'feature_3';
export type ColormapName = 'viridis' | 'plasma' | 'coolwarm' | 'rdylgn' | 'neon';

export const LOD_RESOLUTIONS: Record<LodLevel, number> = { low: 30, standard: 60, high: 100 };

export interface AvailableFeature {
  name: string;
  featureIndex: number;
  importance: number | null;
}

/** Phase 41 — one vector isoline set (mirrors ContourLevel2D). */
export interface ContourLevel2D {
  threshold: number;
  classIndex: number;
  /** Phase 43 adds 'margin' (SVM f = ±1 parallels). */
  kind: 'decision' | 'confidence' | 'elevation' | 'margin';
  /** World-coord segments [x1, y1, x2, y2] in [-5, +5]. */
  lineSegments: Array<[number, number, number, number]>;
}

/** Phase 42 — one exact analytic line (mirrors Linear2DLine). */
export interface Linear2DLinePayload {
  kind: 'decision' | 'margin_pos' | 'margin_neg' | 'confidence' | 'pairwise' | 'slope';
  level: number;
  /** World-coord segment [x1, y1, x2, y2]; null when the line misses the rect. */
  segmentWorld: [number, number, number, number] | null;
  segmentRaw: [number, number, number, number] | null;
  label: string;
  classA: number | null;
  classB: number | null;
}

/** Phase 42 — exact linear geometry (mirrors Linear2DGeometry, null when N/A). */
export interface Linear2DPayload {
  family: string;
  task: 'classification' | 'regression';
  xFeature: string;
  yFeature: string;
  w1: number;
  w2: number;
  bEff: number;
  decisionThreshold: number;
  normalVector: [number, number];
  corridorWidth: number | null;
  marginLevel: number | null;
  isXZeroed: boolean;
  isYZeroed: boolean;
  bothZeroed: boolean;
  sparsityRatio: number;
  l1Ratio: number | null;
  alpha: number | null;
  penalty: string | null;
  slopeW1: number | null;
  xSweepIntercept: number | null;
  formula: string;
  isMulticlass: boolean;
  nClasses: number;
  lines: Linear2DLinePayload[];
}

/** Phase 43 — one support-vector halo (mirrors SvmHalo2D + world coords). */
export interface SvmHalo2D {
  trainIndex: number;
  xWorld: number;
  yWorld: number;
  dualCoef: number;
  /** |α|/C clamped to [0, 1] — drives the diamond radius. */
  haloScale: number;
  status: 'on_margin' | 'violates_margin' | 'support';
}

/** Phase 43 — static SVM overlay payload (mirrors SvmState2D). */
export interface Svm2DPayload {
  modelType: string;
  kernel: string;
  gamma: number | null;
  c: number | null;
  degree: number;
  coef0: number;
  isRegressor: boolean;
  isBinary: boolean;
  nClasses: number;
  nSupportTotal: number;
  isLinearNoSv: boolean;
  halos: SvmHalo2D[];
  halosTruncated: boolean;
}

/** Phase 43 — static KNN descriptor (mirrors the knn_2d contour field). */
export interface KnnStatic2D {
  isKnnModel: boolean;
  modelType: string;
  k: number;
  weightsMode: string;
  metric: string;
  isRadiusModel: boolean;
  radius: number | null;
}

/** Phase 43 — one spider-line endpoint in world coords. */
export interface KnnFrameNeighbor {
  trainIndex: number;
  xWorld: number;
  yWorld: number;
  distanceFull: number;
  distance2d: number;
  voteWeight: number;
  rank: number;
  label: string | number | null;
}

/** Phase 43 — per-frame radar payload (mirrors KnnFrame2DResponse). */
export interface KnnFrame2DPayload {
  xFeature: string;
  yFeature: string;
  rowIndex: number;
  queryWorld: [number, number];
  radarRadiusRaw: number;
  /** Ellipse radii (world units) — raw circles skew under anisotropic spans. */
  radarRadiusWorld: [number, number];
  neighbors: KnnFrameNeighbor[];
  votes: Record<string, number>;
  predicted: string | number | null;
  k: number;
  weightsMode: string;
  metric: string;
  isRadiusModel: boolean;
  outlier: boolean;
  displayCapped: boolean;
  totalNeighbors: number;
  computationTimeMs: number;
}

/** Phase 41 — dense 2D contour payload (mirrors Contour2DResponse). */
export interface Contour2DPayload {
  gridResolution: number;
  xFeature: string;
  yFeature: string;
  xRangeRaw: [number, number];
  yRangeRaw: [number, number];
  predictionTask: 'classification' | 'regression';
  classes: Array<string | number>;
  heatmapMin: number;
  heatmapMax: number;
  /** Row-major Ny*Nx uint8 field; decode with min + (q/255)*(max-min). */
  heatmapQuantized: Uint8Array;
  contours: ContourLevel2D[];
  responseCurve1D: { xValuesRaw: number[]; yPred: number[] } | null;
  /** Phase 42 — exact linear overlay; null for non-linear / pipelined models. */
  linear2d: Linear2DPayload | null;
  /** Phase 43 — SV halos in world coords; null when N/A (non-SVM / linear SVM). */
  svm2d: Svm2DPayload | null;
  /** Phase 43 — static KNN descriptor; null when the model is not KNN. */
  knn2d: KnnStatic2D | null;
  computationTimeMs: number;
}

export interface SurfaceMeshPayload {
  gridResolution: number;
  xFeature: string;
  yFeature: string;
  xRange: [number, number];
  yRange: [number, number];
  zRange: [number, number];
  zHeightsWorld: Float32Array;
  rawPredictions: Float32Array;
  pointPositionsWorld: Array<[number, number, number]>;
  pointRowIndices: number[];
  /** Full trajectory frames (shared shape with the stage3d store). */
  trajectoryPoints: Array<{
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
  }>;
  pcaVarianceRatio: [number, number, number] | null;
  pcaLoadings: number[][] | null;
  resolvedZMode: ZMode;
  hasConstantAxisWarning: boolean;
  computationTimeMs: number;
  datasetRowCount: number;
}

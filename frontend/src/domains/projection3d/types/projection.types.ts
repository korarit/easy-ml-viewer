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

/** Phase 44 — one class ellipse in world coords (mirrors ClassEllipse2D). */
export interface ClassEllipse2D {
  classIndex: number;
  label: string | number | null;
  meanRaw: [number, number];
  stdRaw: [number, number];
  rotationRad: number;
  centerWorld: [number, number];
  /** 1-sigma radii in world units — multiply by k for the k-sigma ring. */
  radiiWorld1Sigma: [number, number];
  prior: number | null;
}

/** Phase 44 — exact LDA binary line (mirrors DiscriminantLine2D). */
export interface DiscriminantLine2D {
  segmentRaw: [number, number, number, number] | null;
  segmentWorld: [number, number, number, number] | null;
  w1: number;
  w2: number;
  bEff: number;
  label: string;
}

/** Phase 44 — exact QDA binary conic (mirrors Conic2D). */
export interface QdaConic2D {
  kind: string;
  conicType: 'hyperbola' | 'ellipse' | 'parabola' | 'degenerate' | string;
  coefficientsRaw: { A: number; B: number; C: number; D: number; E: number; F: number };
  classA: number;
  classB: number;
  segmentsWorld: Array<[number, number, number, number]>;
  segmentsTruncated: boolean;
  label: string;
}

/** Phase 44 — discrete-NB step legend (mirrors DiscreteNb2D). */
export interface DiscreteNb2D {
  variant: string;
  algorithm: string;
  alpha: number | null;
  binarize: number | null;
  axisLogProb: number[][];
  classes: Array<string | number>;
  formula: string;
}

/** Phase 44 — probabilistic overlay payload (mirrors Probabilistic2DGeometry). */
export interface Probabilistic2DPayload {
  family: 'gnb' | 'discrete_nb' | 'lda' | 'qda' | string;
  algorithm: string;
  xFeature: string;
  yFeature: string;
  xFeatureIdx: number;
  yFeatureIdx: number;
  nClasses: number;
  classes: Array<string | number>;
  ellipses: ClassEllipse2D[];
  ellipsesTruncated: boolean;
  isCovarianceApproximated: boolean;
  discriminantLine: DiscriminantLine2D | null;
  conic: QdaConic2D | null;
  discrete: DiscreteNb2D | null;
  formula: string;
}

/** Phase 45 — one axis-aligned cut in world coords (mirrors TreeCutSegment2D). */
export interface TreeCutSegment2D {
  axis: 'x' | 'y';
  threshold: number;
  nodeId: number;
  treeIndex: number;
  depth: number;
  segmentRaw: [number, number, number, number];
  segmentWorld: [number, number, number, number];
}

/** Phase 45 — one leaf bounding box in world coords (mirrors LeafBox2D). */
export interface LeafBox2D {
  leafId: number;
  treeIndex: number;
  boxRaw: [number, number, number, number];
  /** [xmin, xmax, ymin, ymax] in world coords. */
  boxWorld: [number, number, number, number];
  prediction: string | number | null;
  samples: number;
  impurity: number;
  rule: string;
  isActive: boolean;
}

/** Phase 45 — one weak learner for the cascade slider (mirrors AdaBoostStage2D). */
export interface AdaBoostStage2D {
  stage: number;
  featureIndex: number;
  axis: 'x' | 'y' | 'off';
  threshold: number | null;
  weight: number;
  treeDepth: number;
  segmentWorld: [number, number, number, number] | null;
}

/** Phase 45 — tree & ensemble overlay payload (mirrors Trees2DGeometry). */
export interface Trees2DPayload {
  family: 'decision_tree' | 'random_forest' | 'extra_trees' | 'adaboost' | string;
  algorithm: string;
  xFeature: string;
  yFeature: string;
  xFeatureIdx: number;
  yFeatureIdx: number;
  nEstimators: number;
  treesExported: number;
  segments: TreeCutSegment2D[];
  segmentsTruncated: boolean;
  leaves: LeafBox2D[];
  leavesTruncated: boolean;
  stages: AdaBoostStage2D[];
  stagesTruncated: boolean;
  baseValue: number | null;
  noAxisSplits: boolean;
  multiOutput: boolean;
  formula: string;
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

/** Phase 46 — one boosting cut in world coords (mirrors BoostCutSegment2D). */
export interface BoostCutSegment2D {
  axis: 'x' | 'y';
  threshold: number;
  nodeId: number;
  treeIndex: number;
  depth: number;
  segmentRaw: [number, number, number, number];
  segmentWorld: [number, number, number, number];
}

/** Phase 46 — one first-tree leaf box in world coords (mirrors BoostLeafBox2D). */
export interface BoostLeafBox2D {
  leafId: number;
  treeIndex: number;
  boxRaw: [number, number, number, number];
  /** [xmin, xmax, ymin, ymax] in world coords. */
  boxWorld: [number, number, number, number];
  leafValue: number;
  rule: string;
  isActive: boolean;
}

/** Phase 46 — native TreeSHAP push vector on the (X, Y) plane. */
export interface BoostShap2D {
  phiX: number;
  phiY: number;
  magnitude: number;
  bias: number | null;
  method: string;
}

/** Phase 46 — one CatBoost checkerboard cell (mirrors ObliviousCell2D). */
export interface ObliviousCell2D {
  cellId: number;
  leafIndex: number;
  bitmask: string;
  boxRaw: [number, number, number, number];
  /** [xmin, xmax, ymin, ymax] in world coords. */
  boxWorld: [number, number, number, number];
  isActive: boolean;
}

/** Phase 46 — boosting heavyweight overlay payload (mirrors Boosting2DGeometry). */
export interface Boosting2DPayload {
  family: 'sklearn_gbm' | 'hist_gbm' | 'xgboost' | 'lightgbm' | 'catboost' | string;
  algorithm: string;
  xFeature: string;
  yFeature: string;
  xFeatureIdx: number;
  yFeatureIdx: number;
  nEstimators: number;
  treesExported: number;
  segments: BoostCutSegment2D[];
  segmentsTruncated: boolean;
  leaves: BoostLeafBox2D[];
  leavesTruncated: boolean;
  binEdgesXRaw: number[];
  binEdgesYRaw: number[];
  binLinesXWorld: number[];
  binLinesYWorld: number[];
  activeBin: [number, number] | null;
  obliviousXCutsRaw: number[];
  obliviousYCutsRaw: number[];
  obliviousXLinesWorld: number[];
  obliviousYLinesWorld: number[];
  obliviousCells: ObliviousCell2D[];
  obliviousTruncated: boolean;
  obliviousRule: string;
  shapVector: BoostShap2D | null;
  noAxisSplits: boolean;
  formula: string;
}

/** Phase 47 — one MLP first-layer switching line (mirrors MlpHyperplane2D). */
export interface MlpHyperplane2D {
  neuronIdx: number;
  w1: number;
  w2: number;
  bEff: number;
  segmentRaw: [number, number, number, number] | null;
  segmentWorld: [number, number, number, number] | null;
  isDead: boolean;
  isAlwaysActive: boolean;
}

/** Phase 47 — pipeline raw<->scaled toggle metadata (mirrors PipelineSpace2D). */
export interface PipelineSpace2D {
  hasToggle: boolean;
  scalerKind: string;
  innerFamily: string;
  innerAlgorithm: string;
  muX: number | null;
  sigmaX: number | null;
  muY: number | null;
  sigmaY: number | null;
  scaledXRange: [number, number] | null;
  scaledYRange: [number, number] | null;
  formula: string;
}

/** Phase 47 — latent SVD sample point (mirrors OnnxLatent2D.latent_points_2d). */
export type OnnxLatentPoint2D = [number, number];

/** Phase 47 — ONNX summary + optional latent plane (mirrors OnnxLatent2D). */
export interface OnnxLatent2D {
  operator: string;
  chain: string[];
  nNodes: number;
  latentAvailable: boolean;
  latentPoints2d: OnnxLatentPoint2D[];
  latentExplainedVariance: number[];
  penultimateWidth: number;
  formula: string;
}

/** Phase 47 — neural / pipeline / ONNX overlay payload (mirrors Neural2DGeometry). */
export interface Neural2DPayload {
  family: 'mlp' | 'pipeline' | 'onnx_classical' | 'onnx_deep' | string;
  algorithm: string;
  xFeature: string;
  yFeature: string;
  xFeatureIdx: number;
  yFeatureIdx: number;
  activation: string | null;
  hyperplanes: MlpHyperplane2D[];
  hyperplanesTruncated: boolean;
  deadCount: number;
  activeCount: number;
  activeRatio: number | null;
  pipeline: PipelineSpace2D | null;
  onnx: OnnxLatent2D | null;
  noAxisSplits: boolean;
  formula: string;
}

/** Phase 46 — per-frame boosting payload (mirrors BoostFrame2DResponse). */
export interface BoostFrame2DPayload {
  xFeature: string;
  yFeature: string;
  rowIndex: number;
  queryWorld: [number, number];
  family: string;
  algorithm: string;
  shap: BoostShap2D | null;
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
  /** Phase 44 — probabilistic ellipses / LDA line / QDA conic; null when N/A. */
  probabilistic2d: Probabilistic2DPayload | null;
  /** Phase 45 — tree cuts / leaf boxes / forest cuts / adaboost stages; null when N/A. */
  trees2d: Trees2DPayload | null;
  /** Phase 46 — GBM cuts / hist bins / xgb+lgbm polygons+SHAP / catboost checkerboard; null when N/A. */
  boosting2d: Boosting2DPayload | null;
  /** Phase 47 — MLP neuron cuts / pipeline toggle / ONNX latent plane; null when N/A. */
  neural2d: Neural2DPayload | null;
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

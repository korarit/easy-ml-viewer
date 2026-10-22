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
  kind: 'decision' | 'confidence' | 'elevation';
  /** World-coord segments [x1, y1, x2, y2] in [-5, +5]. */
  lineSegments: Array<[number, number, number, number]>;
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

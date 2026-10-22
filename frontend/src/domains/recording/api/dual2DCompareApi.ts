/** Phase 48 — typed client for POST /api/v1/projection/dual-compare-2d. */
import { apiFetch } from '../../../core/api/client';

export interface Dual2DCompareResult {
  xFeature: string;
  yFeature: string;
  gridResolution: number;
  predictionTask: string;
  disagreementPct: number;
  agreementPct: number;
  nCells: number;
  nConflictCells: number;
  consensusNotice: string;
  conflictMask: number[];
  conflictMaskDownsampled: boolean;
  conflictFrameIndices: number[];
  modelAName: string;
  modelBName: string;
  contourA: {
    contours: Array<{ threshold: number; classIndex: number; kind: string; lineSegments: number[][] }>;
  };
  contourB: {
    contours: Array<{ threshold: number; classIndex: number; kind: string; lineSegments: number[][] }>;
  };
  computationTimeMs: number;
}

export interface FetchDual2DCompareParams {
  datasetId: string;
  modelAId: string;
  modelBId: string;
  mappingId: string;
  xFeature: string;
  yFeature: string;
  gridResolution: number;
  selectedRowIndices: number[];
}

interface Dual2DCompareResponse {
  x_feature: string;
  y_feature: string;
  grid_resolution: number;
  prediction_task: string;
  disagreement_pct: number;
  agreement_pct: number;
  n_cells: number;
  n_conflict_cells: number;
  consensus_notice: string;
  conflict_mask: number[];
  conflict_mask_downsampled: boolean;
  conflict_frame_indices: number[];
  model_a_name: string;
  model_b_name: string;
  contour_a: { contours: Array<{ threshold: number; class_index: number; kind: string; line_segments: number[][] }> };
  contour_b: { contours: Array<{ threshold: number; class_index: number; kind: string; line_segments: number[][] }> };
  computation_time_ms: number;
}

export async function fetchDual2DCompare(
  params: FetchDual2DCompareParams,
  signal?: AbortSignal,
): Promise<Dual2DCompareResult> {
  const data = await apiFetch<Dual2DCompareResponse>('/api/v1/projection/dual-compare-2d', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_a_id: params.modelAId,
      model_b_id: params.modelBId,
      mapping_id: params.mappingId,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      grid_resolution: params.gridResolution,
      selected_row_indices: params.selectedRowIndices,
    }),
    signal,
  });
  const slim = (c: Dual2DCompareResponse['contour_a']) => ({
    contours: c.contours.map((l) => ({
      threshold: l.threshold,
      classIndex: l.class_index,
      kind: l.kind,
      lineSegments: l.line_segments,
    })),
  });
  return {
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    gridResolution: data.grid_resolution,
    predictionTask: data.prediction_task,
    disagreementPct: data.disagreement_pct,
    agreementPct: data.agreement_pct,
    nCells: data.n_cells,
    nConflictCells: data.n_conflict_cells,
    consensusNotice: data.consensus_notice,
    conflictMask: data.conflict_mask,
    conflictMaskDownsampled: data.conflict_mask_downsampled,
    conflictFrameIndices: data.conflict_frame_indices,
    modelAName: data.model_a_name,
    modelBName: data.model_b_name,
    contourA: slim(data.contour_a),
    contourB: slim(data.contour_b),
    computationTimeMs: data.computation_time_ms,
  };
}

/** Index of the next conflict frame strictly after `fromFrame` (wraps around). */
export function nextConflictFrame(indices: number[], fromFrame: number): number | null {
  if (indices.length === 0) return null;
  const sorted = [...indices].sort((a, b) => a - b);
  for (const idx of sorted) if (idx > fromFrame) return idx;
  return sorted[0];
}

/** Index of the previous conflict frame strictly before `fromFrame` (wraps around). */
export function prevConflictFrame(indices: number[], fromFrame: number): number | null {
  if (indices.length === 0) return null;
  const sorted = [...indices].sort((a, b) => a - b);
  for (let i = sorted.length - 1; i >= 0; i--) if (sorted[i] < fromFrame) return sorted[i];
  return sorted[sorted.length - 1];
}

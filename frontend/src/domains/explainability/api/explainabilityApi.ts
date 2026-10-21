/** Phase 12 — typed client for the local-explainability endpoints. */
import { apiFetch } from '../../../core/api/client';
import type {
  ExplainabilitySessionPayload,
  FrameExplanationPayload,
} from '../types/explainability.types';

export interface BuildExplainabilitySessionParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  selectedRowIndices: number[];
  sessionId?: string;
}

export async function fetchBuildExplainabilitySession(
  params: BuildExplainabilitySessionParams,
  signal?: AbortSignal,
): Promise<ExplainabilitySessionPayload> {
  return apiFetch<ExplainabilitySessionPayload>('/api/v1/explainability/sessions/build', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      selected_row_indices: params.selectedRowIndices,
      ...(params.sessionId ? { session_id: params.sessionId } : {}),
    }),
    signal,
  });
}

export async function fetchFrameExplanation(
  sessionId: string,
  frameIndex: number,
  topK: number,
  signal?: AbortSignal,
): Promise<FrameExplanationPayload> {
  return apiFetch<FrameExplanationPayload>(
    `/api/v1/explainability/${encodeURIComponent(sessionId)}/frames/${frameIndex}?top_k=${topK}`,
    { signal },
  );
}

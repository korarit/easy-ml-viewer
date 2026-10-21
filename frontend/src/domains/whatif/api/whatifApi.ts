/** Phase 13 — typed client for the what-if sandbox endpoints. */
import { apiFetch } from '../../../core/api/client';
import type {
  CounterfactualResult,
  WhatIfEvaluation,
  WhatIfSessionPayload,
} from '../types/whatif.types';

export interface BuildWhatIfSessionParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  selectedRowIndices: number[];
  axisXFeatureIndex?: number;
  axisYFeatureIndex?: number;
}

export async function fetchBuildWhatIfSession(
  params: BuildWhatIfSessionParams,
  signal?: AbortSignal,
): Promise<WhatIfSessionPayload> {
  return apiFetch<WhatIfSessionPayload>('/api/v1/whatif/sessions/build', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      selected_row_indices: params.selectedRowIndices,
      ...(params.axisXFeatureIndex !== undefined
        ? { axis_x_feature_index: params.axisXFeatureIndex }
        : {}),
      ...(params.axisYFeatureIndex !== undefined
        ? { axis_y_feature_index: params.axisYFeatureIndex }
        : {}),
    }),
    signal,
  });
}

export async function fetchEvaluateWhatIf(
  sessionId: string,
  frameIndex: number,
  mutatedFeatureValues: Record<string, number>,
  signal?: AbortSignal,
): Promise<WhatIfEvaluation> {
  return apiFetch<WhatIfEvaluation>(
    `/api/v1/whatif/${encodeURIComponent(sessionId)}/frames/${frameIndex}/evaluate`,
    {
      method: 'POST',
      body: JSON.stringify({ mutated_feature_values: mutatedFeatureValues }),
      signal,
    },
  );
}

export interface CounterfactualRequestParams {
  targetClassOrThreshold?: number;
  lockedFeatureIndices: number[];
}

export async function fetchCounterfactual(
  sessionId: string,
  frameIndex: number,
  params: CounterfactualRequestParams,
  signal?: AbortSignal,
): Promise<CounterfactualResult> {
  return apiFetch<CounterfactualResult>(
    `/api/v1/whatif/${encodeURIComponent(sessionId)}/frames/${frameIndex}/counterfactual`,
    {
      method: 'POST',
      body: JSON.stringify({
        ...(params.targetClassOrThreshold !== undefined
          ? { target_class_or_threshold: params.targetClassOrThreshold }
          : {}),
        locked_feature_indices: params.lockedFeatureIndices,
      }),
      signal,
    },
  );
}

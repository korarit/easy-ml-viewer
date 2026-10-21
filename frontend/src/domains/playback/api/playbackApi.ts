/** Phase 10 — typed client for the playback manifest endpoints. */
import { apiFetch } from '../../../core/api/client';
import type { PlaybackSequenceManifest } from '../types/playback.types';

export interface BuildManifestParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  selectedRowIndices: number[];
  sessionId?: string;
  defaultFps?: number;
}

export async function fetchBuildPlaybackManifest(
  params: BuildManifestParams,
  signal?: AbortSignal,
): Promise<PlaybackSequenceManifest> {
  return apiFetch<PlaybackSequenceManifest>('/api/v1/playback/manifest', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      selected_row_indices: params.selectedRowIndices,
      ...(params.sessionId ? { session_id: params.sessionId } : {}),
      ...(params.defaultFps ? { default_fps: params.defaultFps } : {}),
    }),
    signal,
  });
}

export async function fetchPlaybackManifest(
  sessionId: string,
  signal?: AbortSignal,
): Promise<PlaybackSequenceManifest> {
  return apiFetch<PlaybackSequenceManifest>(
    `/api/v1/playback/${encodeURIComponent(sessionId)}/manifest`,
    { signal },
  );
}

/** Phase 11 — typed client for the timeline tracks + bookmarks endpoints. */
import { apiFetch } from '../../../core/api/client';
import type {
  BookmarkColor,
  TimelineTracksPayload,
  UserBookmark,
} from '../types/timeline.types';

export interface BuildTracksParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  selectedRowIndices: number[];
  sessionId?: string;
}

export async function fetchBuildTimelineTracks(
  params: BuildTracksParams,
  signal?: AbortSignal,
): Promise<TimelineTracksPayload> {
  return apiFetch<TimelineTracksPayload>('/api/v1/timeline/tracks/build', {
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

export async function fetchTimelineTracks(
  sessionId: string,
  signal?: AbortSignal,
): Promise<TimelineTracksPayload> {
  return apiFetch<TimelineTracksPayload>(
    `/api/v1/timeline/${encodeURIComponent(sessionId)}/tracks`,
    { signal },
  );
}

export interface CreateBookmarkParams {
  sessionId: string;
  frameIndex: number;
  datasetRowIndex: number;
  color: BookmarkColor;
  noteText: string;
}

export async function fetchCreateTimelineBookmark(
  params: CreateBookmarkParams,
  signal?: AbortSignal,
): Promise<UserBookmark> {
  return apiFetch<UserBookmark>(
    `/api/v1/timeline/${encodeURIComponent(params.sessionId)}/bookmarks`,
    {
      method: 'POST',
      body: JSON.stringify({
        frame_index: params.frameIndex,
        dataset_row_index: params.datasetRowIndex,
        color: params.color,
        note_text: params.noteText,
      }),
      signal,
    },
  );
}

export async function fetchDeleteTimelineBookmark(
  sessionId: string,
  bookmarkId: string,
  signal?: AbortSignal,
): Promise<{ deleted: string; session_id: string }> {
  return apiFetch<{ deleted: string; session_id: string }>(
    `/api/v1/timeline/${encodeURIComponent(sessionId)}/bookmarks/${encodeURIComponent(bookmarkId)}`,
    { method: 'DELETE', signal },
  );
}

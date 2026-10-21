/** Phase 15 — typed client for `.mlv` bake / load / replay endpoints. */
import { API_BASE_URL } from '../../../core/api/client';
import type { MlvBookmark, PrebakedMlvSession } from '../types/recording.types';

export interface BakeMlvParams {
  sessionId: string;
  datasetId: string;
  modelId: string;
  mappingId: string;
  selectedRowIndices: number[];
  title: string;
  notes: string;
  surfaceResolution: number;
  includeExplainability: boolean;
  includeBookmarks: boolean;
}

async function throwForStatus(res: Response): Promise<void> {
  if (res.ok) return;
  let code = `HTTP_${res.status}`;
  let message = `Request failed (${res.status})`;
  try {
    const data = (await res.clone().json()) as {
      error?: { code?: string; message?: string };
    };
    if (data?.error?.code) code = data.error.code;
    if (data?.error?.message) message = data.error.message;
  } catch {
    /* keep defaults */
  }
  const err = new Error(message) as Error & { code: string; httpStatus: number };
  err.code = code;
  err.httpStatus = res.status;
  throw err;
}

export async function fetchEstimateMlvSize(params: {
  totalFrames: number;
  surfaceResolution: number;
  includeExplainability: boolean;
}): Promise<number> {
  const res = await fetch(`${API_BASE_URL}/api/v1/recordings/estimate-size`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      total_frames: params.totalFrames,
      surface_resolution: params.surfaceResolution,
      include_explainability: params.includeExplainability,
    }),
  });
  await throwForStatus(res);
  const data = (await res.json()) as { estimated_compressed_bytes: number };
  return data.estimated_compressed_bytes;
}

export async function fetchBakeMlv(params: BakeMlvParams): Promise<Blob> {
  const res = await fetch(
    `${API_BASE_URL}/api/v1/recordings/${encodeURIComponent(params.sessionId)}/bake`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: params.datasetId,
        model_id: params.modelId,
        mapping_id: params.mappingId,
        selected_row_indices: params.selectedRowIndices,
        session_id: params.sessionId,
        title: params.title,
        notes: params.notes,
        surface_resolution: params.surfaceResolution,
        include_explainability: params.includeExplainability,
        include_bookmarks: params.includeBookmarks,
      }),
    },
  );
  await throwForStatus(res);
  return res.blob();
}

export async function fetchLoadMlv(file: File): Promise<PrebakedMlvSession> {
  const form = new FormData();
  form.append('file', file, file.name);
  const res = await fetch(`${API_BASE_URL}/api/v1/recordings/load`, {
    method: 'POST',
    body: form,
  });
  await throwForStatus(res);
  return (await res.json()) as PrebakedMlvSession;
}

export async function fetchMlvInfo(sessionId: string): Promise<PrebakedMlvSession> {
  const res = await fetch(
    `${API_BASE_URL}/api/v1/recordings/${encodeURIComponent(sessionId)}/info`,
  );
  await throwForStatus(res);
  return (await res.json()) as PrebakedMlvSession;
}

export async function fetchRebakeBookmarks(
  sessionId: string,
  bookmarks: MlvBookmark[],
): Promise<Blob> {
  const res = await fetch(
    `${API_BASE_URL}/api/v1/recordings/${encodeURIComponent(sessionId)}/bookmarks`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bookmarks }),
    },
  );
  await throwForStatus(res);
  return res.blob();
}

/** Trigger a browser download for a baked `.mlv` blob. */
export function downloadMlvBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.mlv') ? filename : `${filename}.mlv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 5_000);
}

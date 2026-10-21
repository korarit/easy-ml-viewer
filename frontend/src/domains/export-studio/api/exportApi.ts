/** Phase 16 — typed client for the enriched-table export endpoints. */
import { API_BASE_URL } from '../../../core/api/client';
import type { ExportPreviewResponse, ExportTableParams } from '../types/export.types';

export interface ExportFilenames {
  csv: string;
  xlsx: string;
  parquet: string;
}

export function exportFilename(params: Pick<ExportTableParams, 'format'>, stem: string, n: number): string {
  const ext = params.format === 'xlsx' ? 'xlsx' : params.format === 'parquet' ? 'parquet' : 'csv';
  const safe = stem.replace(/[^A-Za-z0-9-_]+/g, '_').slice(0, 40) || 'export';
  return `enriched-${safe}-n${n}.${ext}`;
}

async function throwForStatus(res: Response): Promise<void> {
  if (res.ok) return;
  let code = `HTTP_${res.status}`;
  let message = `Request failed (${res.status})`;
  try {
    const data = (await res.clone().json()) as { error?: { code?: string; message?: string } };
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

function toBody(params: ExportTableParams): Record<string, unknown> {
  return {
    dataset_id: params.datasetId,
    model_id: params.modelId,
    mapping_id: params.mappingId,
    session_id: params.sessionId,
    format: params.format,
    scope: params.scope,
    range_start: params.rangeStart,
    range_end: params.rangeEnd,
    selected_row_indices: params.selectedRowIndices,
    include_attribution_columns: params.includeAttributionColumns,
    include_bookmarks: params.includeBookmarks,
  };
}

export async function fetchExportEnrichedTable(params: ExportTableParams): Promise<Blob> {
  const res = await fetch(`${API_BASE_URL}/api/v1/exports/enriched-table`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toBody(params)),
  });
  await throwForStatus(res);
  return res.blob();
}

export async function fetchExportPreview(params: ExportTableParams): Promise<ExportPreviewResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/exports/enriched-table/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toBody(params)),
  });
  await throwForStatus(res);
  return (await res.json()) as ExportPreviewResponse;
}

/** Trigger a browser download for an exported table blob (revokes the URL after 5s). */
export function downloadExportBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 5_000);
}

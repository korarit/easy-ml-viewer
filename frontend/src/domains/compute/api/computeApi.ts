/** Phase 17 — typed client for `/api/v1/compute/*` telemetry & job endpoints. */
import { API_BASE_URL } from '../../../core/api/client';
import type {
  ComputeChunkSize,
  ComputeDeviceMode,
  ComputeJobInfo,
  ComputeTelemetry,
} from '../types/compute.types';

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

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  await throwForStatus(res);
  return (await res.json()) as T;
}

export async function fetchComputeTelemetry(): Promise<ComputeTelemetry> {
  const res = await fetch(`${API_BASE_URL}/api/v1/compute/telemetry`);
  await throwForStatus(res);
  return (await res.json()) as ComputeTelemetry;
}

export async function updateComputeConfig(params: {
  deviceMode: ComputeDeviceMode;
  chunkSize: ComputeChunkSize;
}): Promise<{ device_mode: ComputeDeviceMode; chunk_size: number }> {
  return postJson('/api/v1/compute/config', {
    device_mode: params.deviceMode,
    chunk_size: params.chunkSize,
  });
}

export async function registerComputeJob(params: {
  totalRows: number;
  chunkSize: ComputeChunkSize;
  deviceMode: ComputeDeviceMode;
}): Promise<ComputeJobInfo> {
  const data = await postJson<{
    job_id: string;
    total_rows: number;
    chunk_size: number;
    chunks_total: number;
    device_mode: ComputeDeviceMode;
    route: ComputeJobInfo['route'];
    status: ComputeJobInfo['status'];
  }>('/api/v1/compute/jobs', {
    total_rows: params.totalRows,
    chunk_size: params.chunkSize,
    device_mode: params.deviceMode,
  });
  return {
    job_id: data.job_id,
    total_rows: data.total_rows,
    chunk_size: data.chunk_size,
    chunks_total: data.chunks_total,
    device_mode: data.device_mode,
    route: data.route,
    status: data.status,
  };
}

export async function cancelComputeJob(
  jobId: string,
): Promise<{ job_id: string; cancelled: boolean; status: string }> {
  return postJson(`/api/v1/compute/jobs/${encodeURIComponent(jobId)}/cancel`);
}

export async function purgeComputeCaches(): Promise<{ freed_mb: number; remaining_cache_mb: number }> {
  return postJson('/api/v1/compute/cache/purge');
}

export async function estimateComputeJob(params: {
  totalRows: number;
  nFeatures: number;
  chunkSize: number;
  isDeepModel?: boolean;
}): Promise<{ route: string; num_chunks: number; chunk_size: number; estimated_peak_extra_mb: number }> {
  return postJson('/api/v1/compute/estimate', {
    total_rows: params.totalRows,
    n_features: params.nFeatures,
    chunk_size: params.chunkSize,
    is_deep_model: params.isDeepModel ?? false,
  });
}

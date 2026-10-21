/** Phase 17 — compute engine shared types (mirrors backend ComputeTelemetryVO). */

export type ComputeDeviceMode = 'auto_smart_router' | 'force_cpu_only' | 'prefer_gpu';

export type ComputeExecutionRoute = 'cpu_vectorized' | 'gpu_accelerated';

export type ComputeMemoryGuardLevel = 'normal' | 'elevated' | 'guard_active';

export type ComputeChunkSize = 2500 | 5000 | 10000;

export interface ComputeTelemetry {
  physical_cores: number;
  logical_cores: number;
  optimal_workers: number;
  available_onnx_providers: string[];
  active_device_mode: ComputeDeviceMode;
  process_rss_mb: number;
  system_available_mb: number;
  system_memory_percent: number;
  memory_guard_level: ComputeMemoryGuardLevel;
  active_jobs_count: number;
  default_chunk_size: number;
}

export interface ComputeJobInfo {
  job_id: string;
  total_rows: number;
  chunk_size: number;
  chunks_total: number;
  device_mode: ComputeDeviceMode;
  route: ComputeExecutionRoute;
  status: 'running' | 'cancelled' | 'completed' | 'not_found';
}

export interface ChunkedJobProgress {
  jobId: string;
  totalRows: number;
  processedRows: number;
  chunkSize: number;
  chunksTotal: number;
  chunksDone: number;
  percent: number;
  rowsPerSecond: number;
  etaSeconds: number;
  status: 'running' | 'cancelled' | 'completed';
}

export const COMPUTE_DEVICE_MODE_LABELS: Record<ComputeDeviceMode, string> = {
  auto_smart_router: 'Auto Smart Router (Recommended)',
  force_cpu_only: 'Force Multi-Core CPU Only',
  prefer_gpu: 'Prefer GPU Acceleration (CUDA / DirectML)',
};

export const COMPUTE_CHUNK_SIZE_LABELS: Record<ComputeChunkSize, string> = {
  2500: '2,500 rows (Low RAM)',
  5000: '5,000 rows (Balanced — Default)',
  10000: '10,000 rows (High Throughput)',
};

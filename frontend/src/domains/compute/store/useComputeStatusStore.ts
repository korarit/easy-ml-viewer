import { create } from 'zustand';
import {
  cancelComputeJob,
  fetchComputeTelemetry,
  purgeComputeCaches,
  registerComputeJob,
  updateComputeConfig,
} from '../api/computeApi';
import type {
  ChunkedJobProgress,
  ComputeChunkSize,
  ComputeDeviceMode,
  ComputeTelemetry,
} from '../types/compute.types';

interface ComputeStatusState {
  telemetry: ComputeTelemetry | null;
  isLoading: boolean;
  lastError: string | null;
  lastRefreshedAt: number | null;
  deviceMode: ComputeDeviceMode;
  chunkSize: ComputeChunkSize;
  workerThreads: number;
  isDiagnosticsOpen: boolean;
  activeJob: ChunkedJobProgress | null;
  refreshTelemetry: () => Promise<void>;
  setDeviceMode: (mode: ComputeDeviceMode) => Promise<void>;
  setChunkSize: (size: ComputeChunkSize) => Promise<void>;
  setWorkerThreads: (n: number) => void;
  setDiagnosticsOpen: (open: boolean) => void;
  purgeCaches: () => Promise<number>;
  startLocalJob: (totalRows: number) => Promise<void>;
  cancelActiveJob: () => Promise<void>;
  dismissActiveJob: () => void;
}

function clampChunkSize(n: number): ComputeChunkSize {
  if (n <= 2500) return 2500;
  if (n >= 10000) return 10000;
  return 5000;
}

/** Local chunk ticker — real Phase 18+ simulations stream progress via this shape. */
function tickLocalJob(job: ChunkedJobProgress): ChunkedJobProgress {
  const step = Math.min(job.chunkSize, job.totalRows - job.processedRows);
  const processedRows = job.processedRows + step;
  const chunksDone = Math.ceil(processedRows / job.chunkSize);
  const percent = job.totalRows === 0 ? 100 : Math.round((processedRows / job.totalRows) * 1000) / 10;
  const elapsedSec = Math.max(0.05, (Date.now() - (job as unknown as { startedAt: number }).startedAt) / 1000);
  const rowsPerSecond = Math.round(processedRows / elapsedSec);
  const remaining = job.totalRows - processedRows;
  return {
    ...job,
    processedRows,
    chunksDone,
    percent,
    rowsPerSecond,
    etaSeconds: rowsPerSecond > 0 ? Math.round((remaining / rowsPerSecond) * 10) / 10 : 0,
    status: processedRows >= job.totalRows ? 'completed' : 'running',
  };
}

let ticker: ReturnType<typeof setInterval> | null = null;

export const useComputeStatusStore = create<ComputeStatusState>()((set, get) => ({
  telemetry: null,
  isLoading: false,
  lastError: null,
  lastRefreshedAt: null,
  deviceMode: 'auto_smart_router',
  chunkSize: 5000,
  workerThreads: 0,
  isDiagnosticsOpen: false,
  activeJob: null,

  refreshTelemetry: async () => {
    set({ isLoading: true });
    try {
      const telemetry = await fetchComputeTelemetry();
      set((s) => ({
        telemetry,
        isLoading: false,
        lastError: null,
        lastRefreshedAt: Date.now(),
        deviceMode: telemetry.active_device_mode ?? s.deviceMode,
        chunkSize: clampChunkSize(telemetry.default_chunk_size ?? s.chunkSize),
        workerThreads: s.workerThreads || telemetry.optimal_workers,
      }));
    } catch (e) {
      set({ isLoading: false, lastError: e instanceof Error ? e.message : String(e) });
    }
  },

  setDeviceMode: async (mode) => {
    const prev = get().deviceMode;
    set({ deviceMode: mode });
    try {
      await updateComputeConfig({ deviceMode: mode, chunkSize: get().chunkSize });
      await get().refreshTelemetry();
    } catch (e) {
      set({ deviceMode: prev, lastError: e instanceof Error ? e.message : String(e) });
      throw e;
    }
  },

  setChunkSize: async (size) => {
    const prev = get().chunkSize;
    set({ chunkSize: size });
    try {
      await updateComputeConfig({ deviceMode: get().deviceMode, chunkSize: size });
      await get().refreshTelemetry();
    } catch (e) {
      set({ chunkSize: prev, lastError: e instanceof Error ? e.message : String(e) });
      throw e;
    }
  },

  setWorkerThreads: (n) => set({ workerThreads: Math.max(1, Math.floor(n)) }),

  setDiagnosticsOpen: (open) => set({ isDiagnosticsOpen: open }),

  purgeCaches: async () => {
    const res = await purgeComputeCaches();
    await get().refreshTelemetry();
    return res.freed_mb;
  },

  startLocalJob: async (totalRows) => {
    const { chunkSize, deviceMode } = get();
    const info = await registerComputeJob({ totalRows, chunkSize, deviceMode });
    const job: ChunkedJobProgress & { startedAt: number } = {
      jobId: info.job_id,
      totalRows,
      processedRows: 0,
      chunkSize: info.chunk_size,
      chunksTotal: info.chunks_total,
      chunksDone: 0,
      percent: 0,
      rowsPerSecond: 0,
      etaSeconds: 0,
      status: 'running',
      startedAt: Date.now(),
    };
    set({ activeJob: job });
    if (ticker) window.clearInterval(ticker);
    ticker = window.setInterval(() => {
      const cur = get().activeJob as (ChunkedJobProgress & { startedAt: number }) | null;
      if (!cur || cur.status !== 'running') {
        if (ticker) window.clearInterval(ticker);
        ticker = null;
        return;
      }
      const next = tickLocalJob(cur);
      set({ activeJob: next });
      if (next.status !== 'running' && ticker) {
        window.clearInterval(ticker);
        ticker = null;
      }
    }, 120);
  },

  cancelActiveJob: async () => {
    const job = get().activeJob;
    if (!job) return;
    try {
      await cancelComputeJob(job.jobId);
    } catch {
      /* local ticker still stops — cooperative cancel is best-effort */
    } finally {
      if (ticker) window.clearInterval(ticker);
      ticker = null;
      set((s) => (s.activeJob ? { activeJob: { ...s.activeJob, status: 'cancelled' as const } } : {}));
      await get().refreshTelemetry();
    }
  },

  dismissActiveJob: () => {
    if (ticker) window.clearInterval(ticker);
    ticker = null;
    set({ activeJob: null });
  },
}));

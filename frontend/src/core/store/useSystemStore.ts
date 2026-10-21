import { create } from 'zustand';
import { api } from '../api/client';

export interface MemorySnapshot {
  process_rss_mb: number;
  process_budget_mb: number;
  process_usage_percent: number;
  system_total_mb: number;
  system_available_mb: number;
  is_low_memory: boolean;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'offline';
  os_platform: string;
  python_version: string;
  cpu_logical_cores: number;
  omp_threads: number;
  memory: MemorySnapshot;
  cache_size_mb: number;
  latency_ms: number;
  webgl2_renderer: string | null;
}

export type WorkflowStep = 1 | 2 | 3 | 4 | 5;

interface SystemState {
  activeStep: WorkflowStep;
  unlockedMaxStep: WorkflowStep;
  health: SystemHealth | null;
  isDiagnosticsOpen: boolean;
  lastError: string | null;
  setActiveStep: (step: WorkflowStep) => void;
  unlockStep: (step: WorkflowStep) => void;
  setDiagnosticsOpen: (open: boolean) => void;
  refreshHealth: () => Promise<void>;
  triggerGarbageCollection: () => Promise<void>;
  clearDiskCache: () => Promise<number>;
}

/** Probe WebGL2 renderer exactly once, then destroy the temp context. */
function probeWebGL2Renderer(): string | null {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
    if (!gl) return null;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = ext
      ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) ?? 'WebGL2')
      : 'WebGL2';
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    canvas.remove();
    return renderer;
  } catch {
    return null;
  }
}

let cachedRenderer: string | null | undefined;

function getRenderer(): string | null {
  if (cachedRenderer === undefined) cachedRenderer = probeWebGL2Renderer();
  return cachedRenderer;
}

interface HealthResponse {
  status: 'healthy' | 'degraded';
  os_platform: string;
  python_version: string;
  cpu_logical_cores: number;
  omp_threads: number;
  memory: MemorySnapshot;
  cache_size_mb: number;
  _latencyMs?: number;
}

export const useSystemStore = create<SystemState>((set, get) => ({
  activeStep: 1,
  unlockedMaxStep: 1,
  health: null,
  isDiagnosticsOpen: false,
  lastError: null,

  setActiveStep: (step) => {
    if (step <= get().unlockedMaxStep) set({ activeStep: step });
  },
  unlockStep: (step) =>
    set((s) => ({
      unlockedMaxStep: Math.max(s.unlockedMaxStep, step) as WorkflowStep,
      activeStep: step,
    })),
  setDiagnosticsOpen: (open) => set({ isDiagnosticsOpen: open }),

  refreshHealth: async () => {
    try {
      const data = await api.get<HealthResponse>('/api/v1/system/health');
      set({
        health: {
          ...data,
          status: data.status,
          latency_ms: Math.round(data._latencyMs ?? 0),
          webgl2_renderer: getRenderer(),
        },
        lastError: null,
      });
    } catch (err) {
      set((s) => ({
        health: s.health
          ? { ...s.health, status: 'offline' as const }
          : {
              status: 'offline' as const,
              os_platform: 'unknown',
              python_version: 'unknown',
              cpu_logical_cores: 0,
              omp_threads: 0,
              memory: {
                process_rss_mb: 0,
                process_budget_mb: 4096,
                process_usage_percent: 0,
                system_total_mb: 0,
                system_available_mb: 0,
                is_low_memory: false,
              },
              cache_size_mb: 0,
              latency_ms: 0,
              webgl2_renderer: getRenderer(),
            },
        lastError: err instanceof Error ? err.message : String(err),
      }));
    }
  },

  triggerGarbageCollection: async () => {
    await api.post('/api/v1/system/gc');
    await get().refreshHealth();
  },

  clearDiskCache: async () => {
    const res = await api.del<{ freed_mb: number }>('/api/v1/system/cache');
    await get().refreshHealth();
    return res.freed_mb;
  },
}));

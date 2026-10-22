import { create } from 'zustand';
import {
  fetchMlpSurfacePreview,
  fetchMlpTracePreview,
  type MlpRowTrace,
  type MlpSurfacePreview,
} from '../api/mlpApi';

export type MlpViewMode = 'split' | 'brain' | 'surface';

export interface MlpSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  isRegressor: boolean;
  featureNames: string[];
  classCount: number;
  rowInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  viewMode: MlpViewMode;
  showDeadOnly: boolean;
  fluxThreshold: number;
  showSurface: boolean;
  showWireframe: boolean;
  traces: MlpRowTrace[];
  surface: MlpSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setViewMode: (v: MlpViewMode) => void;
  setShowDeadOnly: (v: boolean) => void;
  setFluxThreshold: (v: number) => void;
  setShowSurface: (v: boolean) => void;
  setShowWireframe: (v: boolean) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: {
    modelId: string;
    algorithmKey: string;
    isRegressor: boolean;
    featureNames: string[];
    classCount?: number;
  }) => void;
  runTrace: () => Promise<void>;
  runSurface: () => Promise<void>;
  clearError: () => void;
}

function parseCells(text: string): Array<number | string> {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const n = Number(s);
      return s !== '' && Number.isFinite(n) ? n : s;
    });
}

export const useMlpSliceStore = create<MlpSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  isRegressor: false,
  featureNames: [],
  classCount: 2,
  rowInput: '0.5, -0.3, 0.8, 0.1',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  viewMode: 'split',
  showDeadOnly: false,
  fluxThreshold: 0.15,
  showSurface: true,
  showWireframe: false,
  traces: [],
  surface: null,
  isTracing: false,
  isSurfacing: false,
  sliceError: null,
  showTelemetry: false,

  setRowInput: (v) => set({ rowInput: v }),
  setClassIdx: (v) => set({ classIdx: Math.max(0, v) }),
  setAxisX: (x) => {
    const { axisX, axisY } = get();
    if (x === axisY) set({ axisX: axisY, axisY: axisX });
    else set({ axisX: x });
  },
  setAxisY: (y) => {
    const { axisX, axisY } = get();
    if (y === axisX) set({ axisX: axisY, axisY: axisY });
    else set({ axisY: y });
  },
  setViewMode: (v) => set({ viewMode: v }),
  setShowDeadOnly: (v) => set({ showDeadOnly: v }),
  setFluxThreshold: (v) => set({ fluxThreshold: Math.max(0, v) }),
  setShowSurface: (v) => set({ showSurface: v }),
  setShowWireframe: (v) => set({ showWireframe: v }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  bindModel: (args) => {
    const count = args.classCount && args.classCount > 0 ? args.classCount : 2;
    set({
      modelId: args.modelId,
      algorithmKey: args.algorithmKey,
      isRegressor: args.isRegressor,
      featureNames: args.featureNames,
      classCount: count,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      classIdx: 0,
      traces: [],
      surface: null,
      sliceError: null,
    });
  },

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load an MLP model first.' });
      return;
    }
    const values = parseCells(s.rowInput);
    if (values.length !== s.featureNames.length) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} numeric values.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchMlpTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_idx: s.classIdx,
      });
      set({ traces: res.traces });
    } catch (err) {
      set({ sliceError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isTracing: false });
    }
  },

  runSurface: async () => {
    const s = get();
    if (!s.modelId || s.featureNames.length === 0) {
      set({ sliceError: 'Load an MLP model first.' });
      return;
    }
    const values = parseCells(s.rowInput);
    const medians: Record<number, number> = {};
    s.featureNames.forEach((_, i) => {
      if (i !== s.axisX && i !== s.axisY) {
        const v = values[i];
        medians[i] = typeof v === 'number' && Number.isFinite(v) ? v : 0;
      }
    });
    const span = (i: number, fallback: [number, number]): [number, number] => {
      const v = values[i];
      if (typeof v !== 'number' || !Number.isFinite(v)) return fallback;
      const d = Math.abs(v) * 0.9 + 0.8;
      return [v - d, v + d];
    };
    const [xMin, xMax] = span(s.axisX, [-2, 2]);
    const [yMin, yMax] = span(s.axisY, [-2, 2]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchMlpSurfacePreview(s.modelId, {
        axis_x_feature_index: s.axisX,
        axis_y_feature_index: s.axisY,
        x_min: xMin,
        x_max: xMax,
        y_min: yMin,
        y_max: yMax,
        resolution: 50,
        fixed_feature_medians: medians,
        target_idx: s.classIdx,
      });
      set({ surface: res });
    } catch (err) {
      set({ sliceError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isSurfacing: false });
    }
  },

  clearError: () => set({ sliceError: null }),
}));

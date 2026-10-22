import { create } from 'zustand';
import {
  fetchPipelineSurfacePreview,
  fetchPipelineTracePreview,
  type PipelineRowTrace,
  type PipelineSurfacePreview,
} from '../api/pipelineApi';

export type PipelineCoordSpace = 'raw' | 'transformed';
export type PipelineWaterfallMode = 'grouped' | 'expanded';

export interface PipelineSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  featureNames: string[];
  rowInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  coordSpace: PipelineCoordSpace;
  waterfallMode: PipelineWaterfallMode;
  showSurface: boolean;
  showWireframe: boolean;
  traces: PipelineRowTrace[];
  surface: PipelineSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setCoordSpace: (v: PipelineCoordSpace) => void;
  setWaterfallMode: (v: PipelineWaterfallMode) => void;
  setShowSurface: (v: boolean) => void;
  setShowWireframe: (v: boolean) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: { modelId: string; algorithmKey: string; featureNames: string[] }) => void;
  runTrace: () => Promise<void>;
  runSurface: () => Promise<void>;
  clearError: () => void;
}

function parseCells(text: string): Array<number | string | null> {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => {
      if (s.toLowerCase() === 'nan' || s.toLowerCase() === 'none' || s === '?') return null;
      const n = Number(s);
      return s !== '' && Number.isFinite(n) ? n : s;
    });
}

export const usePipelineSliceStore = create<PipelineSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  featureNames: [],
  rowInput: '55000, 34, Bangkok, Rent',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  coordSpace: 'raw',
  waterfallMode: 'grouped',
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
  setCoordSpace: (v) => set({ coordSpace: v }),
  setWaterfallMode: (v) => set({ waterfallMode: v }),
  setShowSurface: (v) => set({ showSurface: v }),
  setShowWireframe: (v) => set({ showWireframe: v }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  bindModel: (args) => {
    set({
      modelId: args.modelId,
      algorithmKey: args.algorithmKey,
      featureNames: args.featureNames,
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
      set({ sliceError: 'Load a Pipeline model first.' });
      return;
    }
    const values = parseCells(s.rowInput);
    if (values.length !== s.featureNames.length) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} values (use NaN for missing).` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchPipelineTracePreview(s.modelId, {
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
      set({ sliceError: 'Load a Pipeline model first.' });
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
      if (typeof v === 'number' && Number.isFinite(v)) {
        const d = Math.abs(v) * 0.5 + 1;
        return [v - d, v + d];
      }
      return fallback;
    };
    const [xMin, xMax] = span(s.axisX, [15, 120]);
    const [yMin, yMax] = span(s.axisY, [20, 65]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchPipelineSurfacePreview(s.modelId, {
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

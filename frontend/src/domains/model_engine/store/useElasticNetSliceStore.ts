import { create } from 'zustand';
import {
  fetchElasticNetSurfacePreview,
  fetchElasticNetTracePreview,
  type ElasticNetRowTrace,
  type ElasticNetSurfacePreview,
} from '../api/elasticNetApi';
import {
  computeElasticNetPhaseShift,
} from '../../stage3d/components/surfaces/elasticNetSurfaceMath';

export interface ElasticNetSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  featureNames: string[];
  weights: (number | null)[];
  intercept: number | null;
  alpha: number | null;
  l1Ratio: number | null;
  taskCount: number;
  rowInput: string;
  actualInput: string;
  taskIdx: number;
  axisX: number;
  axisY: number;
  traces: ElasticNetRowTrace[];
  surface: ElasticNetSurfacePreview | null;
  phaseShift: number | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setActualInput: (v: string) => void;
  setTaskIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: {
    modelId: string;
    algorithmKey: string;
    featureNames: string[];
    weights: (number | null)[];
    intercept: number | null;
    alpha?: number | null;
    l1Ratio?: number | null;
    taskCount?: number;
  }) => void;
  runTrace: () => Promise<void>;
  runSurface: () => Promise<void>;
  clearError: () => void;
}

function parseNumbers(text: string): number[] {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
}

export const useElasticNetSliceStore = create<ElasticNetSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  featureNames: [],
  weights: [],
  intercept: null,
  alpha: null,
  l1Ratio: null,
  taskCount: 1,
  rowInput: '6.2, 12.5, 320, 0.1',
  actualInput: '',
  taskIdx: 0,
  axisX: 0,
  axisY: 1,
  traces: [],
  surface: null,
  phaseShift: null,
  isTracing: false,
  isSurfacing: false,
  sliceError: null,
  showTelemetry: false,

  setRowInput: (v) => set({ rowInput: v }),
  setActualInput: (v) => set({ actualInput: v }),
  setTaskIdx: (v) => set({ taskIdx: Math.max(0, v) }),
  setAxisX: (x) => {
    const { axisX, axisY } = get();
    if (x === axisY) set({ axisX: axisY, axisY: axisX });
    else set({ axisX: x });
  },
  setAxisY: (y) => {
    const { axisX, axisY } = get();
    if (y === axisX) set({ axisX: axisY, axisY: axisX });
    else set({ axisY: y });
  },
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  bindModel: (args) =>
    set({
      modelId: args.modelId,
      algorithmKey: args.algorithmKey,
      featureNames: args.featureNames,
      weights: args.weights,
      intercept: args.intercept,
      alpha: args.alpha ?? null,
      l1Ratio: args.l1Ratio ?? null,
      taskCount: args.taskCount ?? 1,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      taskIdx: 0,
      traces: [],
      surface: null,
      phaseShift: null,
      sliceError: null,
    }),

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load an elastic-net model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchElasticNetTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_idx: s.taskIdx,
      });
      const weights = s.weights.map((w) => (typeof w === 'number' ? w : 0));
      const phaseShift = computeElasticNetPhaseShift(weights, s.intercept ?? 0, values, s.axisX, s.axisY);
      set({ traces: res.traces, phaseShift });
    } catch (err) {
      set({ sliceError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isTracing: false });
    }
  },

  runSurface: async () => {
    const s = get();
    if (!s.modelId || s.featureNames.length === 0) {
      set({ sliceError: 'Load an elastic-net model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    const medians: Record<number, number> = {};
    s.featureNames.forEach((_, i) => {
      if (i !== s.axisX && i !== s.axisY) medians[i] = Number.isFinite(values[i]) ? values[i] : 0;
    });
    const span = (i: number, fallback: [number, number]): [number, number] => {
      const v = Number.isFinite(values[i]) ? values[i] : null;
      if (v === null || v === 0) return fallback;
      const d = Math.abs(v) * 0.6;
      return [v - d, v + d];
    };
    const [xMin, xMax] = span(s.axisX, [-3, 3]);
    const [yMin, yMax] = span(s.axisY, [0, 1]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchElasticNetSurfacePreview(s.modelId, {
        axis_x_feature_index: s.axisX,
        axis_y_feature_index: s.axisY,
        x_min: xMin,
        x_max: xMax,
        y_min: yMin,
        y_max: yMax,
        resolution: 60,
        fixed_feature_medians: medians,
        target_idx: s.taskIdx,
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

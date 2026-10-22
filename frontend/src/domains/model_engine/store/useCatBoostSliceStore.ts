import { create } from 'zustand';
import {
  fetchCatboostSurfacePreview,
  fetchCatboostTracePreview,
  type CatboostRowTrace,
  type CatboostSurfacePreview,
} from '../api/catboostApi';

export type CatboostImportanceMetric = 'prediction_values_change' | 'gain' | 'loss_function_change' | 'split';

export interface CatBoostSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  isRegressor: boolean;
  featureNames: string[];
  categoricalIdx: number[];
  classCount: number;
  rowInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  selectedRound: number;
  selectedTree: number;
  importanceMetric: CatboostImportanceMetric;
  showForceVectors: boolean;
  showSplitLines: boolean;
  showCategoricalTicks: boolean;
  showWireframe: boolean;
  traces: CatboostRowTrace[];
  surface: CatboostSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setSelectedRound: (v: number) => void;
  setSelectedTree: (v: number) => void;
  setImportanceMetric: (v: CatboostImportanceMetric) => void;
  setShowForceVectors: (v: boolean) => void;
  setShowSplitLines: (v: boolean) => void;
  setShowCategoricalTicks: (v: boolean) => void;
  setShowWireframe: (v: boolean) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: {
    modelId: string;
    algorithmKey: string;
    isRegressor: boolean;
    featureNames: string[];
    categoricalIdx?: number[];
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

export const useCatBoostSliceStore = create<CatBoostSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  isRegressor: false,
  featureNames: [],
  categoricalIdx: [],
  classCount: 2,
  rowInput: '55, Bangkok',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  selectedRound: 0,
  selectedTree: 0,
  importanceMetric: 'prediction_values_change',
  showForceVectors: true,
  showSplitLines: true,
  showCategoricalTicks: true,
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
  setSelectedRound: (v) => set({ selectedRound: Math.max(0, v) }),
  setSelectedTree: (v) => set({ selectedTree: Math.max(0, v) }),
  setImportanceMetric: (v) => set({ importanceMetric: v }),
  setShowForceVectors: (v) => set({ showForceVectors: v }),
  setShowSplitLines: (v) => set({ showSplitLines: v }),
  setShowCategoricalTicks: (v) => set({ showCategoricalTicks: v }),
  setShowWireframe: (v) => set({ showWireframe: v }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  bindModel: (args) => {
    const count = args.classCount && args.classCount > 0 ? args.classCount : 2;
    set({
      modelId: args.modelId,
      algorithmKey: args.algorithmKey,
      isRegressor: args.isRegressor,
      featureNames: args.featureNames,
      categoricalIdx: args.categoricalIdx ?? [],
      classCount: count,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      classIdx: 0,
      selectedRound: 0,
      selectedTree: 0,
      traces: [],
      surface: null,
      sliceError: null,
    });
  },

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load a CatBoost model first.' });
      return;
    }
    const values = parseCells(s.rowInput);
    if (values.length !== s.featureNames.length) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} values (numbers or category labels).` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchCatboostTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_idx: s.classIdx,
      });
      const mech = res.traces[0]?.extra_mechanics;
      set({
        traces: res.traces,
        categoricalIdx: mech?.categorical_feature_idx ?? s.categoricalIdx,
        classCount:
          mech && Array.isArray(mech.classes) && mech.classes.length > 0
            ? mech.classes.length
            : s.classCount,
        selectedRound: mech ? Math.max(0, mech.n_iterations - 1) : 0,
        selectedTree: 0,
      });
    } catch (err) {
      set({ sliceError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isTracing: false });
    }
  },

  runSurface: async () => {
    const s = get();
    if (!s.modelId || s.featureNames.length === 0) {
      set({ sliceError: 'Load a CatBoost model first.' });
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
    const [xMin, xMax] = span(s.axisX, [20, 80]);
    const [yMin, yMax] = span(s.axisY, [18, 70]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchCatboostSurfacePreview(s.modelId, {
        axis_x_feature_index: s.axisX,
        axis_y_feature_index: s.axisY,
        x_min: xMin,
        x_max: xMax,
        y_min: yMin,
        y_max: yMax,
        resolution: 60,
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

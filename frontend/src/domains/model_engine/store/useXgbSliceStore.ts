import { create } from 'zustand';
import {
  fetchXgbSurfacePreview,
  fetchXgbTracePreview,
  type XgbRowTrace,
  type XgbSurfacePreview,
} from '../api/xgboostApi';

export type XgbImportanceMetric = 'gain' | 'weight' | 'cover' | 'total_gain' | 'total_cover';

export interface XgbSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  isRegressor: boolean;
  featureNames: string[];
  classCount: number;
  rowInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  selectedRound: number;
  selectedTree: number;
  importanceMetric: XgbImportanceMetric;
  showForceVectors: boolean;
  showSplitLines: boolean;
  showWireframe: boolean;
  traces: XgbRowTrace[];
  surface: XgbSurfacePreview | null;
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
  setImportanceMetric: (v: XgbImportanceMetric) => void;
  setShowForceVectors: (v: boolean) => void;
  setShowSplitLines: (v: boolean) => void;
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

function parseNumbers(text: string): number[] {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
}

export const useXgbSliceStore = create<XgbSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  isRegressor: false,
  featureNames: [],
  classCount: 2,
  rowInput: '0.5, -0.3',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  selectedRound: 0,
  selectedTree: 0,
  importanceMetric: 'gain',
  showForceVectors: true,
  showSplitLines: true,
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
      set({ sliceError: 'Load an XGBoost model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchXgbTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_idx: s.classIdx,
      });
      const mech = res.traces[0]?.extra_mechanics;
      set({
        traces: res.traces,
        classCount:
          mech && Array.isArray(mech.classes) && mech.classes.length > 0
            ? mech.classes.length
            : s.classCount,
        selectedRound: mech ? Math.max(0, mech.n_boost_rounds - 1) : 0,
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
      set({ sliceError: 'Load an XGBoost model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    const medians: Record<number, number> = {};
    s.featureNames.forEach((_, i) => {
      if (i !== s.axisX && i !== s.axisY) medians[i] = Number.isFinite(values[i]) ? values[i] : 0;
    });
    const span = (i: number, fallback: [number, number]): [number, number] => {
      const v = Number.isFinite(values[i]) ? values[i] : null;
      if (v === null) return fallback;
      const d = Math.abs(v) * 0.9 + 0.8;
      return [v - d, v + d];
    };
    const [xMin, xMax] = span(s.axisX, [-4, 6]);
    const [yMin, yMax] = span(s.axisY, [-3, 5]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchXgbSurfacePreview(s.modelId, {
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

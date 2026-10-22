import { create } from 'zustand';
import {
  fetchExtraTreesSurfacePreview,
  fetchExtraTreesTracePreview,
  type ExtraTreesRowTrace,
  type ExtraTreesSurfacePreview,
} from '../api/extraTreesApi';

export type ExtraTreesSurfaceMode = 'mean' | 'uncertainty';

export interface ExtraTreesSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  isRegressor: boolean;
  featureNames: string[];
  classCount: number;
  rowInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  selectedTree: number;
  surfaceMode: ExtraTreesSurfaceMode;
  showVotes: boolean;
  showHalo: boolean;
  showRug: boolean;
  showWireframe: boolean;
  traces: ExtraTreesRowTrace[];
  surface: ExtraTreesSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setSelectedTree: (v: number) => void;
  setSurfaceMode: (m: ExtraTreesSurfaceMode) => void;
  setShowVotes: (v: boolean) => void;
  setShowHalo: (v: boolean) => void;
  setShowRug: (v: boolean) => void;
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

export const useExtraTreesSliceStore = create<ExtraTreesSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  isRegressor: false,
  featureNames: [],
  classCount: 2,
  rowInput: '0.5, -0.3',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  selectedTree: 0,
  surfaceMode: 'mean',
  showVotes: true,
  showHalo: true,
  showRug: true,
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
  setSelectedTree: (v) => set({ selectedTree: Math.max(0, v) }),
  setSurfaceMode: (m) => set({ surfaceMode: m }),
  setShowVotes: (v) => set({ showVotes: v }),
  setShowHalo: (v) => set({ showHalo: v }),
  setShowRug: (v) => set({ showRug: v }),
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
      selectedTree: 0,
      traces: [],
      surface: null,
      sliceError: null,
    });
  },

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load an extra-trees model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchExtraTreesTracePreview(s.modelId, {
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
      set({ sliceError: 'Load an extra-trees model first.' });
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
      const res = await fetchExtraTreesSurfacePreview(s.modelId, {
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

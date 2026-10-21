import { create } from 'zustand';
import {
  fetchSvmSurfacePreview,
  fetchSvmTracePreview,
  type SvmRowTrace,
  type SvmSurfacePreview,
} from '../api/svmApi';

export interface SvmSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  featureNames: string[];
  isClassifier: boolean;
  isOneClass: boolean;
  classCount: number;
  rowInput: string;
  actualInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  showPillars: boolean;
  showTethers: boolean;
  showContours: boolean;
  traces: SvmRowTrace[];
  surface: SvmSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setActualInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setShowPillars: (v: boolean) => void;
  setShowTethers: (v: boolean) => void;
  setShowContours: (v: boolean) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: {
    modelId: string;
    algorithmKey: string;
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

export const useSvmSliceStore = create<SvmSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  featureNames: [],
  isClassifier: true,
  isOneClass: false,
  classCount: 2,
  rowInput: '0.4, -0.2, 0.5, 0.1',
  actualInput: '',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  showPillars: true,
  showTethers: true,
  showContours: true,
  traces: [],
  surface: null,
  isTracing: false,
  isSurfacing: false,
  sliceError: null,
  showTelemetry: false,

  setRowInput: (v) => set({ rowInput: v }),
  setActualInput: (v) => set({ actualInput: v }),
  setClassIdx: (v) => set({ classIdx: Math.max(0, v) }),
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
  setShowPillars: (v) => set({ showPillars: v }),
  setShowTethers: (v) => set({ showTethers: v }),
  setShowContours: (v) => set({ showContours: v }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  bindModel: (args) =>
    set({
      modelId: args.modelId,
      algorithmKey: args.algorithmKey,
      featureNames: args.featureNames,
      isClassifier: args.algorithmKey === 'svm_classifier',
      isOneClass: args.algorithmKey === 'svm_oneclass',
      classCount: args.classCount ?? 2,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      classIdx: 0,
      traces: [],
      surface: null,
      sliceError: null,
    }),

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load an SVM model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchSvmTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_idx: s.classIdx,
      });
      const mech = res.traces[0]?.extra_mechanics;
      set({
        traces: res.traces,
        classCount:
          mech && Array.isArray(mech.classes) && mech.classes.length > 2
            ? mech.classes.length
            : s.classCount,
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
      set({ sliceError: 'Load an SVM model first.' });
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
      const res = await fetchSvmSurfacePreview(s.modelId, {
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

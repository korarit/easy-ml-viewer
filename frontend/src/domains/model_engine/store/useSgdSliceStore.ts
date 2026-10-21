import { create } from 'zustand';
import {
  fetchSgdSurfacePreview,
  fetchSgdTracePreview,
  type SgdRowTrace,
  type SgdSurfacePreview,
} from '../api/sgdApi';
import {
  computeSgdPhaseShift,
} from '../../stage3d/components/surfaces/sgdSurfaceMath';

export type SgdMetricMode = 'raw' | 'geometric' | 'proba';

export interface SgdSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  featureNames: string[];
  weights: (number | null)[];
  intercept: number | null;
  loss: string | null;
  hasNativeProba: boolean | null;
  isClassifier: boolean;
  classCount: number;
  rowInput: string;
  actualInput: string;
  classIdx: number;
  metricMode: SgdMetricMode;
  showCorridor: boolean;
  showLaser: boolean;
  axisX: number;
  axisY: number;
  traces: SgdRowTrace[];
  surface: SgdSurfacePreview | null;
  phaseShift: number | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setActualInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setMetricMode: (v: SgdMetricMode) => void;
  setShowCorridor: (v: boolean) => void;
  setShowLaser: (v: boolean) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: {
    modelId: string;
    algorithmKey: string;
    featureNames: string[];
    weights: (number | null)[];
    intercept: number | null;
    loss?: string | null;
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

export const useSgdSliceStore = create<SgdSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  featureNames: [],
  weights: [],
  intercept: null,
  loss: null,
  hasNativeProba: null,
  isClassifier: true,
  classCount: 2,
  rowInput: '55.0, 3.2, 22.0, 1.1',
  actualInput: '',
  classIdx: 0,
  metricMode: 'raw',
  showCorridor: true,
  showLaser: true,
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
  setClassIdx: (v) => set({ classIdx: Math.max(0, v) }),
  setMetricMode: (v) => set({ metricMode: v }),
  setShowCorridor: (v) => set({ showCorridor: v }),
  setShowLaser: (v) => set({ showLaser: v }),
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
      loss: args.loss ?? null,
      hasNativeProba: null,
      isClassifier: args.algorithmKey === 'sgd_classifier',
      classCount: args.classCount ?? 2,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      classIdx: 0,
      metricMode: 'raw',
      showCorridor: true,
      showLaser: true,
      traces: [],
      surface: null,
      phaseShift: null,
      sliceError: null,
    }),

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load an SGD model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchSgdTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_idx: s.classIdx,
      });
      const weights = s.weights.map((w) => (typeof w === 'number' ? w : 0));
      const phaseShift = computeSgdPhaseShift(weights, s.intercept ?? 0, values, s.axisX, s.axisY);
      const mech = res.traces[0]?.extra_mechanics;
      set({
        traces: res.traces,
        phaseShift,
        hasNativeProba:
          mech && typeof mech.has_native_proba === 'boolean' ? mech.has_native_proba : s.hasNativeProba,
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
      set({ sliceError: 'Load an SGD model first.' });
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
      const res = await fetchSgdSurfacePreview(s.modelId, {
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

import { create } from 'zustand';
import {
  fetchLogisticSurfacePreview,
  fetchLogisticTracePreview,
  type LogisticRowTrace,
  type LogisticSurfacePreview,
} from '../api/logisticApi';
import { computeSigmoidPhaseShift } from '../../stage3d/components/surfaces/sigmoidSurfaceMath';

export type LogisticZMode = 'probability' | 'logodds';

export interface LogisticSliceState {
  modelId: string | null;
  featureNames: string[];
  weights: (number | null)[];
  intercept: number | null;
  classes: Array<string | number>;
  solver: string | null;
  penalty: string | null;
  cValue: number | null;
  rowInput: string;
  threshold: number;
  zMode: LogisticZMode;
  targetClassIdx: number;
  axisX: number;
  axisY: number;
  traces: LogisticRowTrace[];
  surface: LogisticSurfacePreview | null;
  phaseShift: number | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setThreshold: (v: number) => void;
  setZMode: (v: LogisticZMode) => void;
  setTargetClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: {
    modelId: string;
    featureNames: string[];
    weights: (number | null)[];
    intercept: number | null;
    classes: Array<string | number>;
    solver?: string | null;
    penalty?: string | null;
    cValue?: number | null;
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

export const useLogisticSliceStore = create<LogisticSliceState>((set, get) => ({
  modelId: null,
  featureNames: [],
  weights: [],
  intercept: null,
  classes: [],
  solver: null,
  penalty: null,
  cValue: null,
  rowInput: '55000, 34, 0.42, 680',
  threshold: 0.5,
  zMode: 'probability',
  targetClassIdx: 1,
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
  setThreshold: (v) =>
    set({ threshold: Math.min(0.95, Math.max(0.05, v)) }),
  setZMode: (v) => set({ zMode: v }),
  setTargetClassIdx: (v) => set({ targetClassIdx: Math.max(0, v) }),
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
      featureNames: args.featureNames,
      weights: args.weights,
      intercept: args.intercept,
      classes: args.classes,
      solver: args.solver ?? null,
      penalty: args.penalty ?? null,
      cValue: args.cValue ?? null,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      targetClassIdx: args.classes.length > 1 ? 1 : 0,
      traces: [],
      surface: null,
      phaseShift: null,
      sliceError: null,
    }),

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load a logistic model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchLogisticTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_class_idx: s.targetClassIdx,
      });
      const weights = s.weights.map((w) => (typeof w === 'number' ? w : 0));
      const phaseShift = computeSigmoidPhaseShift(
        weights,
        s.intercept ?? 0,
        values,
        s.axisX,
        s.axisY,
      );
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
      set({ sliceError: 'Load a logistic model first.' });
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
    const [yMin, yMax] = span(s.axisY, [-3, 3]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchLogisticSurfacePreview(s.modelId, {
        axis_x_feature_index: s.axisX,
        axis_y_feature_index: s.axisY,
        x_min: xMin,
        x_max: xMax,
        y_min: yMin,
        y_max: yMax,
        resolution: 60,
        fixed_feature_medians: medians,
        target_class_idx: s.targetClassIdx,
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

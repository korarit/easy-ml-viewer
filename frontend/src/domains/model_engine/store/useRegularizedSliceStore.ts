import { create } from 'zustand';
import {
  fetchRegularizedSurfacePreview,
  fetchRegularizedTracePreview,
  type RegularizedRowTrace,
  type RegularizedSurfacePreview,
} from '../api/regularizedApi';
import { computeRegularizedPhaseShift } from '../../stage3d/components/surfaces/regularizedSurfaceMath';

export interface RegularizedSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  featureNames: string[];
  weights: (number | null)[];
  intercept: number | null;
  penaltyType: string | null;
  alpha: number | null;
  targetCount: number;
  rowInput: string;
  actualInput: string;
  targetIdx: number;
  axisX: number;
  axisY: number;
  traces: RegularizedRowTrace[];
  surface: RegularizedSurfacePreview | null;
  phaseShift: number | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setActualInput: (v: string) => void;
  setTargetIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (args: {
    modelId: string;
    algorithmKey: string;
    featureNames: string[];
    weights: (number | null)[];
    intercept: number | null;
    penaltyType?: string | null;
    alpha?: number | null;
    targetCount?: number;
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

export const useRegularizedSliceStore = create<RegularizedSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  featureNames: [],
  weights: [],
  intercept: null,
  penaltyType: null,
  alpha: null,
  targetCount: 1,
  rowInput: '120, 3, 8.5, 0.2',
  actualInput: '',
  targetIdx: 0,
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
  setTargetIdx: (v) => set({ targetIdx: Math.max(0, v) }),
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
      penaltyType: args.penaltyType ?? null,
      alpha: args.alpha ?? null,
      targetCount: args.targetCount ?? 1,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      targetIdx: 0,
      traces: [],
      surface: null,
      phaseShift: null,
      sliceError: null,
    }),

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load a ridge/lasso model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchRegularizedTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        feature_names: s.featureNames,
        target_idx: s.targetIdx,
      });
      const weights = s.weights.map((w) => (typeof w === 'number' ? w : 0));
      const phaseShift = computeRegularizedPhaseShift(weights, s.intercept ?? 0, values, s.axisX, s.axisY);
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
      set({ sliceError: 'Load a ridge/lasso model first.' });
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
      const res = await fetchRegularizedSurfacePreview(s.modelId, {
        axis_x_feature_index: s.axisX,
        axis_y_feature_index: s.axisY,
        x_min: xMin,
        x_max: xMax,
        y_min: yMin,
        y_max: yMax,
        resolution: 60,
        fixed_feature_medians: medians,
        target_idx: s.targetIdx,
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

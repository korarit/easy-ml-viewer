import { create } from 'zustand';
import {
  fetchSurfacePreview,
  fetchTracePreview,
  type RowTrace,
  type SurfacePreview,
} from '../api/linearSliceApi';

export interface LinearSliceState {
  modelId: string | null;
  featureNames: string[];
  weights: (number | null)[];
  intercept: number | null;
  rowInput: string;
  actualInput: string;
  targetIdx: number;
  targetCount: number;
  traces: RowTrace[];
  isTracing: boolean;
  sliceError: string | null;
  axisX: number;
  axisY: number;
  surface: SurfacePreview | null;
  isSurfacing: boolean;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setActualInput: (v: string) => void;
  setTargetIdx: (v: number) => void;
  setAxes: (x: number, y: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setShowTelemetry: (v: boolean) => void;
  bindModel: (modelId: string, featureNames: string[], weights: (number | null)[], intercept: number | null) => void;
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

export const useLinearSliceStore = create<LinearSliceState>((set, get) => ({
  modelId: null,
  featureNames: [],
  weights: [],
  intercept: null,
  rowInput: '80, 2, 2.2',
  actualInput: '3900000',
  targetIdx: 0,
  targetCount: 1,
  traces: [],
  isTracing: false,
  sliceError: null,
  axisX: 0,
  axisY: 2,
  surface: null,
  isSurfacing: false,
  showTelemetry: false,

  setRowInput: (v) => set({ rowInput: v }),
  setActualInput: (v) => set({ actualInput: v }),
  setTargetIdx: (v) => set({ targetIdx: v }),
  setAxes: (x, y) => {
    if (x === y) {
      // Edge case #6: same variable on both axes -> swap automatically.
      const prev = get();
      set({ axisX: prev.axisY, axisY: prev.axisX });
    } else set({ axisX: x, axisY: y });
  },
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

  bindModel: (modelId, featureNames, weights, intercept) =>
    set({
      modelId,
      featureNames,
      weights,
      intercept,
      axisX: 0,
      axisY: featureNames.length > 1 ? Math.min(2, featureNames.length - 1) : 0,
      traces: [],
      surface: null,
      sliceError: null,
      targetIdx: 0,
    }),

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load a linear model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    const actuals = s.actualInput.trim() === '' ? undefined : parseNumbers(s.actualInput);
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchTracePreview(s.modelId, {
        rows: [values],
        row_indices: [0],
        actual_targets: actuals,
        feature_names: s.featureNames,
        target_idx: s.targetIdx,
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
      set({ sliceError: 'Load a linear model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    const medians: Record<number, number> = {};
    s.featureNames.forEach((_, i) => {
      if (i !== s.axisX && i !== s.axisY) medians[i] = Number.isFinite(values[i]) ? values[i] : 0;
    });
    // Axis ranges: pad around the current row value (fallback 0..1).
    const span = (i: number, fallback: [number, number]): [number, number] => {
      const v = Number.isFinite(values[i]) ? values[i] : null;
      if (v === null || v === 0) return fallback;
      const d = Math.abs(v) * 0.6;
      return [v - d, v + d];
    };
    const [xMin, xMax] = span(s.axisX, [0, 1]);
    const [yMin, yMax] = span(s.axisY, [0, 1]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchSurfacePreview(s.modelId, {
        axis_x_feature_index: s.axisX,
        axis_y_feature_index: s.axisY,
        x_min: xMin,
        x_max: xMax,
        y_min: yMin,
        y_max: yMax,
        resolution: 50,
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

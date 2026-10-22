import { create } from 'zustand';
import {
  fetchDiscreteNbSurfacePreview,
  fetchDiscreteNbTracePreview,
  type DiscreteNbRowTrace,
  type DiscreteNbSurfacePreview,
} from '../api/discreteNbApi';
import type { PresenceFilter } from '../../stage3d/components/surfaces/discreteNbSurfaceMath';

export interface DiscreteNbSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  featureNames: string[];
  classCount: number;
  rowInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  showTowers: boolean;
  showWalls: boolean;
  presenceFilter: PresenceFilter;
  traces: DiscreteNbRowTrace[];
  surface: DiscreteNbSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setShowTowers: (v: boolean) => void;
  setShowWalls: (v: boolean) => void;
  setPresenceFilter: (v: PresenceFilter) => void;
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

export const useDiscreteNbSliceStore = create<DiscreteNbSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  featureNames: [],
  classCount: 2,
  rowInput: '3, 2, 1, 0, 0, 1',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  showTowers: true,
  showWalls: true,
  presenceFilter: 'all',
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
  setShowTowers: (v) => set({ showTowers: v }),
  setShowWalls: (v) => set({ showWalls: v }),
  setPresenceFilter: (v) => set({ presenceFilter: v }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  bindModel: (args) => {
    const count = args.classCount && args.classCount > 0 ? args.classCount : 2;
    set({
      modelId: args.modelId,
      algorithmKey: args.algorithmKey,
      featureNames: args.featureNames,
      classCount: count,
      axisX: 0,
      axisY: args.featureNames.length > 1 ? 1 : 0,
      classIdx: 0,
      presenceFilter: 'all',
      traces: [],
      surface: null,
      sliceError: null,
    });
  },

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load a discrete Naive Bayes model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchDiscreteNbTracePreview(s.modelId, {
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
      set({ sliceError: 'Load a discrete Naive Bayes model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    const medians: Record<number, number> = {};
    s.featureNames.forEach((_, i) => {
      if (i !== s.axisX && i !== s.axisY) medians[i] = Number.isFinite(values[i]) ? Math.max(0, values[i]) : 0;
    });
    const span = (i: number, fallback: [number, number]): [number, number] => {
      const v = Number.isFinite(values[i]) ? Math.max(0, values[i]) : null;
      // Count domain never dips below zero (blueprint edge #1).
      if (v === null) return fallback;
      const d = Math.abs(v) * 0.9 + 0.5;
      return [Math.max(0, v - d), v + d];
    };
    const [xMin, xMax] = span(s.axisX, [0, 8]);
    const [yMin, yMax] = span(s.axisY, [0, 6]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchDiscreteNbSurfacePreview(s.modelId, {
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

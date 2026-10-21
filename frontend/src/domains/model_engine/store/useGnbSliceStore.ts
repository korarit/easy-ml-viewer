import { create } from 'zustand';
import {
  fetchGnbSurfacePreview,
  fetchGnbTracePreview,
  type GnbRowTrace,
  type GnbSurfacePreview,
} from '../api/gnbApi';

export type GnbSurfaceMode = 'generative' | 'posterior';

export interface GnbSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  featureNames: string[];
  classCount: number;
  rowInput: string;
  actualInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  surfaceMode: GnbSurfaceMode;
  visibleHills: number[];
  showPins: boolean;
  showRings: boolean;
  traces: GnbRowTrace[];
  surface: GnbSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setActualInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setSurfaceMode: (v: GnbSurfaceMode) => void;
  toggleHill: (k: number) => void;
  setShowPins: (v: boolean) => void;
  setShowRings: (v: boolean) => void;
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

export const useGnbSliceStore = create<GnbSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  featureNames: [],
  classCount: 2,
  rowInput: '5.9, 3.0, 4.2, 1.3',
  actualInput: '',
  classIdx: 0,
  axisX: 2,
  axisY: 3,
  surfaceMode: 'generative',
  visibleHills: [],
  showPins: true,
  showRings: true,
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
    if (y === axisX) set({ axisX: axisY, axisY: axisY });
    else set({ axisY: y });
  },
  setSurfaceMode: (v) => set({ surfaceMode: v }),
  toggleHill: (k) => {
    const { visibleHills, surface } = get();
    const order = surface?.extra_overlays?.hill_class_order ?? [];
    const all = order.length > 0 ? order : visibleHills;
    if (visibleHills.includes(k)) {
      if (visibleHills.length <= 1) return;
      set({ visibleHills: visibleHills.filter((h) => h !== k) });
    } else {
      set({ visibleHills: [...visibleHills, k].filter((h) => all.includes(h)).sort((a, b) => a - b) });
    }
  },
  setShowPins: (v) => set({ showPins: v }),
  setShowRings: (v) => set({ showRings: v }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  bindModel: (args) => {
    const count = args.classCount && args.classCount > 0 ? args.classCount : 2;
    set({
      modelId: args.modelId,
      algorithmKey: args.algorithmKey,
      featureNames: args.featureNames,
      classCount: count,
      axisX: args.featureNames.length > 2 ? 2 : 0,
      axisY: args.featureNames.length > 3 ? 3 : args.featureNames.length > 1 ? 1 : 0,
      classIdx: 0,
      surfaceMode: 'generative',
      visibleHills: Array.from({ length: count }, (_, i) => i),
      traces: [],
      surface: null,
      sliceError: null,
    });
  },

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load a GaussianNB model first.' });
      return;
    }
    const values = parseNumbers(s.rowInput);
    if (values.length !== s.featureNames.length || values.some((v) => !Number.isFinite(v))) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} finite numbers.` });
      return;
    }
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchGnbTracePreview(s.modelId, {
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
        visibleHills:
          mech && Array.isArray(mech.classes) && mech.classes.length > 0
            ? mech.classes.map((_, i) => i)
            : s.visibleHills,
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
      set({ sliceError: 'Load a GaussianNB model first.' });
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
      const d = Math.abs(v) * 0.9 + 0.5;
      return [v - d, v + d];
    };
    const [xMin, xMax] = span(s.axisX, [0, 7]);
    const [yMin, yMax] = span(s.axisY, [0, 2.6]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchGnbSurfacePreview(s.modelId, {
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
      const order = res.extra_overlays?.hill_class_order ?? [];
      set({
        surface: res,
        visibleHills: order.length > 0 ? [...order] : s.visibleHills,
      });
    } catch (err) {
      set({ sliceError: err instanceof Error ? err.message : String(err) });
    } finally {
      set({ isSurfacing: false });
    }
  },

  clearError: () => set({ sliceError: null }),
}));

import { create } from 'zustand';
import {
  fetchHistGbmSurfacePreview,
  fetchHistGbmTracePreview,
  type HistGbmRowTrace,
  type HistGbmSurfacePreview,
} from '../api/histGbmApi';

export interface HistGbmSliceState {
  modelId: string | null;
  algorithmKey: string | null;
  isRegressor: boolean;
  featureNames: string[];
  classCount: number;
  rowInput: string;
  classIdx: number;
  axisX: number;
  axisY: number;
  selectedStage: number;
  showSpiral: boolean;
  showBinGrid: boolean;
  showWireframe: boolean;
  traces: HistGbmRowTrace[];
  surface: HistGbmSurfacePreview | null;
  isTracing: boolean;
  isSurfacing: boolean;
  sliceError: string | null;
  showTelemetry: boolean;
  setRowInput: (v: string) => void;
  setClassIdx: (v: number) => void;
  setAxisX: (x: number) => void;
  setAxisY: (y: number) => void;
  setSelectedStage: (v: number) => void;
  setShowSpiral: (v: boolean) => void;
  setShowBinGrid: (v: boolean) => void;
  setShowWireframe: (v: boolean) => void;
  setShowTelemetry: (v: boolean) => void;
  injectNan: (featureIdx: number) => void;
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

/** Row tokens: finite numbers or NaN literals ("NaN", "nan", "null", "?"). */
export function parseHistGbmRow(text: string): (number | null)[] | null {
  const tokens = text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (tokens.length === 0) return null;
  return tokens.map((t) => {
    if (/^(nan|n\/a|null|none|\?|missing)$/i.test(t)) return NaN;
    const v = Number(t);
    return Number.isFinite(v) ? v : null;
  });
}

export const useHistGbmSliceStore = create<HistGbmSliceState>((set, get) => ({
  modelId: null,
  algorithmKey: null,
  isRegressor: false,
  featureNames: [],
  classCount: 2,
  rowInput: '0.5, -0.3',
  classIdx: 0,
  axisX: 0,
  axisY: 1,
  selectedStage: 0,
  showSpiral: true,
  showBinGrid: true,
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
  setSelectedStage: (v) => set({ selectedStage: Math.max(0, v) }),
  setShowSpiral: (v) => set({ showSpiral: v }),
  setShowBinGrid: (v) => set({ showBinGrid: v }),
  setShowWireframe: (v) => set({ showWireframe: v }),
  setShowTelemetry: (v) => set({ showTelemetry: v }),

  injectNan: (featureIdx) => {
    const s = get();
    const parsed = parseHistGbmRow(s.rowInput) ?? [];
    const vals: string[] = [];
    for (let i = 0; i < s.featureNames.length; i += 1) {
      if (i === featureIdx) vals.push('NaN');
      else {
        const cur = parsed[i];
        vals.push(typeof cur === 'number' && Number.isFinite(cur) ? String(cur) : '0');
      }
    }
    set({ rowInput: vals.join(', ') });
  },

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
      selectedStage: 0,
      traces: [],
      surface: null,
      sliceError: null,
    });
  },

  runTrace: async () => {
    const s = get();
    if (!s.modelId) {
      set({ sliceError: 'Load a HistGBM model first.' });
      return;
    }
    const parsed = parseHistGbmRow(s.rowInput);
    if (!parsed || parsed.length !== s.featureNames.length || parsed.some((v) => v === null)) {
      set({ sliceError: `Row must hold exactly ${s.featureNames.length} values (numbers or NaN).` });
      return;
    }
    // Raw JSON NaN is not JSON-compliant — encode missing as "NaN" strings.
    const wireRow = (parsed as number[]).map((v) => (Number.isFinite(v) ? v : 'NaN'));
    set({ isTracing: true, sliceError: null });
    try {
      const res = await fetchHistGbmTracePreview(s.modelId, {
        rows: [wireRow],
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
        selectedStage: mech ? Math.max(0, mech.n_iter_total - 1) : 0,
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
      set({ sliceError: 'Load a HistGBM model first.' });
      return;
    }
    const parsed = parseHistGbmRow(s.rowInput) ?? [];
    const medians: Record<number, number> = {};
    s.featureNames.forEach((_, i) => {
      if (i === s.axisX || i === s.axisY) return;
      const v = parsed[i];
      medians[i] = typeof v === 'number' && Number.isFinite(v) ? v : 0;
    });
    const span = (i: number, fallback: [number, number]): [number, number] => {
      const v = parsed[i];
      if (typeof v !== 'number' || !Number.isFinite(v)) return fallback;
      const d = Math.abs(v) * 0.9 + 0.8;
      return [v - d, v + d];
    };
    const [xMin, xMax] = span(s.axisX, [-4, 6]);
    const [yMin, yMax] = span(s.axisY, [-3, 5]);
    set({ isSurfacing: true, sliceError: null });
    try {
      const res = await fetchHistGbmSurfacePreview(s.modelId, {
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

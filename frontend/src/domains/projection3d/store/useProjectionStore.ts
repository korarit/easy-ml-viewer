import { create } from 'zustand';
import { fetchContour2D, fetchSurfaceMesh } from '../api/projectionApi';
import type {
  AvailableFeature,
  ColormapName,
  Contour2DPayload,
  LodLevel,
  PinningMode,
  ProjectionMode,
  SurfaceMeshPayload,
  ZMode,
} from '../types/projection.types';
import { LOD_RESOLUTIONS } from '../types/projection.types';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import type { AxisScaleMeta } from '../../stage3d/api/stage3dApi';

interface ProjectionState {
  mode: ProjectionMode;
  xFeature: string;
  yFeature: string;
  zMode: ZMode;
  targetClassIndex: number;
  pinningMode: PinningMode;
  lodLevel: LodLevel;
  useRobustRange: boolean;
  colormap: ColormapName;
  opacity: number;
  wireframe: boolean;
  surfaceData: SurfaceMeshPayload | null;
  isLoading: boolean;
  projError: string | null;
  /** Phase 41 — dense 2D contour grid for the planar viewport. */
  contourData: Contour2DPayload | null;
  contourLoading: boolean;
  contourError: string | null;
  ensureDefaults: (features: AvailableFeature[]) => void;
  setXFeature: (name: string) => void;
  setYFeature: (name: string) => void;
  setAxes: (x: string, y: string) => void;
  swapAxes: () => void;
  setMode: (mode: ProjectionMode) => void;
  setZMode: (z: ZMode) => void;
  setPinningMode: (p: PinningMode) => void;
  setLodLevel: (lod: LodLevel) => void;
  setUseRobustRange: (on: boolean) => void;
  setColormap: (c: ColormapName) => void;
  setOpacity: (v: number) => void;
  setWireframe: (on: boolean) => void;
  triangleCount: () => number;
  fetchSurface: (params: {
    datasetId: string;
    modelId: string;
    mappingId: string;
    queueRowIndices: number[];
    currentFrameIndex: number | null;
    featureNames: string[];
  }) => Promise<void>;
  fetchContour: (params: {
    datasetId: string;
    modelId: string;
    mappingId: string;
    queueRowIndices: number[];
    currentFrameIndex: number | null;
  }) => Promise<void>;
  clearError: () => void;
}

let inFlightAbort: AbortController | null = null;
let contourAbort: AbortController | null = null;
/** Phase 41 — frontend contour cache so 2D↔3D toggles never refetch. */
const contourCache = new Map<string, Contour2DPayload>();

function contourKey(
  datasetId: string,
  modelId: string,
  mappingId: string,
  x: string,
  y: string,
  pinning: string,
  frame: number | null,
  robust: boolean,
): string {
  return [datasetId, modelId, mappingId, x, y, pinning, frame ?? -1, robust ? 1 : 0].join('|');
}

function buildAxisMeta(name: string, rawMin: number, rawMax: number, featureIndex: number | null): AxisScaleMeta {
  const lo = Number.isFinite(rawMin) ? rawMin : 0;
  const hi = Number.isFinite(rawMax) ? rawMax : 1;
  const span = hi - lo;
  const ticks = [-5, -2.5, 0, 2.5, 5];
  return {
    axis_name: name,
    feature_index: featureIndex,
    raw_min: lo,
    raw_max: hi,
    tick_world_positions: ticks,
    tick_raw_labels: ticks.map((w) => Math.round((lo + ((w + 5) / 10) * span) * 10000) / 10000),
  };
}

function topTwo(features: AvailableFeature[]): [string, string] {
  const ranked = [...features].sort(
    (a, b) => Math.abs(b.importance ?? 0) - Math.abs(a.importance ?? 0),
  );
  return [ranked[0]?.name ?? '', ranked[1]?.name ?? ranked[0]?.name ?? ''];
}

export const useProjectionStore = create<ProjectionState>((set, get) => ({
  mode: 'axis_explorer',
  xFeature: '',
  yFeature: '',
  zMode: 'prediction',
  targetClassIndex: 0,
  pinningMode: 'median',
  lodLevel: 'standard',
  useRobustRange: false,
  colormap: 'coolwarm',
  opacity: 0.78,
  wireframe: false,
  surfaceData: null,
  isLoading: false,
  projError: null,
  contourData: null,
  contourLoading: false,
  contourError: null,

  ensureDefaults: (features) => {
    if (features.length === 0) return;
    const s = get();
    const names = new Set(features.map((f) => f.name));
    const stillValid = !!s.xFeature && !!s.yFeature && s.xFeature !== s.yFeature && names.has(s.xFeature) && names.has(s.yFeature);
    if (stillValid) return;
    const [x, y] = topTwo(features);
    if (!x) return;
    set({ xFeature: x, yFeature: y || x, projError: null, surfaceData: null });
  },

  setXFeature: (name) => {
    const s = get();
    if (name === s.yFeature) {
      // Auto-swap (edge #3): keep axes distinct without an error state.
      set({ xFeature: name, yFeature: s.xFeature });
    } else {
      set({ xFeature: name });
    }
  },

  setYFeature: (name) => {
    const s = get();
    if (name === s.xFeature) {
      set({ xFeature: s.yFeature, yFeature: name });
    } else {
      set({ yFeature: name });
    }
  },

  setAxes: (x, y) => {
    if (x === y) {
      const prev = get();
      set({ xFeature: x, yFeature: prev.xFeature === x ? prev.yFeature : prev.xFeature });
      return;
    }
    set({ xFeature: x, yFeature: y });
  },

  swapAxes: () => {
    const s = get();
    set({ xFeature: s.yFeature, yFeature: s.xFeature });
  },

  setMode: (mode) => set({ mode }),
  setZMode: (z) => set({ zMode: z }),
  setPinningMode: (p) => set({ pinningMode: p }),
  setLodLevel: (lod) => set({ lodLevel: lod }),
  setUseRobustRange: (on) => set({ useRobustRange: on }),
  setColormap: (c) => set({ colormap: c }),
  setOpacity: (v) => set({ opacity: Math.min(1, Math.max(0.1, v)) }),
  setWireframe: (on) => set({ wireframe: on }),

  triangleCount: () => {
    const r = LOD_RESOLUTIONS[get().lodLevel];
    return 2 * (r - 1) * (r - 1);
  },

  fetchSurface: async ({ datasetId, modelId, mappingId, queueRowIndices, currentFrameIndex, featureNames }) => {
    const s = get();
    if (!datasetId || !modelId || !mappingId) return;
    if (s.mode === 'axis_explorer' && (!s.xFeature || !s.yFeature)) return;
    // Drop stale requests whose axes are not in the current mapping
    // (happens right after a dataset/model switch before ensureDefaults re-runs).
    if (s.mode === 'axis_explorer' && featureNames.length > 0) {
      if (!featureNames.includes(s.xFeature) || !featureNames.includes(s.yFeature)) return;
    }
    if (inFlightAbort) inFlightAbort.abort();
    const controller = new AbortController();
    inFlightAbort = controller;
    set({ isLoading: true, projError: null });
    try {
      const data = await fetchSurfaceMesh(
        {
          datasetId,
          modelId,
          mappingId,
          mode: s.mode,
          xFeature: s.xFeature,
          yFeature: s.yFeature,
          zMode: s.zMode,
          targetClassIndex: s.targetClassIndex,
          pinningMode: s.pinningMode,
          lodLevel: s.lodLevel,
          currentFrameIndex,
          useRobustRange: s.useRobustRange,
          selectedRowIndices: queueRowIndices,
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      set({ surfaceData: data });
      // Single render source of truth stays in the stage3d store.
      const xIdx = featureNames.indexOf(data.xFeature);
      const yIdx = featureNames.indexOf(data.yFeature);
      useStage3DStore.getState().applyProjectionFrame({
        datasetId,
        modelId,
        mappingId,
        featureNames,
        axisXFeatureIndex: xIdx >= 0 ? xIdx : 0,
        axisYFeatureIndex: yIdx >= 0 ? yIdx : Math.min(1, Math.max(featureNames.length - 1, 0)),
        axes: {
          x: buildAxisMeta(data.xFeature, data.xRange[0], data.xRange[1], xIdx >= 0 ? xIdx : null),
          y: buildAxisMeta(data.yFeature, data.yRange[0], data.yRange[1], yIdx >= 0 ? yIdx : null),
          z: buildAxisMeta(
            data.resolvedZMode === 'probability'
              ? `P(class)`
              : data.resolvedZMode === 'residual_error'
                ? 'Residual |y - ŷ|'
                : 'Predicted value',
            data.zRange[0],
            data.zRange[1],
            null,
          ),
        },
        points: data.trajectoryPoints,
        surfaceResolution: data.gridResolution,
        surfaceVisualType: s.mode === 'pca_3d' ? 'pca_manifold' : 'axis_slice_surface',
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      set({ projError: err instanceof Error ? err.message : String(err) });
    } finally {
      if (inFlightAbort === controller) inFlightAbort = null;
      set({ isLoading: false });
    }
  },

  fetchContour: async ({ datasetId, modelId, mappingId, queueRowIndices, currentFrameIndex }) => {
    const s = get();
    if (!datasetId || !modelId || !mappingId) return;
    if (s.mode !== 'axis_explorer' || !s.xFeature || !s.yFeature) return;
    const key = contourKey(
      datasetId, modelId, mappingId, s.xFeature, s.yFeature,
      s.pinningMode, s.pinningMode === 'current_row' ? currentFrameIndex : null,
      s.useRobustRange,
    );
    const cached = contourCache.get(key);
    if (cached) {
      if (get().contourData !== cached) set({ contourData: cached, contourError: null });
      return;
    }
    if (contourAbort) contourAbort.abort();
    const controller = new AbortController();
    contourAbort = controller;
    set({ contourLoading: true, contourError: null });
    try {
      const data = await fetchContour2D(
        {
          datasetId,
          modelId,
          mappingId,
          xFeature: s.xFeature,
          yFeature: s.yFeature,
          pinningMode: s.pinningMode,
          currentFrameIndex: s.pinningMode === 'current_row' ? currentFrameIndex : null,
          useRobustRange: s.useRobustRange,
          gridResolution: 120,
          selectedRowIndices: queueRowIndices,
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      contourCache.set(key, data);
      set({ contourData: data });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      set({ contourError: err instanceof Error ? err.message : String(err) });
    } finally {
      if (contourAbort === controller) contourAbort = null;
      set({ contourLoading: false });
    }
  },

  clearError: () => set({ projError: null, contourError: null }),
}));

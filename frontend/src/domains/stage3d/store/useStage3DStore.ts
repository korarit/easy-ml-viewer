import { create } from 'zustand';
import {
  fetchInitializeScene,
  type AxisScaleMeta,
  type TrajectoryPoint3D,
} from '../api/stage3dApi';

export type { TrajectoryPoint3D };
export type CameraPresetMode = 'isometric' | 'top_xy' | 'front_xz' | 'side_yz' | 'follow_ball';

/** Phase 41 — dual-mode viewport: orthographic 2D planar vs 3D spatial. */
export type ViewMode = '2d' | '3d';

export const CAMERA_PRESET_KEYS: Record<string, CameraPresetMode> = {
  '1': 'isometric',
  '2': 'top_xy',
  '3': 'front_xz',
  '4': 'side_yz',
  '5': 'follow_ball',
};

interface Stage3DState {
  datasetId: string | null;
  modelId: string | null;
  mappingId: string | null;
  featureNames: string[];
  axisXFeatureIndex: number;
  axisYFeatureIndex: number;
  surfaceResolution: number;
  axes: { x: AxisScaleMeta | null; y: AxisScaleMeta | null; z: AxisScaleMeta | null };
  surfaceVisualType: string;
  surfaceVerticesFloat32: Float32Array | null;
  points: TrajectoryPoint3D[];
  datasetRowCount: number;
  cameraPreset: CameraPresetMode;
  showSurface: boolean;
  showDropLines: boolean;
  showResidualBars: boolean;
  showBackgroundCloud: boolean;
  showTickLabels: boolean;
  activeFrameIndex: number;
  viewMode: ViewMode;
  showBoundaryLines: boolean;
  showConfidenceIsolines: boolean;
  showHeatmap: boolean;
  showRulers: boolean;
  isLoading: boolean;
  stageError: string | null;
  contextLost: boolean;
  sceneKey: number;
  fps: number;
  drawCalls: number;
  setCameraPreset: (preset: CameraPresetMode) => void;
  setViewMode: (mode: ViewMode) => void;
  toggleViewMode: () => void;
  setShowBoundaryLines: (v: boolean) => void;
  setShowConfidenceIsolines: (v: boolean) => void;
  setShowHeatmap: (v: boolean) => void;
  setShowRulers: (v: boolean) => void;
  setActiveFrameIndex: (frameIdx: number) => void;
  setAxes: (xIdx: number, yIdx: number) => Promise<void>;
  setSurfaceResolution: (res: number) => Promise<void>;
  toggleLayer: (key: 'showSurface' | 'showDropLines' | 'showResidualBars' | 'showBackgroundCloud' | 'showTickLabels') => void;
  resetCamera: () => void;
  setContextLost: (lost: boolean) => void;
  restoreContext: () => void;
  reportPerf: (fps: number, drawCalls: number) => void;
  initializeScene: (params: {
    datasetId: string;
    modelId: string;
    mappingId: string;
    featureNames: string[];
    queueRowIndices: number[];
  }) => Promise<void>;
  refreshQueue: (queueRowIndices: number[]) => Promise<void>;
  applyProjectionFrame: (frame: {
    datasetId: string;
    modelId: string;
    mappingId: string;
    featureNames: string[];
    axisXFeatureIndex: number;
    axisYFeatureIndex: number;
    axes: { x: AxisScaleMeta; y: AxisScaleMeta; z: AxisScaleMeta };
    points: TrajectoryPoint3D[];
    surfaceResolution: number;
    surfaceVisualType: string;
  }) => void;
  clearError: () => void;
}

let inFlightAbort: AbortController | null = null;

function toFloat32(vertices: number[]): Float32Array {
  return new Float32Array(vertices);
}

export const useStage3DStore = create<Stage3DState>((set, get) => ({
  datasetId: null,
  modelId: null,
  mappingId: null,
  featureNames: [],
  axisXFeatureIndex: 0,
  axisYFeatureIndex: 1,
  surfaceResolution: 50,
  axes: { x: null, y: null, z: null },
  surfaceVisualType: 'flat_hyperplane',
  surfaceVerticesFloat32: null,
  points: [],
  datasetRowCount: 0,
  cameraPreset: 'isometric',
  showSurface: true,
  showDropLines: true,
  showResidualBars: true,
  showBackgroundCloud: true,
  showTickLabels: true,
  activeFrameIndex: 0,
  viewMode: '3d',
  showBoundaryLines: true,
  showConfidenceIsolines: true,
  showHeatmap: true,
  showRulers: true,
  isLoading: false,
  stageError: null,
  contextLost: false,
  sceneKey: 0,
  fps: 0,
  drawCalls: 0,

  setCameraPreset: (preset) => set({ cameraPreset: preset }),

  setViewMode: (mode) => set({ viewMode: mode }),

  toggleViewMode: () => set((s) => ({ viewMode: s.viewMode === '2d' ? '3d' : '2d' })),

  setShowBoundaryLines: (v) => set({ showBoundaryLines: v }),

  setShowConfidenceIsolines: (v) => set({ showConfidenceIsolines: v }),

  setShowHeatmap: (v) => set({ showHeatmap: v }),

  setShowRulers: (v) => set({ showRulers: v }),

  setActiveFrameIndex: (frameIdx) => {
    const { points } = get();
    if (points.length === 0) {
      set({ activeFrameIndex: 0 });
      return;
    }
    const clamped = Math.max(0, Math.min(frameIdx, points.length - 1));
    set({ activeFrameIndex: clamped });
  },

  setAxes: async (xIdx, yIdx) => {
    const s = get();
    if (!s.datasetId || !s.modelId || !s.mappingId) return;
    set({ axisXFeatureIndex: xIdx, axisYFeatureIndex: yIdx });
    await s.initializeScene({
      datasetId: s.datasetId,
      modelId: s.modelId,
      mappingId: s.mappingId,
      featureNames: s.featureNames,
      queueRowIndices: s.points.map((p) => p.original_row_index),
    });
  },

  setSurfaceResolution: async (res) => {
    const s = get();
    if (!s.datasetId || !s.modelId || !s.mappingId) return;
    const clamped = Math.max(10, Math.min(80, Math.round(res)));
    set({ surfaceResolution: clamped });
    await s.initializeScene({
      datasetId: s.datasetId,
      modelId: s.modelId,
      mappingId: s.mappingId,
      featureNames: s.featureNames,
      queueRowIndices: s.points.map((p) => p.original_row_index),
    });
  },

  toggleLayer: (key) => set((s) => ({ [key]: !s[key] }) as Partial<Stage3DState>),

  resetCamera: () => set({ cameraPreset: 'isometric' }),

  setContextLost: (lost) => set({ contextLost: lost }),

  restoreContext: () => set((s) => ({ contextLost: false, sceneKey: s.sceneKey + 1 })),

  reportPerf: (fps, drawCalls) => set({ fps, drawCalls }),

  initializeScene: async ({ datasetId, modelId, mappingId, featureNames, queueRowIndices }) => {
    const prev = get();
    const sameScene =
      prev.datasetId === datasetId &&
      prev.modelId === modelId &&
      prev.mappingId === mappingId &&
      prev.axisXFeatureIndex === get().axisXFeatureIndex;
    if (inFlightAbort) inFlightAbort.abort();
    const controller = new AbortController();
    inFlightAbort = controller;
    set({
      isLoading: true,
      stageError: null,
      datasetId,
      modelId,
      mappingId,
      featureNames,
    });
    try {
      const res = await fetchInitializeScene(
        {
          datasetId,
          modelId,
          mappingId,
          axisXFeatureIndex: get().axisXFeatureIndex,
          axisYFeatureIndex: get().axisYFeatureIndex,
          selectedRowIndices: queueRowIndices,
          gridResolution: get().surfaceResolution,
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      set({
        axes: res.axes,
        surfaceVisualType: res.surface.visual_type,
        surfaceVerticesFloat32: toFloat32(res.surface.world_vertices_xyz),
        surfaceResolution: res.surface.resolution,
        points: res.trajectory_points,
        datasetRowCount: res.dataset_row_count,
        activeFrameIndex: 0,
      });
      void sameScene;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      set({ stageError: err instanceof Error ? err.message : String(err) });
    } finally {
      if (inFlightAbort === controller) inFlightAbort = null;
      set({ isLoading: false });
    }
  },

  refreshQueue: async (queueRowIndices) => {
    const s = get();
    if (!s.datasetId || !s.modelId || !s.mappingId) return;
    await s.initializeScene({
      datasetId: s.datasetId,
      modelId: s.modelId,
      mappingId: s.mappingId,
      featureNames: s.featureNames,
      queueRowIndices,
    });
  },

  applyProjectionFrame: (frame) => {
    const prev = get();
    set({
      datasetId: frame.datasetId,
      modelId: frame.modelId,
      mappingId: frame.mappingId,
      featureNames: frame.featureNames,
      axisXFeatureIndex: frame.axisXFeatureIndex,
      axisYFeatureIndex: frame.axisYFeatureIndex,
      axes: frame.axes,
      points: frame.points,
      surfaceResolution: frame.surfaceResolution,
      surfaceVisualType: frame.surfaceVisualType,
      // Phase 09 surface arrives as z-heights; the projection mesh hook
      // owns its own buffers, so clear the legacy interleaved buffer to
      // avoid double-rendering two surfaces.
      surfaceVerticesFloat32: null,
      activeFrameIndex:
        frame.points.length === 0
          ? 0
          : Math.max(0, Math.min(prev.activeFrameIndex, frame.points.length - 1)),
    });
  },

  clearError: () => set({ stageError: null }),
}));

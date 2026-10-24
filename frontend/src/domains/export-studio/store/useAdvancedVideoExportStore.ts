import { create } from 'zustand';
import type { AdvancedVideoExportOptions } from '../types/advancedVideoExport.types';
import { DEFAULT_ADVANCED_VIDEO_OPTIONS } from '../types/advancedVideoExport.types';

interface AdvancedVideoExportState extends AdvancedVideoExportOptions {
  isRendering: boolean;
  renderProgress: number; // 0..100
  renderedFrames: number;
  totalFrames: number;
  renderError: string | null;
  lastFilename: string | null;

  setOptions: (patch: Partial<AdvancedVideoExportOptions>) => void;
  resetRange: (totalFrames: number) => void;
  setRendering: (rendering: boolean, progress?: number) => void;
  setRenderCount: (rendered: number, total: number) => void;
  setRenderError: (msg: string | null) => void;
  setLastFile: (filename: string | null) => void;
}

export const useAdvancedVideoExportStore = create<AdvancedVideoExportState>()((set) => ({
  ...DEFAULT_ADVANCED_VIDEO_OPTIONS,

  isRendering: false,
  renderProgress: 0,
  renderedFrames: 0,
  totalFrames: 0,
  renderError: null,
  lastFilename: null,

  setOptions: (patch) =>
    set((s) => {
      const next = { ...s, ...patch };
      // Keep the frame window ordered + non-negative.
      const start = Math.max(0, Math.floor(next.rangeStartFrame));
      let end = Math.max(0, Math.floor(next.rangeEndFrame));
      if (end < start) end = start;
      return { ...next, rangeStartFrame: start, rangeEndFrame: end };
    }),
  resetRange: (totalFrames) =>
    set({ rangeStartFrame: 0, rangeEndFrame: Math.max(0, totalFrames - 1) }),
  setRendering: (rendering, progress) =>
    set((s) => ({
      isRendering: rendering,
      renderProgress: progress ?? (rendering ? s.renderProgress : 0),
      ...(rendering ? { renderError: null } : {}),
    })),
  setRenderCount: (rendered, total) =>
    set({
      renderedFrames: rendered,
      totalFrames: total,
      renderProgress: total > 0 ? Math.round((rendered / total) * 100) : 0,
    }),
  setRenderError: (msg) => set({ renderError: msg }),
  setLastFile: (filename) => set({ lastFilename: filename }),
}));

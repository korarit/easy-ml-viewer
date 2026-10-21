import { create } from 'zustand';
import type {
  EnrichmentOptions,
  ExportFormat,
  ExportTab,
  RowScope,
  SnapshotBackground,
  SnapshotScale,
} from '../types/export.types';

interface ExportStudioState {
  isModalOpen: boolean;
  activeTab: ExportTab;
  format: ExportFormat;
  scope: RowScope;
  rangeStart: number;
  rangeEnd: number;
  enrichment: EnrichmentOptions;
  includeAttributionColumns: boolean;
  includeBookmarks: boolean;
  isExporting: boolean;
  exportProgress: number;
  exportError: string | null;
  lastExportFilename: string | null;
  // Video recording indicator state (driven by the recorder hook).
  isRecording: boolean;
  recordingElapsedSec: number;
  recordingMime: string;
  recordingError: string | null;
  // Snapshot state.
  snapshotScale: SnapshotScale;
  snapshotBackground: SnapshotBackground;
  snapshotError: string | null;
  isSnapshotting: boolean;

  setModal: (open: boolean) => void;
  setTab: (tab: ExportTab) => void;
  setFormat: (format: ExportFormat) => void;
  setScope: (scope: RowScope) => void;
  setRange: (start: number, end: number) => void;
  setEnrichment: (patch: Partial<EnrichmentOptions>) => void;
  setExporting: (exporting: boolean, progress?: number) => void;
  setExportError: (msg: string | null) => void;
  setLastExport: (filename: string | null) => void;
  setRecording: (recording: boolean, mime?: string) => void;
  setRecordingElapsed: (sec: number) => void;
  setRecordingError: (msg: string | null) => void;
  setSnapshotOptions: (scale: SnapshotScale, background: SnapshotBackground) => void;
  setSnapshotError: (msg: string | null) => void;
  setSnapshotting: (busy: boolean) => void;
}

export const useExportStore = create<ExportStudioState>()((set) => ({
  isModalOpen: false,
  activeTab: 'data',
  format: 'csv',
  scope: 'all_queued',
  rangeStart: 0,
  rangeEnd: -1,
  enrichment: {
    includePredictions: true,
    includeConfidence: true,
    includeAttributionColumns: true,
    includeBookmarks: true,
  },
  includeAttributionColumns: true,
  includeBookmarks: true,
  isExporting: false,
  exportProgress: 0,
  exportError: null,
  lastExportFilename: null,
  isRecording: false,
  recordingElapsedSec: 0,
  recordingMime: '',
  recordingError: null,
  snapshotScale: 1,
  snapshotBackground: 'studio',
  snapshotError: null,
  isSnapshotting: false,

  setModal: (open) => set({ isModalOpen: open, ...(open ? { exportError: null } : {}) }),
  setTab: (tab) => set({ activeTab: tab }),
  setFormat: (format) => set({ format }),
  setScope: (scope) => set({ scope }),
  setRange: (start, end) => set({ rangeStart: Math.max(0, start), rangeEnd: end }),
  setEnrichment: (patch) =>
    set((s) => {
      const next = { ...s.enrichment, ...patch };
      return {
        enrichment: next,
        includeAttributionColumns: next.includeAttributionColumns,
        includeBookmarks: next.includeBookmarks,
      };
    }),
  setExporting: (exporting, progress) =>
    set((s) => ({
      isExporting: exporting,
      exportProgress: progress ?? (exporting ? s.exportProgress : 0),
    })),
  setExportError: (msg) => set({ exportError: msg }),
  setLastExport: (filename) => set({ lastExportFilename: filename }),
  setRecording: (recording, mime) =>
    set({
      isRecording: recording,
      ...(mime !== undefined ? { recordingMime: mime } : {}),
      ...(recording ? { recordingElapsedSec: 0, recordingError: null } : {}),
    }),
  setRecordingElapsed: (sec) => set({ recordingElapsedSec: sec }),
  setRecordingError: (msg) => set({ recordingError: msg }),
  setSnapshotOptions: (scale, background) =>
    set({ snapshotScale: scale, snapshotBackground: background }),
  setSnapshotError: (msg) => set({ snapshotError: msg }),
  setSnapshotting: (busy) => set({ isSnapshotting: busy }),
}));

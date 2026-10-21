/** Phase 16 — export-studio shared types (enriched table + WebGL capture). */

export type ExportFormat = 'csv' | 'xlsx' | 'parquet';
export type RowScope = 'all_queued' | 'anomaly_bookmarked' | 'sub_range';
export type ExportTab = 'data' | 'video' | 'snapshot';
export type SnapshotScale = 1 | 2;
export type SnapshotBackground = 'studio' | 'transparent';

export interface EnrichmentOptions {
  includePredictions: boolean;
  includeConfidence: boolean;
  includeAttributionColumns: boolean;
  includeBookmarks: boolean;
}

export interface ExportTableParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  sessionId: string | null;
  selectedRowIndices: number[];
  format: ExportFormat;
  scope: RowScope;
  rangeStart: number;
  rangeEnd: number;
  includeAttributionColumns: boolean;
  includeBookmarks: boolean;
}

export interface ExportPreviewResponse {
  header: string[];
  rows: (string | number | boolean | null)[][];
  total_frames: number;
  is_classification: boolean;
  format: ExportFormat;
  scope: RowScope;
}

export interface SnapshotOptions {
  scale: SnapshotScale;
  background: SnapshotBackground;
}

export interface VideoRecorderOptions {
  fps: 30 | 60;
  bitrateMbps: number;
  hudOverlay: boolean;
  autoStopFrame: number | null;
}

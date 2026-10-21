/** Phase 15 — `.mlv` recording shared types. */

export interface MlvBookmark {
  bookmark_id: string;
  frame_index: number;
  dataset_row_index: number;
  color: string;
  note_text: string;
  created_at_iso: string;
}

export interface MlvAutoEvent {
  frame_index: number;
  kind: string;
  severity: number;
  label: string;
}

export interface MlvHeader {
  schema_version: string;
  title: string;
  notes: string;
  created_at_iso: string;
  model_type: string;
  model_family: string;
  is_classification: boolean;
  feature_names: string[];
  total_frames: number;
  class_aliases: Record<string, unknown>;
  feature_aliases: Record<string, unknown>;
  bookmarks: MlvBookmark[];
  auto_events: MlvAutoEvent[];
}

export interface PrebakedMlvSession {
  session_id: string;
  header: MlvHeader;
  total_frames: number;
  is_prebaked_replay: boolean;
}

export type BakeLod = 'standard' | 'high';

export interface BakeOptions {
  title: string;
  notes: string;
  surfaceResolution: number;
  includeExplainability: boolean;
  includeBookmarks: boolean;
}

export function lodToResolution(lod: BakeLod): number {
  return lod === 'high' ? 100 : 60;
}

/** "~1.8 MB (2,500 frames)" estimate chip (spec §2.1). */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

export function formatEstimate(bytes: number, frames: number): string {
  return `Est. Compressed Size: ~${formatBytes(bytes)} (${frames.toLocaleString()} frames)`;
}

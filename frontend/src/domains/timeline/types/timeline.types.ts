/** Phase 11 — smart multi-track timeline shared types. */

export type TimelineEventType =
  | 'misclassified'
  | 'decision_flip'
  | 'low_confidence'
  | 'input_outlier';

export type BookmarkColor = 'amber' | 'cyan' | 'emerald' | 'rose' | 'purple';

export interface AutoEventPin {
  frame_index: number;
  dataset_row_index: number;
  event_type: TimelineEventType;
  severity: number;
  title: string;
  detail: string;
}

export interface UserBookmark {
  bookmark_id: string;
  /** -1 = orphaned: row no longer in the current queue (edge case #6). */
  frame_index: number;
  dataset_row_index: number;
  color: BookmarkColor;
  note_text: string;
  created_at_iso: string;
}

export interface MultiTrackTimelineData {
  session_id: string;
  totalFrames: number;
  isClassification: boolean;
  isLabeled: boolean;
  bucketSize: number;
  predictionCurve: Float32Array;
  actualCurve: Float32Array | null;
  confidenceScores: Float32Array;
  autoEvents: AutoEventPin[];
  bookmarks: UserBookmark[];
}

export interface TimelineTracksPayload {
  session_id: string;
  total_frames: number;
  is_classification: boolean;
  is_labeled: boolean;
  bucket_size: number;
  prediction_curve: number[];
  actual_curve: number[] | null;
  confidence_scores: number[];
  auto_events: AutoEventPin[];
  bookmarks: UserBookmark[];
}

export interface PinClusterNode {
  startFrame: number;
  endFrame: number;
  centerXPx: number;
  pins: AutoEventPin[];
}

/** Resolved bookmark rendered on the pin lane. */
export interface BookmarkPin extends AutoEventPin {
  bookmarkId: string;
  color: BookmarkColor;
}

export type JumpFilter =
  | 'all'
  | 'errors'
  | 'flips'
  | 'lowconf'
  | 'outliers'
  | 'bookmarks';

export const JUMP_FILTER_LABEL: Record<JumpFilter, string> = {
  all: 'All Events',
  errors: 'Errors Only',
  flips: 'Decision Flips Only',
  lowconf: 'Low-Confidence Only',
  outliers: 'Outliers Only',
  bookmarks: 'My Bookmarks Only',
};

/** Pin glyph colors (canvas + legend). */
export const EVENT_TYPE_COLOR: Record<TimelineEventType, string> = {
  misclassified: '#f43e5e',
  decision_flip: '#f59e0b',
  low_confidence: '#facc15',
  input_outlier: '#c084fc',
};

export const BOOKMARK_COLOR_HEX: Record<BookmarkColor, string> = {
  amber: '#f59e0b',
  cyan: '#22d3ee',
  emerald: '#34d399',
  rose: '#fb7185',
  purple: '#c084fc',
};

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  fetchBuildTimelineTracks,
  fetchCreateTimelineBookmark,
  fetchDeleteTimelineBookmark,
  type BuildTracksParams,
} from '../api/timelineApi';
import type {
  AutoEventPin,
  BookmarkColor,
  BookmarkPin,
  JumpFilter,
  MultiTrackTimelineData,
  TimelineEventType,
  TimelineTracksPayload,
  UserBookmark,
} from '../types/timeline.types';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';

export type { BookmarkPin, JumpFilter };

const MAX_ZOOM_SPAN = 1e9;

/** Edge case #6: bookmarks follow `dataset_row_index`, not queue position. */
export function resolveBookmarkFrame(
  bookmark: UserBookmark,
  rowIndices: number[],
): number | null {
  const frame = rowIndices.indexOf(bookmark.dataset_row_index);
  if (frame !== -1) return frame;
  // Fall back to the server-rebased frame when the row is still in range.
  if (bookmark.frame_index >= 0 && bookmark.frame_index < rowIndices.length) {
    return bookmark.frame_index;
  }
  return null;
}

export function bookmarkToPin(
  bookmark: UserBookmark,
  frameIndex: number,
): BookmarkPin {
  return {
    frame_index: frameIndex,
    dataset_row_index: bookmark.dataset_row_index,
    event_type: 'misclassified',
    severity: 1.0,
    title: bookmark.note_text || `Bookmark @ frame ${frameIndex}`,
    detail: bookmark.note_text,
    bookmarkId: bookmark.bookmark_id,
    color: bookmark.color,
  } as BookmarkPin;
}

function payloadToData(payload: TimelineTracksPayload): MultiTrackTimelineData {
  return {
    session_id: payload.session_id,
    totalFrames: payload.total_frames,
    isClassification: payload.is_classification,
    isLabeled: payload.is_labeled,
    bucketSize: Math.max(1, payload.bucket_size),
    predictionCurve: new Float32Array(payload.prediction_curve),
    actualCurve: payload.actual_curve ? new Float32Array(payload.actual_curve) : null,
    confidenceScores: new Float32Array(payload.confidence_scores),
    autoEvents: payload.auto_events,
    bookmarks: payload.bookmarks,
  };
}

/** Frames matching the jump filter (wrap-around handled by callers). */
export function collectJumpFrames(
  data: MultiTrackTimelineData | null,
  jumpFilter: JumpFilter,
  rowIndices: number[],
  visible: Record<TimelineEventType, boolean>,
  showBookmarks: boolean,
): number[] {
  if (!data) return [];
  const frames = new Set<number>();
  if (jumpFilter === 'all' || jumpFilter === 'errors') {
    if (jumpFilter === 'errors' || visible.misclassified) {
      for (const p of data.autoEvents) if (p.event_type === 'misclassified') frames.add(p.frame_index);
    }
  }
  if (jumpFilter === 'all' || jumpFilter === 'flips') {
    if (jumpFilter === 'flips' || visible.decision_flip) {
      for (const p of data.autoEvents) if (p.event_type === 'decision_flip') frames.add(p.frame_index);
    }
  }
  if (jumpFilter === 'all' || jumpFilter === 'lowconf') {
    if (jumpFilter === 'lowconf' || visible.low_confidence) {
      for (const p of data.autoEvents) if (p.event_type === 'low_confidence') frames.add(p.frame_index);
    }
  }
  if (jumpFilter === 'all' || jumpFilter === 'outliers') {
    if (jumpFilter === 'outliers' || visible.input_outlier) {
      for (const p of data.autoEvents) if (p.event_type === 'input_outlier') frames.add(p.frame_index);
    }
  }
  if ((jumpFilter === 'all' && showBookmarks) || jumpFilter === 'bookmarks') {
    for (const b of data.bookmarks) {
      const f = resolveBookmarkFrame(b, rowIndices);
      if (f !== null) frames.add(f);
    }
  }
  return [...frames].sort((a, b) => a - b);
}

export function nextJumpFrame(current: number, frames: number[]): number | null {
  if (frames.length === 0) return null;
  const nxt = frames.find((f) => f > current);
  return nxt ?? frames[0] ?? null; // Edge #5: wrap-around to the first pin.
}

export function prevJumpFrame(current: number, frames: number[]): number | null {
  if (frames.length === 0) return null;
  const rev = [...frames].reverse().find((f) => f < current);
  return rev ?? frames[frames.length - 1] ?? null;
}

interface TimelineState {
  sessionId: string | null;
  data: MultiTrackTimelineData | null;
  buildKey: string | null;
  rowIndices: number[];
  isLoading: boolean;
  timelineError: string | null;
  viewStart: number;
  viewEnd: number;
  visible: Record<TimelineEventType, boolean>;
  showBookmarks: boolean;
  jumpFilter: JumpFilter;
  selectedBookmarkId: string | null;
  editorOpen: boolean;
  editorFrame: number | null;
  editingBookmarkId: string | null;
  wrapNoticeAt: number;
  loadTracks: (params: BuildTracksParams & { buildKey: string }) => Promise<void>;
  setView: (start: number, end: number) => void;
  zoomBy: (factor: number, centerFrame?: number) => void;
  resetZoom: () => void;
  toggleEventType: (t: TimelineEventType) => void;
  toggleBookmarks: () => void;
  setJumpFilter: (f: JumpFilter) => void;
  jumpToNext: (rowIndices: number[]) => void;
  jumpToPrev: (rowIndices: number[]) => void;
  clearWrapNotice: () => void;
  openEditor: (frame: number, bookmark?: UserBookmark) => void;
  closeEditor: () => void;
  saveBookmark: (noteText: string, color: BookmarkColor) => Promise<void>;
  deleteBookmark: (bookmarkId: string) => Promise<void>;
  setSelectedBookmark: (id: string | null) => void;
  clearError: () => void;
}

let inFlightAbort: AbortController | null = null;
let requestSeq = 0;

function clampView(start: number, end: number, total: number): [number, number] {
  if (total <= 0) return [0, 0];
  const maxF = total - 1;
  let s = Math.max(0, Math.min(maxF, Math.floor(start)));
  let e = Math.max(0, Math.min(maxF, Math.ceil(end)));
  if (e - s < 1) {
    if (s >= maxF) s = Math.max(0, maxF - 1);
    e = Math.min(maxF, s + 1);
  }
  if (s > e) [s, e] = [e, s];
  return [s, e];
}

export const useTimelineStore = create<TimelineState>()(
  subscribeWithSelector((set, get) => ({
    sessionId: null,
    data: null,
    buildKey: null,
    rowIndices: [],
    isLoading: false,
    timelineError: null,
    viewStart: 0,
    viewEnd: 0,
    visible: {
      misclassified: true,
      decision_flip: true,
      low_confidence: true,
      input_outlier: true,
    },
    showBookmarks: true,
    jumpFilter: 'all',
    selectedBookmarkId: null,
    editorOpen: false,
    editorFrame: null,
    editingBookmarkId: null,
    wrapNoticeAt: 0,

    loadTracks: async (params) => {
      const s = get();
      if (s.buildKey === params.buildKey && s.data) return;
      if (inFlightAbort) inFlightAbort.abort();
      const controller = new AbortController();
      inFlightAbort = controller;
      const seq = (requestSeq += 1);
      const prevCtx = s.buildKey?.split('|queue:')[0];
      const nextCtx = params.buildKey.split('|queue:')[0];
      const isNewContext = prevCtx !== undefined && prevCtx !== nextCtx;
      set({
        isLoading: true,
        timelineError: null,
        // New dataset/model/mapping context → drop the old session + view.
        ...(isNewContext
          ? { sessionId: null, data: null, viewStart: 0, viewEnd: 0, selectedBookmarkId: null }
          : {}),
      });
      try {
        const payload = await fetchBuildTimelineTracks(
          {
            datasetId: params.datasetId,
            modelId: params.modelId,
            mappingId: params.mappingId,
            selectedRowIndices: params.selectedRowIndices,
            ...(get().sessionId && !isNewContext ? { sessionId: get().sessionId as string } : {}),
          },
          controller.signal,
        );
        if (controller.signal.aborted || seq !== requestSeq) return;
        const data = payloadToData(payload);
        const total = data.totalFrames;
        const prev = get();
        const keepView =
          !isNewContext &&
          prev.viewEnd > prev.viewStart &&
          prev.viewEnd < MAX_ZOOM_SPAN &&
          prev.data?.totalFrames === total;
        const [vs, ve] = keepView
          ? clampView(prev.viewStart, prev.viewEnd, total)
          : [0, Math.max(0, total - 1)];
        set({
          sessionId: data.session_id,
          data,
          buildKey: params.buildKey,
          rowIndices: params.selectedRowIndices,
          viewStart: vs,
          viewEnd: ve,
        });
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (controller.signal.aborted || seq !== requestSeq) return;
        set({ timelineError: err instanceof Error ? err.message : String(err) });
      } finally {
        if (inFlightAbort === controller) inFlightAbort = null;
        if (seq === requestSeq) set({ isLoading: false });
      }
    },

    setView: (start, end) => {
      const { data } = get();
      const [s, e] = clampView(start, end, data?.totalFrames ?? 0);
      set({ viewStart: s, viewEnd: e });
    },

    zoomBy: (factor, centerFrame) => {
      const { data, viewStart, viewEnd } = get();
      const total = data?.totalFrames ?? 0;
      if (total <= 1 || !(factor > 0)) return;
      const center =
        centerFrame ?? Math.round(usePlaybackStore.getState().currentFrameIndex);
      const c = Math.max(0, Math.min(total - 1, center));
      const half = Math.max(1, Math.round(((viewEnd - viewStart) / 2) * factor));
      set({
        viewStart: Math.max(0, c - half),
        viewEnd: Math.min(total - 1, c + half),
      });
    },

    resetZoom: () => {
      const total = get().data?.totalFrames ?? 0;
      set({ viewStart: 0, viewEnd: Math.max(0, total - 1) });
    },

    toggleEventType: (t) => set((s) => ({ visible: { ...s.visible, [t]: !s.visible[t] } })),

    toggleBookmarks: () => set((s) => ({ showBookmarks: !s.showBookmarks })),

    setJumpFilter: (f) => set({ jumpFilter: f }),

    jumpToNext: (rowIndices) => {
      const { data, jumpFilter, visible, showBookmarks } = get();
      const frames = collectJumpFrames(data, jumpFilter, rowIndices, visible, showBookmarks);
      const current = usePlaybackStore.getState().currentFrameIndex;
      const target = nextJumpFrame(current, frames);
      if (target === null) return;
      usePlaybackStore.getState().seekToFrame(target, true);
      if (target <= current) set({ wrapNoticeAt: Date.now() });
    },

    jumpToPrev: (rowIndices) => {
      const { data, jumpFilter, visible, showBookmarks } = get();
      const frames = collectJumpFrames(data, jumpFilter, rowIndices, visible, showBookmarks);
      const current = usePlaybackStore.getState().currentFrameIndex;
      const target = prevJumpFrame(current, frames);
      if (target === null) return;
      usePlaybackStore.getState().seekToFrame(target, true);
      if (target >= current) set({ wrapNoticeAt: Date.now() });
    },

    clearWrapNotice: () => set({ wrapNoticeAt: 0 }),

    openEditor: (frame, bookmark) =>
      set({
        editorOpen: true,
        editorFrame: frame,
        editingBookmarkId: bookmark?.bookmark_id ?? null,
        selectedBookmarkId: bookmark?.bookmark_id ?? null,
      }),

    closeEditor: () =>
      set({ editorOpen: false, editorFrame: null, editingBookmarkId: null }),

    saveBookmark: async (noteText, color) => {
      const { sessionId, editorFrame, editingBookmarkId, data, rowIndices } = get();
      if (!sessionId || editorFrame === null || editorFrame < 0) return;
      const rowIndex = rowIndices[editorFrame] ?? data?.bookmarks.find((b) => b.bookmark_id === editingBookmarkId)?.dataset_row_index ?? 0;
      set({ isLoading: true, timelineError: null });
      try {
        if (editingBookmarkId) {
          try {
            await fetchDeleteTimelineBookmark(sessionId, editingBookmarkId);
          } catch {
            /* replaced anyway */
          }
          set((s) => ({
            data: s.data
              ? { ...s.data, bookmarks: s.data.bookmarks.filter((b) => b.bookmark_id !== editingBookmarkId) }
              : s.data,
          }));
        }
        const created = await fetchCreateTimelineBookmark({
          sessionId,
          frameIndex: editorFrame,
          datasetRowIndex: rowIndex,
          color,
          noteText: noteText.slice(0, 280),
        });
        set((s) => ({
          data: s.data ? { ...s.data, bookmarks: [...s.data.bookmarks, created] } : s.data,
          selectedBookmarkId: created.bookmark_id,
          editorOpen: false,
          editorFrame: null,
          editingBookmarkId: null,
        }));
      } catch (err) {
        set({ timelineError: err instanceof Error ? err.message : String(err) });
      } finally {
        set({ isLoading: false });
      }
    },

    deleteBookmark: async (bookmarkId) => {
      const { sessionId } = get();
      if (!sessionId) return;
      set({ isLoading: true, timelineError: null });
      try {
        await fetchDeleteTimelineBookmark(sessionId, bookmarkId);
        set((s) => ({
          data: s.data
            ? { ...s.data, bookmarks: s.data.bookmarks.filter((b) => b.bookmark_id !== bookmarkId) }
            : s.data,
          selectedBookmarkId: s.selectedBookmarkId === bookmarkId ? null : s.selectedBookmarkId,
          editorOpen: false,
          editorFrame: null,
          editingBookmarkId: null,
        }));
      } catch (err) {
        set({ timelineError: err instanceof Error ? err.message : String(err) });
      } finally {
        set({ isLoading: false });
      }
    },

    setSelectedBookmark: (id) => set({ selectedBookmarkId: id }),

    clearError: () => set({ timelineError: null }),
  })),
);

import { useEffect, useMemo } from 'react';
import { usePlaybackStore } from '../store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { useDatasetStore } from '../../dataset/store/useDatasetStore';
import { useModelStore } from '../../model_engine/store/useModelStore';
import { useMappingStore } from '../../mapping/store/useMappingStore';
import { useTimelineStore } from '../../timeline/store/useTimelineStore';
import { useTimelineKeyboardShortcuts } from '../../timeline/hooks/useTimelineKeyboardShortcuts';
import { MultiTrackTimelineCanvas } from '../../timeline/components/MultiTrackTimelineCanvas';
import { TimelineFilterBar, EventJumpNavigator } from '../../timeline/components/EventJumpNavigator';
import { BookmarkEditorPopover } from '../../timeline/components/BookmarkEditorPopover';
import { BookmarkListSidebar } from '../../timeline/components/BookmarkListSidebar';
import { usePlaybackRafLoop } from '../hooks/usePlaybackRafLoop';
import { usePlaybackKeyboardShortcuts } from '../hooks/usePlaybackKeyboardShortcuts';
import { VcrTransportDeck } from './VcrTransportDeck';
import { TimelineScrubberBar } from './TimelineScrubberBar';
import { PlaybackOptionsBar } from './PlaybackOptionsBar';

const DISCONTINUITY_THRESHOLD = 4.5;

function computeDiscontinuityFrames(
  points: { world_x: number; world_y: number; world_z_pred: number }[],
): number[] {
  const jumps: number[] = [];
  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1];
    const b = points[i];
    const dx = b.world_x - a.world_x;
    const dy = b.world_y - a.world_y;
    const dz = b.world_z_pred - a.world_z_pred;
    if (![dx, dy, dz].every(Number.isFinite)) continue;
    if (Math.sqrt(dx * dx + dy * dy + dz * dz) > DISCONTINUITY_THRESHOLD) {
      jumps.push(i);
    }
  }
  return jumps;
}

/**
 * Phase 10 — VCR playback studio panel (bottom dock).
 *
 * Owns the queue <-> playback sync: the 3D stage trajectory becomes the
 * playback queue, and the integer frame drives the legacy stage highlight
 * + inspector while `exactFrameFloat` drives WebGL directly at 60FPS.
 */
export function PlaybackStudioPanel() {
  const points = useStage3DStore((s) => s.points);
  const currentFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const totalFrames = usePlaybackStore((s) => s.totalFrames);
  const status = usePlaybackStore((s) => s.status);
  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId } = useMappingStore();
  const timelineData = useTimelineStore((s) => s.data);
  const timelineError = useTimelineStore((s) => s.timelineError);
  const timelineLoading = useTimelineStore((s) => s.isLoading);

  usePlaybackRafLoop();
  usePlaybackKeyboardShortcuts(true);
  useTimelineKeyboardShortcuts(true);

  // Queue sync: trajectory length drives the playback buffer.
  const pointCount = points.length;
  useEffect(() => {
    const st = usePlaybackStore.getState();
    if (st.totalFrames !== pointCount) {
      st.initializeQueue(pointCount);
    }
  }, [pointCount]);

  // Integer frame sync: inspector + legacy highlight follow the VCR.
  useEffect(() => {
    const clamped = Math.max(0, Math.min(currentFrameIndex, Math.max(pointCount - 1, 0)));
    useStage3DStore.getState().setActiveFrameIndex(clamped);
  }, [currentFrameIndex, pointCount]);

  const rowIndices = useMemo(() => points.map((p) => p.original_row_index), [points]);
  const predictions = useMemo(() => points.map((p) => p.raw_z_pred), [points]);
  const discontinuityFrames = useMemo(() => computeDiscontinuityFrames(points), [points]);

  // Phase 11: smart multi-track analysis follows the same queue (debounced
  // so queue rebuilds don't spam the backend; session persists bookmarks).
  const datasetId = currentDataset?.dataset_id ?? null;
  const modelId = currentModel?.model_id ?? null;
  const queueKey = rowIndices.join(',');
  useEffect(() => {
    if (!datasetId || !modelId || !mappingId || rowIndices.length === 0) return;
    const timer = window.setTimeout(() => {
      void useTimelineStore.getState().loadTracks({
        datasetId,
        modelId,
        mappingId,
        selectedRowIndices: rowIndices,
        buildKey: `${datasetId}|${modelId}|${mappingId}|queue:${queueKey}`,
      });
    }, 300);
    return () => window.clearTimeout(timer);
  }, [datasetId, modelId, mappingId, queueKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeRowIndex =
    points.length > 0
      ? (points[Math.max(0, Math.min(currentFrameIndex, points.length - 1))]?.original_row_index ?? null)
      : null;

  const fps = usePlaybackStore((s) => s.baseFps * s.speedMultiplier);

  return (
    <div
      className="flex flex-col gap-2 rounded-lg px-3 py-2.5"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      aria-label="VCR playback studio"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span
          className="font-mono-tabular text-[10px] tracking-widest uppercase"
          style={{ color: 'var(--text-muted)' }}
        >
          VCR Playback Studio · {status.toUpperCase()} · {fps.toFixed(1)} FPS
        </span>
        {totalFrames > 1 && status === 'paused' && (
          <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            Space = play · ←/→ = ±1 (Shift = ±10) · Home/End · [ ] = loop A/B · L = loop · B = bookmark · N/P = events
          </span>
        )}
      </div>
      {/* Phase 11 — smart multi-track timeline (prediction / confidence / pins). */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <TimelineFilterBar rowIndices={rowIndices} />
        <EventJumpNavigator rowIndices={rowIndices} />
      </div>
      {timelineLoading && !timelineData ? (
        <div className="rounded-md px-3 py-4 text-center text-xs" style={{ color: 'var(--text-secondary)' }} role="status">
          Analyzing timeline tracks…
        </div>
      ) : (
        <MultiTrackTimelineCanvas data={timelineData} rowIndices={rowIndices} />
      )}
      {timelineError && (
        <div className="rounded px-2 py-1 text-[11px]" style={{ background: 'rgba(244,63,94,0.10)', color: '#f43e5e' }} role="alert">
          Timeline failed: {timelineError}{' '}
          <button type="button" onClick={() => useTimelineStore.getState().clearError()} className="underline">
            dismiss
          </button>
        </div>
      )}
      <div className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        <span>Zoom: Ctrl + wheel · Click pin = seek · Click cluster = zoom · Minimap = pan</span>
        <button
          type="button"
          onClick={() => useTimelineStore.getState().resetZoom()}
          className="rounded px-1.5 py-0.5"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          Reset zoom (Ctrl+0)
        </button>
        <button
          type="button"
          onClick={() => {
            const tl = useTimelineStore.getState();
            tl.openEditor(usePlaybackStore.getState().currentFrameIndex);
          }}
          disabled={totalFrames <= 1}
          className="rounded px-1.5 py-0.5 disabled:opacity-40"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          🔖 Add Note (B)
        </button>
      </div>
      <BookmarkListSidebar rowIndices={rowIndices} />
      <BookmarkEditorPopover />
      <TimelineScrubberBar
        rowIndices={rowIndices}
        predictions={predictions}
        discontinuityFrames={discontinuityFrames}
      />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <VcrTransportDeck rowIndex={activeRowIndex} />
        <PlaybackOptionsBar />
      </div>
    </div>
  );
}

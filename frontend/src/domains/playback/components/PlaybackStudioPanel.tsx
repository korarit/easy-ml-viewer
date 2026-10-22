import { useEffect, useMemo } from 'react';
import { Ban, Repeat, Repeat1 } from 'lucide-react';
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
import { useExplainabilityStore } from '../../explainability/store/useExplainabilityStore';
import { useExplainabilityKeyboardShortcuts } from '../../explainability/hooks/useExplainabilityKeyboardShortcuts';
import { VcrTransportDeck } from './VcrTransportDeck';
import { TimelineScrubberBar } from './TimelineScrubberBar';
import { SPEED_PRESETS, type LoopMode } from '../types/playback.types';

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

const LOOP_LABEL: Record<LoopMode, string> = {
  none: 'No loop',
  all: 'Loop all',
  ab_range: 'A–B loop',
};

function LoopIcon({ mode }: { mode: LoopMode }) {
  if (mode === 'none') return <Ban size={14} />;
  if (mode === 'ab_range') return <Repeat1 size={14} />;
  return <Repeat size={14} />;
}

/**
 * Headless sync: queue <-> playback buffer, VCR frame <-> 3D highlight,
 * debounced smart-timeline rebuild, rAF loop + shortcuts. No UI —
 * mount once in the 3D studio so the strip + dock tabs stay in sync.
 */
export function PlaybackSyncHost() {
  const points = useStage3DStore((s) => s.points);
  const currentFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const { currentDataset } = useDatasetStore();
  const { currentModel } = useModelStore();
  const { mappingId } = useMappingStore();

  usePlaybackRafLoop();
  usePlaybackKeyboardShortcuts(true);
  useTimelineKeyboardShortcuts(true);

  const pointCount = points.length;
  useEffect(() => {
    const st = usePlaybackStore.getState();
    if (st.totalFrames !== pointCount) {
      st.initializeQueue(pointCount);
    }
  }, [pointCount]);

  useEffect(() => {
    const clamped = Math.max(0, Math.min(currentFrameIndex, Math.max(pointCount - 1, 0)));
    useStage3DStore.getState().setActiveFrameIndex(clamped);
  }, [currentFrameIndex, pointCount]);

  const rowIndices = useMemo(() => points.map((p) => p.original_row_index), [points]);
  const datasetId = currentDataset?.dataset_id ?? null;
  const modelId = currentModel?.model_id ?? null;
  const queueKey = rowIndices.join(',');
  const explainTopK = useExplainabilityStore((s) => s.topK);

  useExplainabilityKeyboardShortcuts(true);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, modelId, mappingId, queueKey]);

  // Explainability session follows the same queue (debounced like timeline).
  useEffect(() => {
    if (!datasetId || !modelId || !mappingId || rowIndices.length === 0) return;
    const timer = window.setTimeout(() => {
      void useExplainabilityStore.getState().loadSession({
        datasetId,
        modelId,
        mappingId,
        selectedRowIndices: rowIndices,
        buildKey: `${datasetId}|${modelId}|${mappingId}|queue:${queueKey}`,
      });
    }, 300);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, modelId, mappingId, queueKey]);

  // Per-frame "why": fetch the current frame's explanation (debounced so
  // 60 FPS playback aborts in-flight and settles on the latest frame).
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void useExplainabilityStore.getState().fetchFrame(currentFrameIndex);
    }, 200);
    return () => window.clearTimeout(timer);
  }, [currentFrameIndex, explainTopK]);

  return null;
}

/**
 * Slim bottom transport strip: scrubber on top, transport + speed + loop
 * on one row. Always visible, fixed height — never overlaps the canvas.
 */
export function PlaybackTransportStrip() {
  const points = useStage3DStore((s) => s.points);
  const currentFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const totalFrames = usePlaybackStore((s) => s.totalFrames);
  const status = usePlaybackStore((s) => s.status);
  const speedMultiplier = usePlaybackStore((s) => s.speedMultiplier);
  const loopMode = usePlaybackStore((s) => s.loopMode);
  const loopRange = usePlaybackStore((s) => s.loopRange);

  const rowIndices = useMemo(() => points.map((p) => p.original_row_index), [points]);
  const predictions = useMemo(() => points.map((p) => p.raw_z_pred), [points]);
  const discontinuityFrames = useMemo(() => computeDiscontinuityFrames(points), [points]);
  const locked = totalFrames <= 1;

  const activeRowIndex =
    points.length > 0
      ? (points[Math.max(0, Math.min(currentFrameIndex, points.length - 1))]?.original_row_index ?? null)
      : null;

  return (
    <div
      className="flex shrink-0 flex-col gap-1.5 rounded-lg px-3 py-2"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      aria-label="Playback transport"
    >
      <TimelineScrubberBar
        rowIndices={rowIndices}
        predictions={predictions}
        discontinuityFrames={discontinuityFrames}
      />
      <div className="flex items-center gap-2 overflow-x-auto whitespace-nowrap">
        <VcrTransportDeck rowIndex={activeRowIndex} />
        <span className="hidden h-6 w-px shrink-0 sm:block" style={{ background: 'var(--border-subtle)' }} aria-hidden />
        <label className="flex shrink-0 items-center gap-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Speed
          <select
            aria-label="Playback speed"
            disabled={locked}
            value={String(speedMultiplier)}
            onChange={(e) => usePlaybackStore.getState().setSpeedMultiplier(Number(e.target.value))}
            className="font-mono-tabular min-h-[36px] rounded px-1.5 py-1 disabled:opacity-40"
            style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)' }}
          >
            {SPEED_PRESETS.map((p) => (
              <option key={p.multiplier} value={String(p.multiplier)}>
                {p.label} ({p.fpsLabel})
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          aria-label={`Loop mode: ${LOOP_LABEL[loopMode]}. Activate to cycle.`}
          title={`Loop: ${LOOP_LABEL[loopMode]} (L to cycle)${loopRange ? ` [A=${loopRange[0]}, B=${loopRange[1]}]` : ''}`}
          disabled={locked}
          onClick={() => usePlaybackStore.getState().cycleLoopMode()}
          className="flex min-h-[36px] shrink-0 items-center gap-1.5 rounded-md px-2 py-1 text-xs disabled:opacity-40"
          style={{
            border: '1px solid var(--border-strong)',
            color: loopMode === 'none' ? 'var(--text-muted)' : '#f59e0b',
            background: loopMode === 'none' ? 'transparent' : 'rgba(245,158,11,0.12)',
          }}
        >
          <LoopIcon mode={loopMode} />
          {loopMode === 'ab_range' && loopRange ? `A–B (${loopRange[0]}–${loopRange[1]})` : LOOP_LABEL[loopMode]}
        </button>
        <span className="font-mono-tabular shrink-0 text-[11px] uppercase" style={{ color: 'var(--text-muted)' }}>
          {status}
        </span>
      </div>
    </div>
  );
}

/**
 * Timeline inspector for the right dock tab: filters + multi-track canvas
 * + zoom helpers + bookmarks. Scrolls inside the dock, never the page.
 */
export function TimelineInspectorPanel() {
  const points = useStage3DStore((s) => s.points);
  const totalFrames = usePlaybackStore((s) => s.totalFrames);
  const timelineData = useTimelineStore((s) => s.data);
  const timelineError = useTimelineStore((s) => s.timelineError);
  const timelineLoading = useTimelineStore((s) => s.isLoading);

  const rowIndices = useMemo(() => points.map((p) => p.original_row_index), [points]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto" aria-label="Timeline inspector">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <TimelineFilterBar rowIndices={rowIndices} />
        <EventJumpNavigator rowIndices={rowIndices} />
      </div>
      {timelineLoading && !timelineData ? (
        <div className="rounded-md px-3 py-4 text-center text-xs" style={{ color: 'var(--text-secondary)' }} role="status">
          Drawing timeline…
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
      <div className="flex flex-wrap items-center gap-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        <span>Scroll to zoom · click a pin to jump</span>
        <button
          type="button"
          onClick={() => useTimelineStore.getState().resetZoom()}
          className="min-h-[28px] rounded px-1.5 py-0.5"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          Reset zoom
        </button>
        <button
          type="button"
          onClick={() => {
            const tl = useTimelineStore.getState();
            tl.openEditor(usePlaybackStore.getState().currentFrameIndex);
          }}
          disabled={totalFrames <= 1}
          className="min-h-[28px] rounded px-1.5 py-0.5 disabled:opacity-40"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          Add note (B)
        </button>
      </div>
      <BookmarkListSidebar rowIndices={rowIndices} />
      <BookmarkEditorPopover />
    </div>
  );
}

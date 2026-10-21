import { useEffect, useMemo } from 'react';
import { usePlaybackStore } from '../store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
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

  usePlaybackRafLoop();
  usePlaybackKeyboardShortcuts(true);

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
            Space = play · ←/→ = ±1 (Shift = ±10) · Home/End · [ ] = loop A/B · L = loop
          </span>
        )}
      </div>
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

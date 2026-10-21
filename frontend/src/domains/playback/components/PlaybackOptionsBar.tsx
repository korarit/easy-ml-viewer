import { Repeat, Repeat1, Ban } from 'lucide-react';
import { usePlaybackStore } from '../store/usePlaybackStore';
import { SPEED_PRESETS, TRAIL_LENGTH_OPTIONS, type LoopMode } from '../types/playback.types';

const LOOP_LABEL: Record<LoopMode, string> = {
  none: 'No Loop (Stop at End)',
  all: 'Loop All',
  ab_range: 'A-B Range Loop',
};

function LoopIcon({ mode }: { mode: LoopMode }) {
  if (mode === 'none') return <Ban size={14} />;
  if (mode === 'ab_range') return <Repeat1 size={14} />;
  return <Repeat size={14} />;
}

/**
 * Phase 10 — speed, loop-mode and motion-trail settings bar.
 */
export function PlaybackOptionsBar({ loopLabel }: { loopLabel?: string }) {
  const speedMultiplier = usePlaybackStore((s) => s.speedMultiplier);
  const loopMode = usePlaybackStore((s) => s.loopMode);
  const loopRange = usePlaybackStore((s) => s.loopRange);
  const trailLength = usePlaybackStore((s) => s.trailLength);
  const smoothInterpolation = usePlaybackStore((s) => s.smoothInterpolation);
  const totalFrames = usePlaybackStore((s) => s.totalFrames);

  const locked = totalFrames <= 1;

  return (
    <div className="flex flex-wrap items-center gap-3 text-xs" role="toolbar" aria-label="Playback options">
      <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
        Speed
        <select
          aria-label="Playback speed"
          disabled={locked}
          value={String(speedMultiplier)}
          onChange={(e) => usePlaybackStore.getState().setSpeedMultiplier(Number(e.target.value))}
          className="font-mono-tabular rounded px-1.5 py-1 disabled:opacity-40"
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
        className="flex items-center gap-1.5 rounded-md px-2 py-1 disabled:opacity-40"
        style={{
          border: '1px solid var(--border-strong)',
          color: loopMode === 'none' ? 'var(--text-muted)' : '#f59e0b',
          background: loopMode === 'none' ? 'transparent' : 'rgba(245,158,11,0.12)',
        }}
      >
        <LoopIcon mode={loopMode} />
        {loopMode === 'ab_range' && loopRange
          ? `A-B (${loopRange[0]}–${loopRange[1]})`
          : LOOP_LABEL[loopMode]}
      </button>
      {loopLabel && (
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
          {loopLabel}
        </span>
      )}

      <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
        Trail
        <select
          aria-label="Motion trail length"
          value={String(trailLength)}
          onChange={(e) => usePlaybackStore.getState().setTrailLength(Number(e.target.value))}
          className="font-mono-tabular rounded px-1.5 py-1"
          style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)' }}
        >
          {TRAIL_LENGTH_OPTIONS.map((n) => (
            <option key={n} value={String(n)}>
              {n === 0 ? 'Off' : n === 10 ? 'Short (10)' : n === 25 ? 'Medium (25)' : 'Long (50)'}
            </option>
          ))}
        </select>
      </label>

      <label
        className="flex cursor-pointer items-center gap-1.5"
        style={{ color: 'var(--text-secondary)' }}
        title="Smooth 60FPS position interpolation (lerp) between frames"
      >
        <input
          type="checkbox"
          aria-label="Smooth interpolation"
          checked={smoothInterpolation}
          onChange={(e) => usePlaybackStore.getState().setSmoothInterpolation(e.target.checked)}
        />
        Interpolate
      </label>
    </div>
  );
}

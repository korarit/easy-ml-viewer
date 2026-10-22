import { useCallback, useRef } from 'react';
import {
  Pause,
  Play,
  SkipBack,
  SkipForward,
  StepBack,
  StepForward,
} from 'lucide-react';
import { usePlaybackStore } from '../store/usePlaybackStore';
import { useSystemStore } from '../../../core/store/useSystemStore';
import { FrameReadoutBadge } from './FrameReadoutBadge';

function TransportButton({
  label,
  title,
  disabled,
  onClick,
  onHold,
  active,
  children,
}: {
  label: string;
  title: string;
  disabled?: boolean;
  onClick: () => void;
  onHold?: () => void;
  active?: boolean;
  children: React.ReactNode;
}) {
  const holdTimer = useRef<number | null>(null);

  const clearHold = useCallback(() => {
    if (holdTimer.current !== null) {
      window.clearInterval(holdTimer.current);
      holdTimer.current = null;
    }
  }, []);

  const beginHold = useCallback(
    (e: React.PointerEvent) => {
      if (disabled || !onHold) return;
      (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
      // Initial step fires on click; holding repeats continuously.
      holdTimer.current = window.setInterval(onHold, 80);
    },
    [disabled, onHold],
  );

  return (
    <button
      type="button"
      aria-label={label}
      title={title}
      disabled={disabled}
      onClick={onClick}
      onPointerDown={beginHold}
      onPointerUp={clearHold}
      onPointerLeave={clearHold}
      onPointerCancel={clearHold}
      className="flex min-h-[36px] min-w-[36px] items-center justify-center rounded-md p-2 transition-colors disabled:cursor-not-allowed disabled:opacity-35"
      style={{
        border: '1px solid var(--border-strong)',
        color: 'var(--text-primary)',
        background: active ? 'rgba(52,211,153,0.16)' : 'transparent',
        boxShadow: active ? '0 0 12px rgba(52,211,153,0.45)' : undefined,
      }}
    >
      {children}
    </button>
  );
}

/**
 * Phase 10 — VCR transport control deck (play/pause/step/first/last).
 * Step buttons support press-and-hold continuous scrubbing.
 */
export function VcrTransportDeck({ rowIndex }: { rowIndex: number | null }) {
  const status = usePlaybackStore((s) => s.status);
  const totalFrames = usePlaybackStore((s) => s.totalFrames);
  const currentFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const setActiveStep = useSystemStore((s) => s.setActiveStep);

  const locked = totalFrames <= 1;
  const isPlaying = status === 'playing';

  if (locked) {
    return (
      <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="VCR transport controls">
        <button
          type="button"
          disabled
          className="flex min-h-[40px] cursor-not-allowed items-center gap-2 rounded-md px-4 py-2 text-sm font-semibold opacity-40"
          style={{ border: '1px solid var(--border-strong)', background: 'var(--ml-primary)', color: '#fff' }}
          title="Playback needs at least 2 frames"
        >
          <Play size={16} /> PLAY
        </button>
        <button
          type="button"
          onClick={() => setActiveStep(4)}
          className="min-h-[40px] rounded-md px-3 py-2 text-xs font-medium"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          ← Pick at least 2 rows in Step 4 to play
        </button>
      </div>
    );
  }

  return (
    <div
      className="flex flex-wrap items-center gap-2"
      role="toolbar"
      aria-label="VCR transport controls"
    >
      <TransportButton
        label="First frame"
        title="First frame (Home)"
        disabled={locked}
        onClick={() => usePlaybackStore.getState().seekToFrame(0)}
      >
        <SkipBack size={16} />
      </TransportButton>
      <TransportButton
        label="Step back"
        title="Step back 1 frame (←, hold to rewind)"
        disabled={locked}
        onClick={() => usePlaybackStore.getState().stepFrame(-1)}
        onHold={() => usePlaybackStore.getState().stepFrame(-1)}
      >
        <StepBack size={16} />
      </TransportButton>
      <button
        type="button"
        aria-label={isPlaying ? 'Pause' : 'Play'}
        title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
        disabled={locked}
        onClick={() => usePlaybackStore.getState().togglePlayPause()}
        className="flex min-h-[40px] items-center gap-2 rounded-md px-4 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-35"
        style={{
          border: '1px solid var(--border-strong)',
          background: isPlaying ? 'rgba(52,211,153,0.18)' : 'var(--ml-primary)',
          color: isPlaying ? '#34d399' : '#fff',
          boxShadow: isPlaying ? '0 0 14px rgba(52,211,153,0.55)' : undefined,
        }}
      >
        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        {isPlaying ? 'PAUSE' : 'PLAY'}
        {isPlaying && (
          <span
            aria-hidden
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: '#34d399', boxShadow: '0 0 8px #34d399' }}
          />
        )}
      </button>
      <TransportButton
        label="Step forward"
        title="Step forward 1 frame (→, hold to fast-forward)"
        disabled={locked}
        onClick={() => usePlaybackStore.getState().stepFrame(1)}
        onHold={() => usePlaybackStore.getState().stepFrame(1)}
      >
        <StepForward size={16} />
      </TransportButton>
      <TransportButton
        label="Last frame"
        title="Last frame (End)"
        disabled={locked}
        onClick={() => usePlaybackStore.getState().seekToFrame(totalFrames - 1)}
      >
        <SkipForward size={16} />
      </TransportButton>
      <FrameReadoutBadge
        frameIndex={currentFrameIndex}
        totalFrames={totalFrames}
        rowIndex={rowIndex}
      />
    </div>
  );
}

import { useState } from 'react';
import { usePlaybackStore } from '../store/usePlaybackStore';

function pad4(n: number) {
  return String(Math.max(0, n)).padStart(4, '0');
}

/**
 * Phase 10 — frame counter + timecode readout with inline jump input.
 * Click the frame number to type a target frame and jump to it.
 */
export function FrameReadoutBadge({
  frameIndex,
  totalFrames,
  rowIndex,
}: {
  frameIndex: number;
  totalFrames: number;
  rowIndex: number | null;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');

  if (totalFrames <= 0) {
    return (
      <span
        className="font-mono-tabular rounded px-2 py-1 text-xs"
        style={{ color: 'var(--text-muted)', border: '1px dashed var(--border-subtle)' }}
      >
        FRAME ---- / 0
      </span>
    );
  }

  const commitJump = () => {
    const parsed = Number.parseInt(draft, 10);
    if (Number.isFinite(parsed)) {
      usePlaybackStore.getState().seekToFrame(parsed);
    }
    setEditing(false);
  };

  return (
    <span
      className="font-mono-tabular rounded px-2 py-1 text-xs"
      style={{ color: 'var(--text-primary)', border: '1px solid var(--border-subtle)', background: 'rgba(10,15,22,0.5)' }}
    >
      FRAME{' '}
      {editing ? (
        <input
          // eslint-disable-next-line jsx-a11y/no-autofocus
          autoFocus
          aria-label="Jump to frame number"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitJump}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitJump();
            if (e.key === 'Escape') setEditing(false);
            e.stopPropagation();
          }}
          className="w-14 rounded px-1 text-xs"
          style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)' }}
        />
      ) : (
        <button
          type="button"
          aria-label="Jump to frame (click to type a frame number)"
          title="Click to jump to a frame number"
          onClick={() => {
            setDraft(String(frameIndex));
            setEditing(true);
          }}
          className="underline decoration-dotted underline-offset-2"
          style={{ color: '#7dd3fc' }}
        >
          {pad4(frameIndex)}
        </button>
      )}{' '}
      / {totalFrames.toLocaleString('en-US')}
      {rowIndex !== null && (
        <span style={{ color: 'var(--text-secondary)' }}> (Row #{rowIndex})</span>
      )}
    </span>
  );
}

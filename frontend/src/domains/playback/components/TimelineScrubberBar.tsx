import { useEffect, useMemo, useRef, useState } from 'react';
import { usePlaybackStore } from '../store/usePlaybackStore';

function pctOf(frame: number, total: number) {
  if (total <= 1) return 0;
  return (Math.max(0, Math.min(total - 1, frame)) / (total - 1)) * 100;
}

function frameAt(clientX: number, rect: DOMRect, total: number) {
  if (total <= 1) return 0;
  const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(rect.width, 1)));
  return Math.round(ratio * (total - 1));
}

function formatPred(v: number | null | undefined) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const abs = Math.abs(v);
  if (abs !== 0 && (abs >= 1_000_000 || abs < 0.001)) return v.toExponential(2);
  if (Number.isInteger(v)) return String(v);
  return String(Math.round(v * 10000) / 10000);
}

/**
 * Phase 10 — interactive high-precision scrubber bar.
 *
 * 60FPS rule: the playhead is positioned imperatively (ref + rAF reading
 * `getState().exactFrameFloat`) so scrubbing/playback never re-renders
 * React. Loop brackets and discontinuity pins are declarative (rare updates).
 */
export function TimelineScrubberBar({
  rowIndices,
  predictions,
  discontinuityFrames,
}: {
  rowIndices: number[];
  predictions: (number | null)[];
  discontinuityFrames?: number[];
}) {
  const totalFrames = usePlaybackStore((s) => s.totalFrames);
  const currentFrameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const loopMode = usePlaybackStore((s) => s.loopMode);
  const loopRange = usePlaybackStore((s) => s.loopRange);
  const status = usePlaybackStore((s) => s.status);

  const trackRef = useRef<HTMLDivElement>(null);
  const fillRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<HTMLDivElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [hoverX, setHoverX] = useState(0);
  const scrubPointerId = useRef<number | null>(null);

  // Imperative 60fps playhead (exactFrameFloat, sub-frame smooth).
  useEffect(() => {
    let raf = 0;
    const update = () => {
      const st = usePlaybackStore.getState();
      const pct = pctOf(st.exactFrameFloat, st.totalFrames);
      if (fillRef.current) fillRef.current.style.width = `${pct}%`;
      if (headRef.current) headRef.current.style.left = `${pct}%`;
      raf = requestAnimationFrame(update);
    };
    raf = requestAnimationFrame(update);
    return () => cancelAnimationFrame(raf);
  }, []);

  const showLoopBand = loopMode === 'ab_range' && loopRange && totalFrames > 1;

  const pins = useMemo(() => {
    if (!discontinuityFrames || totalFrames <= 1) return [];
    return discontinuityFrames.filter((f) => f >= 0 && f < totalFrames);
  }, [discontinuityFrames, totalFrames]);

  if (totalFrames <= 0) {
    return (
      <div
        className="rounded-md px-3 py-3 text-xs"
        role="status"
        style={{ background: 'rgba(148,163,184,0.08)', border: '1px dashed var(--border-subtle)', color: 'var(--text-muted)' }}
      >
        No frames loaded in playback queue
      </div>
    );
  }

  const locked = totalFrames === 1;

  const seekFromEvent = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    usePlaybackStore.getState().seekToFrame(frameAt(clientX, el.getBoundingClientRect(), totalFrames));
  };

  return (
    <div className="relative w-full">
      {hoverIdx !== null && !locked && (
        <div
          className="font-mono-tabular pointer-events-none absolute top-0 z-10 rounded px-2 py-1 text-[11px]"
          style={{
            left: hoverX,
            transform: 'translate(-50%, -110%)',
            background: 'rgba(10,15,22,0.92)',
            border: '1px solid var(--border-strong)',
            color: 'var(--text-primary)',
            whiteSpace: 'nowrap',
          }}
          role="status"
        >
          Frame #{hoverIdx}
          {rowIndices[hoverIdx] !== undefined && ` | Row #${rowIndices[hoverIdx]}`}
          {` | Pred: ${formatPred(predictions[hoverIdx])}`}
        </div>
      )}
      <div
        ref={trackRef}
        role="slider"
        aria-label="Playback timeline scrubber"
        aria-valuemin={0}
        aria-valuemax={Math.max(totalFrames - 1, 0)}
        aria-valuenow={Math.min(currentFrameIndex, Math.max(totalFrames - 1, 0))}
        aria-disabled={locked}
        tabIndex={locked ? -1 : 0}
        className="relative h-8 w-full rounded-md"
        style={{
          background: locked
            ? 'rgba(148,163,184,0.12)'
            : 'linear-gradient(180deg, rgba(125,211,252,0.10), rgba(125,211,252,0.04))',
          border: '1px solid var(--border-subtle)',
          cursor: locked ? 'not-allowed' : 'crosshair',
          touchAction: 'none',
        }}
        onPointerDown={(e) => {
          if (locked) return;
          scrubPointerId.current = e.pointerId;
          (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
          usePlaybackStore.getState().startScrubbing();
          seekFromEvent(e.clientX);
        }}
        onPointerMove={(e) => {
          const el = trackRef.current;
          if (!el) return;
          const rect = el.getBoundingClientRect();
          if (scrubPointerId.current === e.pointerId) {
            seekFromEvent(e.clientX);
          } else if (!locked && e.pointerType === 'mouse') {
            setHoverIdx(frameAt(e.clientX, rect, totalFrames));
            setHoverX(e.clientX - rect.left);
          }
        }}
        onPointerUp={(e) => {
          if (scrubPointerId.current === e.pointerId) {
            scrubPointerId.current = null;
            usePlaybackStore.getState().endScrubbing();
          }
        }}
        onPointerCancel={() => {
          scrubPointerId.current = null;
          if (usePlaybackStore.getState().status === 'scrubbing') {
            usePlaybackStore.getState().endScrubbing();
          }
        }}
        onPointerLeave={() => {
          if (scrubPointerId.current === null) setHoverIdx(null);
        }}
      >
        {/* Elapsed fill (imperative width). */}
        <div
          ref={fillRef}
          className="absolute top-0 left-0 h-full rounded-l-md"
          style={{ width: '0%', background: 'rgba(125,211,252,0.25)', pointerEvents: 'none' }}
        />
        {/* A-B loop band. */}
        {showLoopBand && (
          <div
            className="absolute top-0 h-full"
            style={{
              left: `${pctOf(loopRange[0], totalFrames)}%`,
              width: `${Math.max(pctOf(loopRange[1], totalFrames) - pctOf(loopRange[0], totalFrames), 0.5)}%`,
              background: 'rgba(245,158,11,0.14)',
              borderLeft: '2px solid #f59e0b',
              borderRight: '2px solid #f59e0b',
              pointerEvents: 'none',
            }}
          />
        )}
        {/* Discontinuity pins. */}
        {pins.map((f) => (
          <div
            key={f}
            title={`Spatial jump at frame ${f}`}
            className="absolute top-0 h-full w-[2px]"
            style={{ left: `${pctOf(f, totalFrames)}%`, background: '#f43e5e', opacity: 0.8, pointerEvents: 'none' }}
          />
        ))}
        {/* A / B draggable brackets. */}
        {showLoopBand && (
          <>
            {(['A', 'B'] as const).map((point) => {
              const frame = point === 'A' ? loopRange[0] : loopRange[1];
              return (
                <div
                  key={point}
                  role="button"
                  aria-label={`Loop point ${point} (frame ${frame}) — drag to adjust`}
                  title={`Loop ${point} @ frame ${frame} (drag, or press ${point === 'A' ? '[' : ']'} here)`}
                  className="font-mono-tabular absolute top-[-6px] flex h-[calc(100%+12px)] w-5 cursor-ew-resize items-center justify-center text-[10px] font-bold"
                  style={{
                    left: `calc(${pctOf(frame, totalFrames)}% - 10px)`,
                    color: '#f59e0b',
                    background: 'rgba(245,158,11,0.22)',
                    border: '1px solid #f59e0b',
                    borderRadius: 4,
                    touchAction: 'none',
                  }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
                    const move = (ev: PointerEvent) => {
                      const el = trackRef.current;
                      if (!el) return;
                      usePlaybackStore
                        .getState()
                        .setLoopRangePoint(point, frameAt(ev.clientX, el.getBoundingClientRect(), totalFrames));
                    };
                    const up = () => {
                      window.removeEventListener('pointermove', move);
                      window.removeEventListener('pointerup', up);
                      window.removeEventListener('pointercancel', up);
                    };
                    window.addEventListener('pointermove', move);
                    window.addEventListener('pointerup', up);
                    window.addEventListener('pointercancel', up);
                  }}
                >
                  {point}
                </div>
              );
            })}
          </>
        )}
        {/* Playhead handle + vertical guide (imperative left). */}
        <div
          ref={headRef}
          className="absolute top-[-4px] h-[calc(100%+8px)]"
          style={{ left: '0%', width: 0, pointerEvents: 'none' }}
        >
          <div
            className="absolute top-0 h-full w-[2px]"
            style={{ background: '#7dd3fc', boxShadow: '0 0 8px #7dd3fc' }}
          />
          <div
            className="absolute top-1/2 h-3.5 w-3.5 rounded-full"
            style={{
              background: '#7dd3fc',
              border: '2px solid #0b0f14',
              boxShadow: '0 0 10px #7dd3fc',
              transform: 'translate(-40%, -50%)',
            }}
          />
        </div>
      </div>
      <div
        className="font-mono-tabular mt-1 flex justify-between text-[10px]"
        style={{ color: 'var(--text-muted)' }}
      >
        <span>0000</span>
        {status === 'scrubbing' && <span style={{ color: '#7dd3fc' }}>SCRUBBING…</span>}
        <span>{String(Math.max(totalFrames - 1, 0)).padStart(4, '0')}</span>
      </div>
    </div>
  );
}

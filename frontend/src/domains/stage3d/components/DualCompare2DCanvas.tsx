import { useEffect, useMemo } from 'react';
import { nextConflictFrame, prevConflictFrame } from '../../recording/api/dual2DCompareApi';
import { useDual2DCompareStore, type Dual2DLayout } from '../store/useDual2DCompareStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';

export interface DualCompare2DBoundary {
  threshold: number;
  classIndex: number;
  kind: string;
  lineSegments: number[][];
}

interface DualCompare2DCanvasProps {
  modelAName: string;
  modelBName: string;
  boundariesA: DualCompare2DBoundary[];
  boundariesB: DualCompare2DBoundary[];
  disagreementAreaPct: number;
  consensusNotice?: string;
  conflictFrameIndices?: number[];
  layout?: Dual2DLayout;
  showConflictHatch?: boolean;
  onNextConflict?: (frame: number) => void;
}

const WORLD_HALF = 5.5;
const MODEL_A_COLOR = '#38bdf8';
const MODEL_B_COLOR = '#fb923c';

/**
 * Phase 48 — Synchronized Dual-Model Compare in the 2D planar viewport.
 *
 * Two layouts on one orthographic stage:
 *  - `superimposed`: both decision boundaries overlaid (A = sky blue,
 *    B = amber) with the disagreement area under a striped magenta hatch.
 *  - `side-by-side`: two half-width SVG stages sharing one conflict badge.
 * Alt+D jumps the VCR to the next frame where the models disagree.
 */
export function DualCompare2DCanvas({
  modelAName,
  modelBName,
  boundariesA,
  boundariesB,
  disagreementAreaPct,
  consensusNotice,
  conflictFrameIndices = [],
  layout = 'superimposed',
  showConflictHatch = true,
  onNextConflict,
}: DualCompare2DCanvasProps) {
  const focusedConflict = useDual2DCompareStore((s) => s.focusedConflict);
  const setFocusedConflict = useDual2DCompareStore((s) => s.setFocusedConflict);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!e.altKey || e.key.toLowerCase() !== 'd') return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
      e.preventDefault();
      const cursor = focusedConflict ?? usePlaybackStore.getState().currentFrameIndex;
      const next = nextConflictFrame(conflictFrameIndices, cursor);
      if (next !== null) {
        setFocusedConflict(next);
        usePlaybackStore.getState().seekToFrame(next, true);
        onNextConflict?.(next);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [conflictFrameIndices, focusedConflict, onNextConflict, setFocusedConflict]);

  const segsA = useMemo(
    () => boundariesA.filter((b) => b.kind === 'decision').flatMap((b) => b.lineSegments),
    [boundariesA],
  );
  const segsB = useMemo(
    () => boundariesB.filter((b) => b.kind === 'decision').flatMap((b) => b.lineSegments),
    [boundariesB],
  );

  const jumpNext = () => {
    const cursor = focusedConflict ?? usePlaybackStore.getState().currentFrameIndex;
    const next = nextConflictFrame(conflictFrameIndices, cursor);
    if (next !== null) {
      setFocusedConflict(next);
      onNextConflict?.(next);
    }
  };
  const jumpPrev = () => {
    const cursor = focusedConflict ?? usePlaybackStore.getState().currentFrameIndex;
    const prev = prevConflictFrame(conflictFrameIndices, cursor);
    if (prev !== null) {
      setFocusedConflict(prev);
      onNextConflict?.(prev);
    }
  };

  const renderStage = (key: string, segs: number[][], color: string, label: string, half: boolean) => (
    <svg
      key={key}
      viewBox={`${-WORLD_HALF} ${-WORLD_HALF} ${WORLD_HALF * 2} ${WORLD_HALF * 2}`}
      className={half ? 'h-full w-1/2' : 'h-full w-full'}
      role="img"
      aria-label={`${label} 2D decision boundary`}
    >
      <defs>
        <pattern
          id={`${key}-conflict-hatch`}
          data-testid="conflict-hatch-pattern"
          width="0.35"
          height="0.35"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <rect width="0.35" height="0.35" fill="rgba(217,70,239,0.10)" />
          <line x1="0" y1="0" x2="0" y2="0.35" stroke="#d946ef" strokeWidth="0.07" opacity="0.85" />
        </pattern>
      </defs>
      <rect x={-5} y={-5} width={10} height={10} fill="none" stroke="#334155" strokeWidth={0.03} />
      {showConflictHatch && disagreementAreaPct > 0 && (
        <rect x={-5} y={-5} width={10} height={10} fill={`url(#${key}-conflict-hatch)`} opacity={0.55} />
      )}
      <g stroke={color} strokeWidth={0.06} fill="none" strokeLinecap="round">
        {segs.slice(0, 2000).map((s, i) => (
          <line key={i} x1={s[0]} y1={-s[1]} x2={s[2]} y2={-s[3]} />
        ))}
      </g>
      <text x={-4.9} y={4.6} fontSize={0.32} fill={color}>
        {label}
      </text>
    </svg>
  );

  return (
    <div className="flex h-full w-full flex-col" data-testid="dual-compare-2d-canvas">
      <div
        className="font-mono-tabular flex flex-wrap items-center gap-2 px-2 py-1 text-[11px]"
        style={{ color: 'var(--text-secondary)' }}
      >
        <span style={{ color: MODEL_A_COLOR }}>━ {modelAName}</span>
        <span style={{ color: MODEL_B_COLOR }}>━ {modelBName}</span>
        <span
          className="rounded px-2 py-0.5"
          style={{ background: 'rgba(217,70,239,0.14)', color: '#e879f9' }}
        >
          {disagreementAreaPct.toFixed(1)}% Disagreement Area
        </span>
        {consensusNotice && (
          <span className="rounded px-2 py-0.5" style={{ background: 'var(--surface-elevated)' }}>
            {consensusNotice}
          </span>
        )}
        <span className="ml-auto flex items-center gap-1">
          <button
            onClick={jumpPrev}
            disabled={conflictFrameIndices.length === 0}
            className="rounded px-2 py-0.5 disabled:opacity-40"
            style={{ border: '1px solid var(--border-subtle)' }}
            title="Previous conflict frame"
          >
            ◀ Prev
          </button>
          <button
            onClick={jumpNext}
            disabled={conflictFrameIndices.length === 0}
            className="rounded px-2 py-0.5 disabled:opacity-40"
            style={{ border: '1px solid var(--border-subtle)' }}
            title="Next conflict (Alt+D)"
          >
            Next Conflict (Alt+D) ▶
          </button>
        </span>
      </div>
      <div className="flex min-h-0 flex-1 flex-row items-stretch">
        {layout === 'side-by-side' ? (
          <>
            {renderStage('dual-a', segsA, MODEL_A_COLOR, `A: ${modelAName}`, true)}
            {renderStage('dual-b', segsB, MODEL_B_COLOR, `B: ${modelBName}`, true)}
          </>
        ) : (
          <svg
            viewBox={`${-WORLD_HALF} ${-WORLD_HALF} ${WORLD_HALF * 2} ${WORLD_HALF * 2}`}
            className="h-full w-full"
            role="img"
            aria-label={`Superimposed 2D decision boundaries: ${modelAName} versus ${modelBName}`}
          >
            <defs>
              <pattern
                id="dual-super-conflict-hatch"
                data-testid="conflict-hatch-pattern"
                width="0.35"
                height="0.35"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <rect width="0.35" height="0.35" fill="rgba(217,70,239,0.10)" />
                <line x1="0" y1="0" x2="0" y2="0.35" stroke="#d946ef" strokeWidth="0.07" opacity="0.85" />
              </pattern>
            </defs>
            <rect x={-5} y={-5} width={10} height={10} fill="none" stroke="#334155" strokeWidth={0.03} />
            {showConflictHatch && disagreementAreaPct > 0 && (
              <rect x={-5} y={-5} width={10} height={10} fill="url(#dual-super-conflict-hatch)" opacity={0.55} />
            )}
            <g stroke={MODEL_B_COLOR} strokeWidth={0.07} fill="none" strokeLinecap="round" opacity={0.95}>
              {segsB.slice(0, 2000).map((s, i) => (
                <line key={`b-${i}`} x1={s[0]} y1={-s[1]} x2={s[2]} y2={-s[3]} />
              ))}
            </g>
            <g stroke={MODEL_A_COLOR} strokeWidth={0.07} fill="none" strokeLinecap="round" opacity={0.95}>
              {segsA.slice(0, 2000).map((s, i) => (
                <line key={`a-${i}`} x1={s[0]} y1={-s[1]} x2={s[2]} y2={-s[3]} />
              ))}
            </g>
          </svg>
        )}
      </div>
      <div className="font-mono-tabular px-2 py-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        {conflictFrameIndices.length > 0 ? (
          <>
            {conflictFrameIndices.length} conflict frames · focused:{' '}
            {focusedConflict !== null ? `Frame ${focusedConflict}` : '—'}
          </>
        ) : (
          'No frame conflicts — models agree on every queued row.'
        )}
      </div>
    </div>
  );
}

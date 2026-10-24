import { wy } from '../DecisionBoundaryContour2D';
import { cometAlpha, cometStrokeWidth } from './pointIntelligence2dMath';

export interface BayesEntropyCometTrail2DProps {
  /** World-space trail points in frame order. */
  points: Array<{ x: number; y: number }>;
  /** Per-point trail color (class color or magenta ambiguity glow). */
  colors: string[];
  activeFrame: number;
  /** Half-window W: renders segments for [t − W, t]. */
  window?: number;
  visible?: boolean;
}

/**
 * Phase 52 (mechanism A) — posterior-entropy comet trail.
 *
 * Same exponential-decay kernel as the Phase-49 trail, but each segment is
 * tinted by the frame's Shannon entropy: the predicted class color when the
 * model is sure, neon magenta while the queue crosses the ambiguity zone
 * (DoD #1). One <line> per in-window segment (≤ W nodes) holds 60 FPS.
 */
export function BayesEntropyCometTrail2D({
  points,
  colors,
  activeFrame,
  window = 8,
  visible = true,
}: BayesEntropyCometTrail2DProps) {
  if (!visible || points.length < 2) return null;
  const w = Math.max(1, Math.round(window));
  const t = Math.max(0, Math.min(activeFrame, points.length - 1));
  const lo = Math.max(0, t - w);
  const segs: Array<{ x1: number; y1: number; x2: number; y2: number; opacity: number; width: number; color: string }> = [];
  for (let f = lo; f < t; f += 1) {
    const a = points[f];
    const b = points[f + 1];
    if (!a || !b) continue;
    segs.push({
      x1: a.x,
      y1: wy(a.y),
      x2: b.x,
      y2: wy(b.y),
      opacity: Number(cometAlpha(f + 1, t, w).toFixed(3)),
      width: cometStrokeWidth(f + 1, t, w),
      color: colors[f + 1] ?? colors[f] ?? '#22d3ee',
    });
  }
  if (segs.length === 0) return null;
  return (
    <g className="bayes-entropy-trail-2d pointer-events-none" aria-label="Entropy comet trail">
      {segs.map((s, i) => (
        <line
          key={lo + i}
          x1={s.x1}
          y1={s.y1}
          x2={s.x2}
          y2={s.y2}
          stroke={s.color}
          strokeWidth={s.width}
          opacity={s.opacity}
          strokeLinecap="round"
        />
      ))}
    </g>
  );
}

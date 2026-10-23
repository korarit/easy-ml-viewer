import { wy } from '../DecisionBoundaryContour2D';
import { cometAlpha, cometStrokeWidth } from './pointIntelligence2dMath';
import { trailSignColor } from './linearPointIntel2dMath';

export interface CometTrail2DProps {
  /** World-space trail points in frame order. */
  points: Array<{ x: number; y: number }>;
  activeFrame: number;
  /** Half-window W: renders segments for [t − W, t]. */
  window?: number;
  stroke?: string;
  visible?: boolean;
  /** Phase 50 — per-segment ascent signs (+1 green / −1 orange / 0 flat). */
  signs?: number[];
}

/**
 * Phase 49 (mechanism A) — Ghost & Trajectory Comet Trail.
 *
 * Past frames fade with the exponential-decay kernel α(f) while the stroke
 * widens toward the head frame. One <line> per in-window segment (≤ W
 * nodes) keeps scrubbing at 60 FPS.
 */
export function CometTrail2D({
  points,
  activeFrame,
  window = 8,
  stroke = '#22d3ee',
  visible = true,
  signs,
}: CometTrail2DProps) {
  if (!visible || points.length < 2) return null;
  const w = Math.max(1, Math.round(window));
  const t = Math.max(0, Math.min(activeFrame, points.length - 1));
  const lo = Math.max(0, t - w);
  const segs: Array<{ x1: number; y1: number; x2: number; y2: number; opacity: number; width: number }> = [];
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
    });
  }
  if (segs.length === 0) return null;
  return (
    <g className="comet-trail-2d pointer-events-none" aria-label="Comet trail">
      {segs.map((s, i) => (
        <line
          key={lo + i}
          x1={s.x1}
          y1={s.y1}
          x2={s.x2}
          y2={s.y2}
          stroke={signs && signs[lo + i] !== undefined ? trailSignColor(signs[lo + i]!, stroke) : stroke}
          strokeWidth={s.width}
          opacity={s.opacity}
          strokeLinecap="round"
        />
      ))}
    </g>
  );
}

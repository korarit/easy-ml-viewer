import { wy } from '../DecisionBoundaryContour2D';
import { cometAlpha, cometStrokeWidth } from './pointIntelligence2dMath';
import { manhattanElbow } from './treesPointIntel2dMath';

export interface ManhattanStepTrail2DProps {
  /** World-space trail points in frame order. */
  points: Array<{ x: number; y: number }>;
  activeFrame: number;
  /** Half-window W: renders hops for [t − W, t]. */
  window?: number;
  /** Destination leaf box to pulse after a branch switch (world [xmin,xmax,ymin,ymax]). */
  pulseBox?: [number, number, number, number] | null;
  visible?: boolean;
}

/**
 * Phase 53 (mechanism A) — orthogonal leaf-hopping trail.
 *
 * Each hop renders as an L-shaped Manhattan step (horizontal then
 * vertical), echoing the axis-aligned branch switches of a decision
 * tree, with the Phase-49 exponential-decay kernel for opacity/width.
 * When the walk crosses a split line, the destination leaf box pulses
 * (Leaf Transition Pulse, spec §2.1). Two segments per hop (≤ 2W lines).
 */
export function ManhattanStepTrail2D({
  points,
  activeFrame,
  window = 8,
  pulseBox = null,
  visible = true,
}: ManhattanStepTrail2DProps) {
  if (!visible || points.length < 2) return null;
  const w = Math.max(1, Math.round(window));
  const t = Math.max(0, Math.min(activeFrame, points.length - 1));
  const lo = Math.max(0, t - w);
  const segs: Array<{ x1: number; y1: number; x2: number; y2: number; opacity: number; width: number }> = [];
  for (let f = lo; f < t; f += 1) {
    const a = points[f];
    const b = points[f + 1];
    if (!a || !b) continue;
    const e = manhattanElbow(a, b, true);
    if (!e) continue;
    const opacity = Number(cometAlpha(f + 1, t, w).toFixed(3));
    const width = cometStrokeWidth(f + 1, t, w);
    segs.push(
      { x1: a.x, y1: wy(a.y), x2: e.x, y2: wy(e.y), opacity, width },
      { x1: e.x, y1: wy(e.y), x2: b.x, y2: wy(b.y), opacity, width },
    );
  }
  if (segs.length === 0 && !pulseBox) return null;
  return (
    <g className="manhattan-trail-2d pointer-events-none" aria-label="Orthogonal leaf-hopping trail">
      {segs.map((s, i) => (
        <line
          key={lo + i}
          x1={s.x1}
          y1={s.y1}
          x2={s.x2}
          y2={s.y2}
          stroke="#22d3ee"
          strokeWidth={s.width}
          opacity={s.opacity}
          strokeLinecap="round"
        />
      ))}
      {pulseBox && (
        <rect
          x={pulseBox[0]}
          y={wy(pulseBox[3])}
          width={Math.max(pulseBox[1] - pulseBox[0], 1e-6)}
          height={Math.max(pulseBox[3] - pulseBox[2], 1e-6)}
          fill="none"
          stroke="#22d3ee"
          strokeWidth={0.05}
          opacity={0.9}
        >
          <title>Leaf transition pulse — the walk crossed a split line</title>
          <animate attributeName="opacity" values="0.9;0.25;0.9" dur="1.6s" repeatCount="indefinite" />
        </rect>
      )}
    </g>
  );
}

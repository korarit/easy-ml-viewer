import { wy } from '../DecisionBoundaryContour2D';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';
import { arcDonutPath, type VoteArcSegment } from './distancePointIntel2dMath';

export interface KnnVoteArcRing2DProps {
  /** World center (the active query point). */
  x: number;
  y: number;
  /** Pie shares tiling the full turn (DoD #3). */
  segments: VoteArcSegment[];
  visible?: boolean;
}

/**
 * Phase 51 (mechanism B, KNN) — vote-breakdown ring.
 *
 * The outer Dual-Ring is split into donut arcs proportional to the true
 * neighbor vote shares (e.g. 60/40 → 0.6/0.4 turns), so the eye reads the
 * majority at a glance. One path per segment (≤ classes draw calls).
 */
export function KnnVoteArcRing2D({ x, y, segments, visible = true }: KnnVoteArcRing2DProps) {
  if (!visible || segments.length === 0) return null;
  const rOuter = RING_OUTER_RADIUS + 0.05;
  const rInner = RING_OUTER_RADIUS - 0.02;
  const cy = wy(y);
  return (
    <g className="knn-vote-arc-2d pointer-events-none" aria-label="KNN vote breakdown ring">
      {segments.map((s) => (
        <path
          key={`${s.label}-${s.start.toFixed(4)}`}
          d={arcDonutPath(x, cy, rOuter, rInner, s.start, s.sweep)}
          fill={s.color}
          fillOpacity={0.9}
          stroke="#0b0f14"
          strokeWidth={0.012}
        >
          <title>{`Class ${s.label}: ${(s.share * 100).toFixed(1)}% of the K votes`}</title>
        </path>
      ))}
    </g>
  );
}

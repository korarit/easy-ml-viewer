import { wy } from '../DecisionBoundaryContour2D';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';
import {
  CONSENSUS_FUZZY_COLOR,
  CONSENSUS_SOLID_COLOR,
  consensusThickness,
  isFuzzyConsensus,
} from './treesPointIntel2dMath';

export interface ConsensusRingDatum {
  x: number;
  y: number;
  /** Per-point agreement max_c v_c over the T trees. */
  maxShare: number;
  color?: string;
  label: string;
}

export interface ForestConsensusRing2DProps {
  rings: ConsensusRingDatum[];
  visible?: boolean;
  /** Cap rendered rings so dense queues stay legible (styles stay exact). */
  ringCap?: number;
}

/**
 * Phase 53 (mechanism B, DoD #1) — forest voting variance dual-ring.
 *
 * Unanimous trees render a thin solid ring; once agreement drops below
 * 60% the ring doubles in thickness and splits into a double-dashed
 * pair (outer + inner), flagging the 52/48 split-vote zone. Two draw
 * calls max (solid batch + fuzzy batch).
 */
export function ForestConsensusRing2D({ rings, visible = true, ringCap = 500 }: ForestConsensusRing2DProps) {
  if (!visible || rings.length === 0) return null;
  const shown = rings.slice(0, Math.max(1, ringCap));
  const solid = shown.filter((r) => !isFuzzyConsensus(r.maxShare));
  const fuzzy = shown.filter((r) => isFuzzyConsensus(r.maxShare));
  const pathCircle = (r: ConsensusRingDatum, radius: number): string => {
    const k = 0.5523 * radius;
    const cx = r.x;
    const cy = wy(r.y);
    return (
      `M${(cx + radius).toFixed(3)} ${cy.toFixed(3)}` +
      `C${(cx + radius).toFixed(3)} ${(cy + k).toFixed(3)} ${(cx + k).toFixed(3)} ${(cy + radius).toFixed(3)} ${cx.toFixed(3)} ${(cy + radius).toFixed(3)}` +
      `C${(cx - k).toFixed(3)} ${(cy + radius).toFixed(3)} ${(cx - radius).toFixed(3)} ${(cy + k).toFixed(3)} ${(cx - radius).toFixed(3)} ${cy.toFixed(3)}` +
      `C${(cx - radius).toFixed(3)} ${(cy - k).toFixed(3)} ${(cx - k).toFixed(3)} ${(cy - radius).toFixed(3)} ${cx.toFixed(3)} ${(cy - radius).toFixed(3)}` +
      `C${(cx + k).toFixed(3)} ${(cy - radius).toFixed(3)} ${(cx + radius).toFixed(3)} ${(cy - k).toFixed(3)} ${(cx + radius).toFixed(3)} ${cy.toFixed(3)}Z`
    );
  };
  return (
    <g className="forest-consensus-ring-2d pointer-events-none" aria-label="Forest voting consensus rings">
      {solid.length > 0 && (
        <path
          d={solid.map((r) => pathCircle(r, RING_OUTER_RADIUS)).join('')}
          fill="none"
          stroke={CONSENSUS_SOLID_COLOR}
          strokeWidth={0.035}
          opacity={0.9}
        >
          <title>{`Unanimous trees (${solid.length} pts, agreement ≥ 60%)`}</title>
        </path>
      )}
      {fuzzy.map((r, i) => (
        <g key={i} opacity={0.95}>
          <path
            d={pathCircle(r, RING_OUTER_RADIUS)}
            fill="none"
            stroke={r.color ?? CONSENSUS_FUZZY_COLOR}
            strokeWidth={0.035 * consensusThickness(r.maxShare)}
            strokeDasharray="0.16 0.1"
          >
            <title>{`${r.label} · agreement ${(r.maxShare * 100).toFixed(1)}% (split vote)`}</title>
          </path>
          <path
            d={pathCircle(r, RING_OUTER_RADIUS * 0.62)}
            fill="none"
            stroke={r.color ?? CONSENSUS_FUZZY_COLOR}
            strokeWidth={0.025}
            strokeDasharray="0.1 0.12"
            opacity={0.8}
          />
        </g>
      ))}
    </g>
  );
}

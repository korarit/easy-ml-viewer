import { wy } from '../DecisionBoundaryContour2D';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';
import {
  LOGIT_CONFIDENT_COLOR,
  LOGIT_DECIDIVE_COLOR,
  LOGIT_DIFFUSE_COLOR,
  type LogitGapStatus,
} from './neuralPointIntel2dMath';

export interface LogitGapRingDatum {
  x: number;
  y: number;
  /** Logit margin Δz = z_(1) − z_(2) for this point. */
  gap: number;
  status: LogitGapStatus;
  color: string;
  label: string;
}

export interface SoftmaxLogitGapRing2DProps {
  rings: LogitGapRingDatum[];
  visible?: boolean;
  /** Cap rendered rings so dense queues stay legible (gaps stay exact). */
  ringCap?: number;
}

/**
 * Phase 55 (mechanism B, DoD #2) — softmax logit-gap dual-ring.
 *
 * Saturated winners (Δz > 3.0) earn a crisp solid ring; tempting ties
 * (Δz < 0.5) breathe through a neon diffuse halo warning the model is
 * torn between its top two classes; decisive mid-gaps keep a thin
 * neutral ring. Confident + decisive rings batch into two paths.
 */
export function SoftmaxLogitGapRing2D({ rings, visible = true, ringCap = 500 }: SoftmaxLogitGapRing2DProps) {
  if (!visible || rings.length === 0) return null;
  const shown = rings.slice(0, Math.max(1, ringCap));
  const confident = shown.filter((r) => r.status === 'confident');
  const diffuse = shown.filter((r) => r.status === 'diffuse');
  const decisive = shown.filter((r) => r.status === 'decisive');
  const circlePath = (list: LogitGapRingDatum[], radius: number): string => {
    const k = 0.5523 * radius;
    return list
      .map((r) => {
        const cx = r.x;
        const cy = wy(r.y);
        return (
          `M${(cx + radius).toFixed(3)} ${cy.toFixed(3)}` +
          `C${(cx + radius).toFixed(3)} ${(cy + k).toFixed(3)} ${(cx + k).toFixed(3)} ${(cy + radius).toFixed(3)} ${cx.toFixed(3)} ${(cy + radius).toFixed(3)}` +
          `C${(cx - k).toFixed(3)} ${(cy + radius).toFixed(3)} ${(cx - radius).toFixed(3)} ${(cy + k).toFixed(3)} ${(cx - radius).toFixed(3)} ${cy.toFixed(3)}` +
          `C${(cx - radius).toFixed(3)} ${(cy - k).toFixed(3)} ${(cx - k).toFixed(3)} ${(cy - radius).toFixed(3)} ${cx.toFixed(3)} ${(cy - radius).toFixed(3)}` +
          `C${(cx + k).toFixed(3)} ${(cy - radius).toFixed(3)} ${(cx + radius).toFixed(3)} ${(cy - k).toFixed(3)} ${(cx + radius).toFixed(3)} ${cy.toFixed(3)}Z`
        );
      })
      .join('');
  };
  return (
    <g className="logit-gap-ring-2d pointer-events-none" aria-label="Softmax logit-gap rings">
      {confident.map((r, i) => (
        <circle
          key={i}
          cx={r.x}
          cy={wy(r.y)}
          r={RING_OUTER_RADIUS}
          fill="none"
          stroke={r.color !== '' ? r.color : LOGIT_CONFIDENT_COLOR}
          strokeWidth={0.04}
          opacity={0.95}
        >
          <title>{`${r.label} · Δz=${r.gap.toPrecision(3)} (saturated confident)`}</title>
        </circle>
      ))}
      {decisive.length > 0 && (
        <path
          d={circlePath(decisive, RING_OUTER_RADIUS * 0.85)}
          fill="none"
          stroke={LOGIT_DECIDIVE_COLOR}
          strokeWidth={0.03}
          opacity={0.8}
        >
          <title>{`Decisive gaps (${decisive.length} pts, 0.5 ≤ Δz ≤ 3.0)`}</title>
        </path>
      )}
      {diffuse.length > 0 && (
        <path
          d={circlePath(diffuse, RING_OUTER_RADIUS * 1.25)}
          fill="none"
          stroke={LOGIT_DIFFUSE_COLOR}
          strokeWidth={0.05}
          strokeDasharray="0.12 0.1"
          opacity={0.95}
        >
          <title>{`Tempting ties (${diffuse.length} pts, Δz < 0.5) — model torn between top classes`}</title>
          <animate attributeName="opacity" values="0.95;0.35;0.95" dur="1.8s" repeatCount="indefinite" />
        </path>
      )}
    </g>
  );
}

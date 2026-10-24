import { wy } from '../DecisionBoundaryContour2D';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';
import {
  MARGIN_CRITICAL_COLOR,
  MARGIN_NEGATIVE_COLOR,
  type MarginStatus,
} from './boostingPointIntel2dMath';

export interface MarginRingDatum {
  x: number;
  y: number;
  /** Functional margin M = y·F(x) for this point. */
  margin: number;
  status: MarginStatus;
  color: string;
  label: string;
}

export interface FunctionalMarginRing2DProps {
  rings: MarginRingDatum[];
  visible?: boolean;
  /** Cap rendered rings so dense queues stay legible (margins stay exact). */
  ringCap?: number;
}

/**
 * Phase 54 (mechanism B, DoD #2) — functional margin dual-ring.
 *
 * Safe points (M > 1.5) render a thin solid class-color ring; critical
 * points (0 ≤ M ≤ 1.5) pulse orange near the flip; negative points
 * (M < 0) carry a red ring plus a cross — the exact rows boosting must
 * correct next round. Three batched paths + one cross path hold 60 FPS.
 */
export function FunctionalMarginRing2D({ rings, visible = true, ringCap = 500 }: FunctionalMarginRing2DProps) {
  if (!visible || rings.length === 0) return null;
  const shown = rings.slice(0, Math.max(1, ringCap));
  const safe = shown.filter((r) => r.status === 'safe');
  const critical = shown.filter((r) => r.status === 'critical');
  const negative = shown.filter((r) => r.status === 'negative');
  const circlePath = (r: MarginRingDatum, radius: number): string => {
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
  const crossPath = negative
    .map((r) => {
      const a = 0.09;
      return `M${(r.x - a).toFixed(3)} ${(wy(r.y) - a).toFixed(3)}L${(r.x + a).toFixed(3)} ${(wy(r.y) + a).toFixed(3)}M${(r.x - a).toFixed(3)} ${(wy(r.y) + a).toFixed(3)}L${(r.x + a).toFixed(3)} ${(wy(r.y) - a).toFixed(3)}`;
    })
    .join('');
  return (
    <g className="margin-ring-2d pointer-events-none" aria-label="Functional margin rings">
      {safe.map((r, i) => (
        <circle
          key={i}
          cx={r.x}
          cy={wy(r.y)}
          r={RING_OUTER_RADIUS}
          fill="none"
          stroke={r.color}
          strokeWidth={0.035}
          opacity={0.9}
        >
          <title>{`${r.label} · M=${r.margin.toPrecision(3)} (safe)`}</title>
        </circle>
      ))}
      {critical.length > 0 && (
        <path
          d={critical.map((r) => circlePath(r, RING_OUTER_RADIUS)).join('')}
          fill="none"
          stroke={MARGIN_CRITICAL_COLOR}
          strokeWidth={0.045}
          strokeDasharray="0.16 0.1"
          opacity={0.95}
        >
          <title>{`Critical margin (${critical.length} pts, 0 ≤ M ≤ 1.5) — near the flip`}</title>
          <animate attributeName="opacity" values="0.95;0.4;0.95" dur="2s" repeatCount="indefinite" />
        </path>
      )}
      {negative.length > 0 && (
        <path
          d={negative.map((r) => circlePath(r, RING_OUTER_RADIUS * 1.25)).join('')}
          fill="none"
          stroke={MARGIN_NEGATIVE_COLOR}
          strokeWidth={0.05}
          opacity={0.95}
        >
          <title>{`Negative margin (${negative.length} pts, M < 0) — boosting target`}</title>
        </path>
      )}
      {crossPath && (
        <path
          d={crossPath}
          fill="none"
          stroke={MARGIN_NEGATIVE_COLOR}
          strokeWidth={0.04}
          strokeLinecap="round"
          opacity={0.95}
        />
      )}
    </g>
  );
}

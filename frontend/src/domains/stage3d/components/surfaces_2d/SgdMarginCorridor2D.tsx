import { wy } from '../DecisionBoundaryContour2D';
import type { WorldSeg } from './linear2dMath';

export interface SgdMarginCorridor2DProps {
  posSegment: WorldSeg | null;
  negSegment: WorldSeg | null;
  /** Exact corridor width W = 2/||w|| in raw units (badge only). */
  corridorWidth: number | null;
  marginLevel?: number;
  opacity?: number;
  visible?: boolean;
}

/**
 * Phase 42 — SGD / Ridge margin corridor (z = ±marginLevel parallels).
 *
 * The translucent band is drawn only when *both* parallels cross the data
 * rect; lone parallels still render as dashed guides. When the weights are
 * tiny the parallels can miss the rect entirely — then only the width badge
 * survives (honest fallback, no invented geometry).
 */
export function SgdMarginCorridor2D({
  posSegment,
  negSegment,
  corridorWidth,
  marginLevel = 1,
  opacity = 0.25,
  visible = true,
}: SgdMarginCorridor2DProps) {
  if (!visible || (!posSegment && !negSegment)) return null;
  const band =
    posSegment && negSegment
      ? `M${posSegment[0].toFixed(3)} ${wy(posSegment[1]).toFixed(3)}` +
        `L${posSegment[2].toFixed(3)} ${wy(posSegment[3]).toFixed(3)}` +
        `L${negSegment[2].toFixed(3)} ${wy(negSegment[3]).toFixed(3)}` +
        `L${negSegment[0].toFixed(3)} ${wy(negSegment[1]).toFixed(3)}Z`
      : null;

  const dashed = (seg: WorldSeg, label: string) => (
    <line
      x1={seg[0]}
      y1={wy(seg[1])}
      x2={seg[2]}
      y2={wy(seg[3])}
      stroke="#7dd3fc"
      strokeWidth={0.045}
      strokeDasharray="0.3 0.14"
      opacity={0.9}
      strokeLinecap="round"
    >
      <title>{label}</title>
    </line>
  );

  return (
    <g className="sgd-margin-corridor-2d pointer-events-none" aria-label="Margin corridor">
      {band && <path d={band} fill="#7dd3fc" opacity={Math.min(0.8, Math.max(0.1, opacity))} />}
      {posSegment && dashed(posSegment, `Margin boundary (z = +${marginLevel})`)}
      {negSegment && dashed(negSegment, `Margin boundary (z = −${marginLevel})`)}
      {band && corridorWidth !== null && Number.isFinite(corridorWidth) && (
        <title>{`Margin corridor width W = 2/‖w‖ = ${corridorWidth.toPrecision(4)} (raw units)`}</title>
      )}
    </g>
  );
}

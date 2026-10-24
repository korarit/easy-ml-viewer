import { wy } from '../DecisionBoundaryContour2D';
import { GRAD_DOWN_COLOR, GRAD_UP_COLOR } from './boostingPointIntel2dMath';

export interface PseudoResidualSpoke {
  x: number;
  y: number;
  /** Normalized spoke length in world units (peak |r| → 1.5). */
  length: number;
  /** +1 needs a push up · −1 needs a pull down. */
  direction: 1 | -1;
  /** Raw |r| for the hard-sample labels. */
  magnitude?: number;
}

export interface PseudoResidualSpoke2DProps {
  spokes: PseudoResidualSpoke[];
  visible?: boolean;
  /** Cap |r| labels so dense queues stay legible (spokes uncut). */
  labelCap?: number;
}

/**
 * Phase 54 (mechanism C, DoD #3) — pseudo-residual gradient spokes.
 *
 * Each spoke stretches from the queued point along the loss gradient
 * with length ∝ |r|: the longest spokes are the hard samples pulling
 * the ensemble hardest this round. Up/down batches share one path
 * each; the top-K magnitudes earn labels (≤ 3 draw calls + labels).
 */
export function PseudoResidualSpoke2D({ spokes, visible = true, labelCap = 20 }: PseudoResidualSpoke2DProps) {
  if (!visible || spokes.length === 0) return null;
  const up = spokes.filter((s) => s.direction > 0 && s.length > 1e-9);
  const down = spokes.filter((s) => s.direction < 0 && s.length > 1e-9);
  const pathFor = (list: PseudoResidualSpoke[], dir: 1 | -1): string =>
    list
      .map((s) => {
        const yEnd = dir > 0 ? s.y + s.length : s.y - s.length;
        return `M${s.x.toFixed(3)} ${wy(s.y).toFixed(3)}L${s.x.toFixed(3)} ${wy(yEnd).toFixed(3)}`;
      })
      .join('');
  const labeled = [...spokes]
    .filter((s) => s.magnitude !== undefined && Number.isFinite(s.magnitude) && s.length > 1e-9)
    .sort((a, b) => (b.magnitude as number) - (a.magnitude as number))
    .slice(0, Math.max(0, labelCap));
  return (
    <g className="pseudo-residual-spoke-2d pointer-events-none" aria-label="Pseudo-residual spokes">
      <defs>
        <marker id="grad-spoke-up" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={GRAD_UP_COLOR} />
        </marker>
        <marker id="grad-spoke-down" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={GRAD_DOWN_COLOR} />
        </marker>
      </defs>
      {up.length > 0 && (
        <path
          d={pathFor(up, 1)}
          fill="none"
          stroke={GRAD_UP_COLOR}
          strokeWidth={0.035}
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#grad-spoke-up)"
        >
          <title>{`Positive pseudo-residuals (${up.length} spokes, push up)`}</title>
        </path>
      )}
      {down.length > 0 && (
        <path
          d={pathFor(down, -1)}
          fill="none"
          stroke={GRAD_DOWN_COLOR}
          strokeWidth={0.035}
          strokeDasharray="0.18 0.1"
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#grad-spoke-down)"
        >
          <title>{`Negative pseudo-residuals (${down.length} spokes, pull down)`}</title>
        </path>
      )}
      {labeled.map((s, i) => (
        <text
          key={i}
          x={s.x + 0.1}
          y={wy(s.direction > 0 ? s.y + s.length : s.y - s.length) - 0.08}
          fontSize={0.26}
          fill="#fda4af"
          textAnchor="start"
          opacity={0.95}
        >
          {`|r|=${(s.magnitude as number).toPrecision(3)}`}
        </text>
      ))}
    </g>
  );
}

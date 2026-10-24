import { wy } from '../DecisionBoundaryContour2D';
import { LOSS_GRAD_COLOR } from './neuralPointIntel2dMath';

export interface LossGradientSpoke {
  x: number;
  y: number;
  /** Unit input-loss-gradient direction (ux, uy). */
  ux: number;
  uy: number;
  /** Normalized spoke length in world units (peak ‖g‖ → 1.5). */
  length: number;
  /** Raw ‖g‖ for the fragility labels. */
  magnitude?: number;
}

export interface LossGradientSpoke2DProps {
  spokes: LossGradientSpoke[];
  visible?: boolean;
  /** Cap ‖g‖ labels so dense queues stay legible (spokes uncut). */
  labelCap?: number;
}

/**
 * Phase 55 (mechanism C) — input loss-gradient spokes.
 *
 * Each spoke leaves the queued point along ∇ₓL: nudge the inputs that
 * way and the loss climbs fastest (local sensitivity / adversarial
 * fragility). The longest spokes are the samples the network is most
 * brittle about. One batched path + arrowhead + top-K labels.
 */
export function LossGradientSpoke2D({ spokes, visible = true, labelCap = 20 }: LossGradientSpoke2DProps) {
  if (!visible || spokes.length === 0) return null;
  const live = spokes.filter((s) => s.length > 1e-9 && Math.hypot(s.ux, s.uy) > 1e-9);
  if (live.length === 0) return null;
  const d = live
    .map((s) => {
      const ex = s.x + s.ux * s.length;
      const ey = s.y + s.uy * s.length;
      return `M${s.x.toFixed(3)} ${wy(s.y).toFixed(3)}L${ex.toFixed(3)} ${wy(ey).toFixed(3)}`;
    })
    .join('');
  const labeled = [...live]
    .filter((s) => s.magnitude !== undefined && Number.isFinite(s.magnitude))
    .sort((a, b) => (b.magnitude as number) - (a.magnitude as number))
    .slice(0, Math.max(0, labelCap));
  return (
    <g className="loss-grad-spoke-2d pointer-events-none" aria-label="Input loss-gradient spokes">
      <defs>
        <marker id="loss-grad-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={LOSS_GRAD_COLOR} />
        </marker>
      </defs>
      <path
        d={d}
        fill="none"
        stroke={LOSS_GRAD_COLOR}
        strokeWidth={0.035}
        opacity={0.9}
        strokeLinecap="round"
        markerEnd="url(#loss-grad-arrow)"
      >
        <title>{`Adversarial spokes (${live.length} pts): ∇ₓL direction, loss climbs fastest this way`}</title>
      </path>
      {labeled.map((s, i) => (
        <text
          key={i}
          x={s.x + s.ux * s.length + 0.1}
          y={wy(s.y + s.uy * s.length) - 0.08}
          fontSize={0.26}
          fill={LOSS_GRAD_COLOR}
          textAnchor="start"
          opacity={0.95}
        >
          {`‖g‖=${(s.magnitude as number).toPrecision(3)}`}
        </text>
      ))}
    </g>
  );
}

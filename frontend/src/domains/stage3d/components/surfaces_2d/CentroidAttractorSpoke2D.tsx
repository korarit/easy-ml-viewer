import { wy } from '../DecisionBoundaryContour2D';
import { SPOKE_PRED_COLOR, SPOKE_TRUE_COLOR, bayesSpokeToSvg } from './probabilisticPointIntel2dMath';

export interface CentroidSpoke {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** 'true' → green (actual centroid) · 'pred' → orange (claimed centroid). */
  kind: 'true' | 'pred';
  /** LLR = ℓ(true) − ℓ(pred); shown on the true-side label. */
  llr?: number;
}

export interface CentroidAttractorSpoke2DProps {
  /** Dual spokes for mispredicted frames (world coords). */
  spokes: CentroidSpoke[];
  visible?: boolean;
  /** Cap LLR labels so dense queues stay legible (spokes themselves uncut). */
  labelCap?: number;
}

/**
 * Phase 52 (mechanism C, DoD #2) — centroid attractor spokes.
 *
 * Each mispredicted point grows two dashed spokes: green to the true class
 * centroid, orange to the class the model claimed. The LLR readout on the
 * true spoke quantifies the Bayes paradox (negative LLR = the model had
 * statistical reason to pick the farther centroid). Two batched paths.
 */
export function CentroidAttractorSpoke2D({ spokes, visible = true, labelCap = 20 }: CentroidAttractorSpoke2DProps) {
  if (!visible || spokes.length === 0) return null;
  const toTrue = spokes.filter((s) => s.kind === 'true');
  const toPred = spokes.filter((s) => s.kind === 'pred');
  const labeled = toTrue.filter((s) => s.llr !== undefined && Number.isFinite(s.llr)).slice(0, Math.max(0, labelCap));
  return (
    <g className="centroid-spoke-2d pointer-events-none" aria-label="Centroid attractor spokes">
      <defs>
        <marker id="bayes-true-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={SPOKE_TRUE_COLOR} />
        </marker>
        <marker id="bayes-pred-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={SPOKE_PRED_COLOR} />
        </marker>
      </defs>
      {toTrue.length > 0 && (
        <path
          d={toTrue.map(bayesSpokeToSvg).join('')}
          fill="none"
          stroke={SPOKE_TRUE_COLOR}
          strokeWidth={0.035}
          strokeDasharray="0.18 0.1"
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#bayes-true-arrow)"
        >
          <title>Spokes to the true class centroid μ_true (green)</title>
        </path>
      )}
      {toPred.length > 0 && (
        <path
          d={toPred.map(bayesSpokeToSvg).join('')}
          fill="none"
          stroke={SPOKE_PRED_COLOR}
          strokeWidth={0.035}
          strokeDasharray="0.18 0.1"
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#bayes-pred-arrow)"
        >
          <title>Spokes to the claimed class centroid μ_pred (orange)</title>
        </path>
      )}
      {labeled.map((s, i) => (
        <text
          key={i}
          x={(s.x1 + s.x2) / 2}
          y={wy((s.y1 + s.y2) / 2) - 0.08}
          fontSize={0.26}
          fill="#a5b4fc"
          textAnchor="middle"
          opacity={0.95}
        >
          {`LLR=${s.llr!.toPrecision(3)}`}
        </text>
      ))}
    </g>
  );
}

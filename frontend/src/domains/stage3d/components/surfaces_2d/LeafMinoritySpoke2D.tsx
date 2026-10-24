import { LEAF_MINORITY_COLOR, treeSpokeToSvg } from './treesPointIntel2dMath';

export interface LeafMinoritySpoke {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Leaf impurity (Gini) at the stranded point — shown on labels. */
  impurity?: number;
}

export interface LeafMinoritySpoke2DProps {
  /** Spokes from minority points to their leaf-box centre (world coords). */
  spokes: LeafMinoritySpoke[];
  visible?: boolean;
  /** Cap impurity labels so dense queues stay legible (spokes uncut). */
  labelCap?: number;
}

/**
 * Phase 53 (mechanism C, DoD #3) — leaf minority spokes.
 *
 * Points stranded as the minority class inside an impure leaf grow a
 * dashed spoke to the leaf-box centre, marking the outlier left behind
 * in the box. Regressor stepwise residuals reuse the same channel as
 * vertical stems (see the orchestrator). One batched path.
 */
export function LeafMinoritySpoke2D({ spokes, visible = true, labelCap = 20 }: LeafMinoritySpoke2DProps) {
  if (!visible || spokes.length === 0) return null;
  const labeled = spokes
    .filter((s) => s.impurity !== undefined && Number.isFinite(s.impurity))
    .slice(0, Math.max(0, labelCap));
  return (
    <g className="leaf-minority-spoke-2d pointer-events-none" aria-label="Leaf minority spokes">
      <defs>
        <marker id="leaf-minority-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={LEAF_MINORITY_COLOR} />
        </marker>
      </defs>
      <path
        d={spokes.map(treeSpokeToSvg).join('')}
        fill="none"
        stroke={LEAF_MINORITY_COLOR}
        strokeWidth={0.035}
        strokeDasharray="0.18 0.1"
        opacity={0.9}
        strokeLinecap="round"
        markerEnd="url(#leaf-minority-arrow)"
      >
        <title>{`Minority strands in impure leaves (${spokes.length} spokes → leaf centres)`}</title>
      </path>
      {labeled.map((s, i) => (
        <text
          key={i}
          x={(s.x1 + s.x2) / 2}
          y={(s.y1 + s.y2) / 2 - 0.08}
          fontSize={0.26}
          fill="#fda4af"
          textAnchor="middle"
          opacity={0.95}
        >
          {`G=${s.impurity!.toFixed(2)}`}
        </text>
      ))}
    </g>
  );
}

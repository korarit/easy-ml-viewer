import { wy } from '../DecisionBoundaryContour2D';
import { SVR_SLACK_XI_COLOR, SVR_SLACK_XI_STAR_COLOR } from './distancePointIntel2dMath';

export interface SvrSlackSpokeDatum {
  x: number;
  y: number;
  /** Normalized stem length in world units (peak slack → 1.5). */
  length: number;
  /** −1 under-predict (ξ) · +1 over-predict (ξ*). */
  side: -1 | 1;
  /** Raw slack magnitude for the tooltip. */
  slack: number;
}

export interface SvrSlackSpoke2DProps {
  /** Only points outside the ε-tube belong here (DoD #2). */
  spokes: SvrSlackSpokeDatum[];
  visible?: boolean;
}

/**
 * Phase 51 (mechanism C, regression) — SVR ε-tube slack spokes.
 *
 * A spoke is drawn *only* for points with |y − ŷ| > ε; its length reflects
 * the slack variable ξ / ξ* directly. Under- (ξ) and over-tube (ξ*) spokes
 * batch into one path each (≤ 4 draw calls total).
 */
export function SvrSlackSpoke2D({ spokes, visible = true }: SvrSlackSpoke2DProps) {
  if (!visible || spokes.length === 0) return null;
  const under = spokes.filter((s) => s.side < 0 && s.length > 1e-9);
  const over = spokes.filter((s) => s.side > 0 && s.length > 1e-9);
  const pathFor = (list: SvrSlackSpokeDatum[], dir: -1 | 1): string =>
    list
      .map((s) => {
        const yEnd = dir > 0 ? s.y + s.length : s.y - s.length;
        return `M${s.x.toFixed(3)} ${wy(s.y).toFixed(3)}L${s.x.toFixed(3)} ${wy(yEnd).toFixed(3)}`;
      })
      .join('');
  return (
    <g className="svr-slack-spoke-2d pointer-events-none" aria-label="SVR epsilon slack spokes">
      <defs>
        <marker id="svr-xi" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={SVR_SLACK_XI_COLOR} />
        </marker>
        <marker id="svr-xi-star" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill={SVR_SLACK_XI_STAR_COLOR} />
        </marker>
      </defs>
      {under.length > 0 && (
        <path
          d={pathFor(under, -1)}
          fill="none"
          stroke={SVR_SLACK_XI_COLOR}
          strokeWidth={0.035}
          strokeDasharray="0.18 0.1"
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#svr-xi)"
        >
          <title>Under-tube spokes (ξ = y − ŷ − ε)</title>
        </path>
      )}
      {over.length > 0 && (
        <path
          d={pathFor(over, 1)}
          fill="none"
          stroke={SVR_SLACK_XI_STAR_COLOR}
          strokeWidth={0.035}
          opacity={0.9}
          strokeLinecap="round"
          markerEnd="url(#svr-xi-star)"
        >
          <title>Over-tube spokes (ξ* = ŷ − y − ε)</title>
        </path>
      )}
    </g>
  );
}

import { wy } from '../DecisionBoundaryContour2D';

export interface ResidualSpoke2DProps {
  /** Spoke segments as [x1, y1, x2, y2] in world coords. */
  segments: Array<[number, number, number, number]>;
  visible?: boolean;
}

/**
 * Phase 49 (mechanism C) — Residual spokes for the Error Lens.
 *
 * One batched dashed path (red/orange) with a shared arrowhead marker, so
 * even a fully-mispredicted queue stays at 2 draw calls.
 */
export function ResidualSpoke2D({ segments, visible = true }: ResidualSpoke2DProps) {
  if (!visible || segments.length === 0) return null;
  const d = segments
    .map(([x1, y1, x2, y2]) => `M${x1.toFixed(3)} ${wy(y1).toFixed(3)}L${x2.toFixed(3)} ${wy(y2).toFixed(3)}`)
    .join('');
  return (
    <g className="residual-spoke-2d pointer-events-none" aria-label="Residual spokes">
      <defs>
        <marker id="pi-spoke-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill="#fb923c" />
        </marker>
      </defs>
      <path
        d={d}
        fill="none"
        stroke="#fb923c"
        strokeWidth={0.035}
        strokeDasharray="0.18 0.1"
        opacity={0.9}
        strokeLinecap="round"
        markerEnd="url(#pi-spoke-arrow)"
      />
    </g>
  );
}

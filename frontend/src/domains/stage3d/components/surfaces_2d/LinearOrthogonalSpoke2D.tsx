import { wy } from '../DecisionBoundaryContour2D';
import type { OrthogonalSpoke } from './linearPointIntel2dMath';
import { segToSvg } from './linearPointIntel2dMath';

export interface LinearOrthogonalSpoke2DProps {
  /** Orthogonal feet for mispredicted frames (world coords + d_perp). */
  spokes: OrthogonalSpoke[];
  visible?: boolean;
  /** Cap d_perp labels so dense queues stay legible (spokes themselves uncut). */
  labelCap?: number;
}

/**
 * Phase 50 (mechanism C, classification) — orthogonal projection spokes.
 *
 * Each spoke drops perpendicularly from a mispredicted point onto the exact
 * decision line `w·x + b = 0`. One batched dashed path holds all spokes
 * (2 draw calls); only the first `labelCap` carry a `d⊥` readout.
 */
export function LinearOrthogonalSpoke2D({ spokes, visible = true, labelCap = 50 }: LinearOrthogonalSpoke2DProps) {
  if (!visible || spokes.length === 0) return null;
  const d = spokes.map(segToSvg).join('');
  const labeled = spokes.slice(0, Math.max(0, labelCap));
  return (
    <g className="linear-spoke-2d pointer-events-none" aria-label="Orthogonal projection spokes">
      <defs>
        <marker id="li-ortho-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0L6,3L0,6Z" fill="#818cf8" />
        </marker>
      </defs>
      <path
        d={d}
        fill="none"
        stroke="#818cf8"
        strokeWidth={0.035}
        strokeDasharray="0.18 0.1"
        opacity={0.9}
        strokeLinecap="round"
        markerEnd="url(#li-ortho-arrow)"
      />
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
          {`d⊥=${s.dPerp.toPrecision(3)}`}
        </text>
      ))}
    </g>
  );
}

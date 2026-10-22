import { wy } from '../DecisionBoundaryContour2D';
import type { WorldSeg } from './linear2dMath';

export interface LinearBoundary2DProps {
  /** Pre-clipped world-coord segment [x1, y1, x2, y2]; null renders nothing. */
  segment: WorldSeg | null;
  /** Unit ascent direction (w1, w2)/||w|| in a raw-isotropic sense. */
  normal?: [number, number];
  label?: string;
  showNormalVector?: boolean;
  visible?: boolean;
}

/**
 * Phase 42 — exact analytic decision line for linear families.
 *
 * Unlike the sampled Marching-Squares boundary, this segment satisfies
 * w1·x1 + w2·x2 + b_eff = t by construction (< 1e-4). The normal arrow
 * points along the direction of steepest ascent of the linear score.
 */
export function LinearBoundary2D({
  segment,
  normal = [0, 0],
  label = 'Exact linear boundary',
  showNormalVector = true,
  visible = true,
}: LinearBoundary2DProps) {
  if (!visible || !segment) return null;
  const [x1, y1, x2, y2] = segment;
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  const nLen = Math.hypot(normal[0], normal[1]);
  const arrowLen = 0.9; // world units
  const hasNormal = showNormalVector && nLen > 1e-9;
  const tx = midX + (hasNormal ? (normal[0] / nLen) * arrowLen : 0);
  const ty = midY + (hasNormal ? (normal[1] / nLen) * arrowLen : 0);

  return (
    <g className="linear-boundary-2d pointer-events-none" aria-label={label}>
      <line
        x1={x1}
        y1={wy(y1)}
        x2={x2}
        y2={wy(y2)}
        stroke="#ffffff"
        strokeWidth={0.09}
        strokeLinecap="round"
        opacity={0.95}
      >
        <title>{label}</title>
      </line>
      {hasNormal && (
        <line
          x1={midX}
          y1={wy(midY)}
          x2={tx}
          y2={wy(ty)}
          stroke="var(--ml-primary)"
          strokeWidth={0.07}
          markerEnd="url(#arrowhead-cyan)"
        >
          <title>Weight normal vector (direction of steepest ascent)</title>
        </line>
      )}
    </g>
  );
}

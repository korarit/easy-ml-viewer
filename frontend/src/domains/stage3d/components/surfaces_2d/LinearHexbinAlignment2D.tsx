import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { hexPath } from './pointIntelligence2dMath';
import type { WorldLine } from './linearPointIntel2dMath';
import { hexbinBoundaryFlags } from './linearPointIntel2dMath';

export interface LinearHexbinAlignment2DProps {
  cells: HexbinCell2D[];
  line: WorldLine;
  visible?: boolean;
}

/**
 * Phase 50 (mechanism D) — linear-partitioned hexbin alignment.
 *
 * Hexbins crossed by the exact decision line get a white dashed outline
 * (`boundary_intersected`), so the eye can trace where the plane slices the
 * density field. One batched path regardless of bin count.
 */
export function LinearHexbinAlignment2D({ cells, line, visible = true }: LinearHexbinAlignment2DProps) {
  if (!visible || cells.length === 0 || line.degenerate) return null;
  const flags = hexbinBoundaryFlags(cells, line);
  const crossed = cells.filter((_, i) => flags[i]);
  if (crossed.length === 0) return null;
  return (
    <g className="linear-hexbin-align-2d pointer-events-none" aria-label="Boundary-intersected hexbins">
      <path
        d={crossed.map((c) => hexPath(c.centerX, c.centerY, c.radius)).join('')}
        fill="none"
        stroke="#ffffff"
        strokeWidth={0.035}
        strokeDasharray="0.2 0.12"
        opacity={0.85}
      >
        <title>{`${crossed.length} hexbins intersect the decision line`}</title>
      </path>
    </g>
  );
}

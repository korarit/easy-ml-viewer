/** Phase 45 — shared math/helpers for tree & ensemble 2D overlays. */
import { wy } from '../DecisionBoundaryContour2D';
import type { LeafBox2D, TreeCutSegment2D } from '../../../projection3d/types/projection.types';

/**
 * Batch cut segments by axis into two SVG path strings (E45.1).
 * A 600-segment forest overlay renders as exactly 2 draw calls.
 */
export function batchCutPaths(segments: TreeCutSegment2D[]): {
  vertical: string;
  horizontal: string;
} {
  let vertical = '';
  let horizontal = '';
  for (const s of segments) {
    const [x1, y1, x2, y2] = s.segmentWorld;
    const d =
      `M${x1.toFixed(3)} ${wy(y1).toFixed(3)}` +
      `L${x2.toFixed(3)} ${wy(y2).toFixed(3)}`;
    if (s.axis === 'x') vertical += d;
    else horizontal += d;
  }
  return { vertical, horizontal };
}

/** World box [xmin, xmax, ymin, ymax] → SVG rect attributes (Y flipped). */
export function boxToRect(boxWorld: LeafBox2D['boxWorld']): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  const [xmin, xmax, ymin, ymax] = boxWorld;
  return { x: xmin, y: wy(ymax), width: Math.max(xmax - xmin, 1e-6), height: Math.max(ymax - ymin, 1e-6) };
}

/** First leaf (in export order) containing the world point, else null. */
export function findLeafAt(
  leaves: LeafBox2D[],
  wx: number,
  wyv: number,
): LeafBox2D | null {
  for (const l of leaves) {
    const [xmin, xmax, ymin, ymax] = l.boxWorld;
    if (wx >= xmin && wx <= xmax && wyv >= ymin && wyv <= ymax) return l;
  }
  return null;
}

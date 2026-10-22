/** Phase 46 — shared math/helpers for boosting heavyweight 2D overlays. */
import { wy } from '../DecisionBoundaryContour2D';
import type { BoostLeafBox2D, ObliviousCell2D } from '../../../projection3d/types/projection.types';

/** Max arrow length in world units — clamps unbounded log-odds SHAP magnitudes. */
export const SHAP_ARROW_MAX_LEN = 2.5;
/** Min visible arrow length so near-zero pushes still render a dot. */
export const SHAP_ARROW_MIN_LEN = 0.15;
/** World-units per unit |phi| before clamping. */
export const SHAP_ARROW_SCALE = 0.5;

/**
 * Scale a raw (phiX, phiY) push vector to a world-space arrow delta.
 * Length = clamp(|phi| * SCALE, MIN, MAX) along the unit push direction;
 * zero vectors collapse to a dot.
 */
export function scaleShapVector(
  phiX: number,
  phiY: number,
): { dx: number; dy: number; magnitude: number } {
  const fx = Number.isFinite(phiX) ? phiX : 0;
  const fy = Number.isFinite(phiY) ? phiY : 0;
  const magnitude = Math.hypot(fx, fy);
  if (magnitude < 1e-12) return { dx: 0, dy: 0, magnitude: 0 };
  const len = Math.min(SHAP_ARROW_MAX_LEN, Math.max(SHAP_ARROW_MIN_LEN, magnitude * SHAP_ARROW_SCALE));
  return { dx: (fx / magnitude) * len, dy: (fy / magnitude) * len, magnitude };
}

/** World box [xmin, xmax, ymin, ymax] → SVG rect attributes (Y flipped). */
export function boostBoxToRect(boxWorld: BoostLeafBox2D['boxWorld']): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  const [xmin, xmax, ymin, ymax] = boxWorld;
  return { x: xmin, y: wy(ymax), width: Math.max(xmax - xmin, 1e-6), height: Math.max(ymax - ymin, 1e-6) };
}

/** First box (export order) containing the world point, else null. */
export function findBoostLeafAt<T extends { boxWorld: [number, number, number, number] }>(
  boxes: T[],
  wx: number,
  wyv: number,
): T | null {
  for (const b of boxes) {
    const [xmin, xmax, ymin, ymax] = b.boxWorld;
    if (wx >= xmin && wx <= xmax && wyv >= ymin && wyv <= ymax) return b;
  }
  return null;
}

/** Leaf with the largest |leafValue| — the LightGBM best-first highlight. */
export function findStrongestLeaf(leaves: BoostLeafBox2D[]): BoostLeafBox2D | null {
  let best: BoostLeafBox2D | null = null;
  let bestAbs = -Infinity;
  for (const l of leaves) {
    const a = Math.abs(l.leafValue);
    if (a > bestAbs) {
      bestAbs = a;
      best = l;
    }
  }
  return best;
}

/**
 * Bin index rect highlight for the HistGBM lattice: neighboring world line
 * positions bound the active (binX, binY) cell; the outer data rect bounds
 * the edges.
 */
export function binCellRect(
  linesXWorld: number[],
  linesYWorld: number[],
  binX: number,
  binY: number,
): { x: number; y: number; width: number; height: number } | null {
  if (linesXWorld.length === 0 || linesYWorld.length === 0) return null;
  const xs = [...linesXWorld].sort((a, b) => a - b);
  const ys = [...linesYWorld].sort((a, b) => a - b);
  // binX counts edges <= value (searchsorted left): cell spans edges[binX-1..binX].
  const x0 = binX <= 0 ? -5 : (xs[binX - 1] ?? -5);
  const x1 = binX >= xs.length ? 5 : (xs[binX] ?? 5);
  const y0 = binY <= 0 ? -5 : (ys[binY - 1] ?? -5);
  const y1 = binY >= ys.length ? 5 : (ys[binY] ?? 5);
  if (!(x1 > x0 && y1 > y0)) return null;
  return { x: x0, y: wy(y1), width: x1 - x0, height: y1 - y0 };
}

/** Oblivious cells already carry world boxes — thin alias for symmetry. */
export function cellToRect(cell: ObliviousCell2D): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  return boostBoxToRect(cell.boxWorld);
}

/**
 * Phase 21 — pure math helpers for the SGD tilted hyperplane + margin
 * corridor (±1 score planes, ±ε tube for regressors).
 *
 * Kept dependency-free (no React / Three imports) so the analytical plane
 * formula, the geometric-margin conversion and the corridor status are
 * unit-testable and reusable by the inspector + WebGL surface.
 *
 * Plane convention: z(x, y) = w_x * x + w_y * y + interceptOffset,
 * where interceptOffset = b_k + Σ_{j ∉ {x,y}} w_{k,j} * x_j.
 *
 * Margin convention (blueprint §2.1): d_geom(x) = f(x) / ||w||₂ and the
 * corridor planes f(x) = ±m sit at ±m / ||w||₂ geometric units out.
 * |f(x)| < m  →  INSIDE MARGIN CORRIDOR.
 */

export const SGD_W_NORM_FLOOR = 1e-12;

export type SgdMarginStatus =
  | 'INSIDE MARGIN CORRIDOR'
  | 'SAFE MARGIN (> +1)'
  | 'SAFE MARGIN (< -1)'
  | 'INSIDE EPSILON TUBE'
  | 'OUTSIDE EPSILON TUBE';

/** Signed geometric distance of a raw score from the decision plane. */
export function geometricDistance(rawScore: number, wNorm: number): number {
  const norm = Number.isFinite(wNorm) && wNorm > SGD_W_NORM_FLOOR ? wNorm : SGD_W_NORM_FLOOR;
  if (!Number.isFinite(rawScore)) return 0;
  return rawScore / norm;
}

/** Corridor membership test: |f(x)| < marginWidth. */
export function isInsideCorridor(rawScore: number, marginWidth: number): boolean {
  if (!Number.isFinite(rawScore) || !Number.isFinite(marginWidth)) return false;
  return Math.abs(rawScore) < marginWidth;
}

/** HUD status label for a raw score given the corridor half-width. */
export function marginStatusFor(rawScore: number, marginWidth: number, isRegressor: boolean): SgdMarginStatus {
  const inside = isInsideCorridor(rawScore, marginWidth);
  if (isRegressor) return inside ? 'INSIDE EPSILON TUBE' : 'OUTSIDE EPSILON TUBE';
  if (inside) return 'INSIDE MARGIN CORRIDOR';
  return rawScore >= 0 ? 'SAFE MARGIN (> +1)' : 'SAFE MARGIN (< -1)';
}

/** Calibrated pseudo-confidence σ(f) for hinge/perceptron HUD coloring only. */
export function pseudoConfidence(rawScore: number): number {
  if (!Number.isFinite(rawScore)) return 0.5;
  if (rawScore >= 0) return 1 / (1 + Math.exp(-rawScore));
  const e = Math.exp(rawScore);
  return e / (1 + e);
}

/**
 * GPU phase-shift uniform for off-axis What-If sliders.
 */
export function computeSgdPhaseShift(
  weights: number[],
  intercept: number,
  features: number[],
  axisX: number,
  axisY: number,
): number {
  let offset = Number.isFinite(intercept) ? intercept : 0;
  for (let j = 0; j < weights.length; j += 1) {
    if (j === axisX || j === axisY) continue;
    const w = weights[j];
    const x = j < features.length ? features[j] : 0;
    if (!Number.isFinite(w) || !Number.isFinite(x)) continue;
    offset += w * x;
  }
  return offset;
}

/** Height of the SGD plane at (x, y). */
export function sgdPlaneHeight(
  wx: number,
  wy: number,
  interceptOffset: number,
  x: number,
  y: number,
): number {
  return wx * x + wy * y + interceptOffset;
}

/**
 * Live What-If height without API recall: off-axis features come from the
 * live vector while (x, y) sweep the plane.
 */
export function liveSgdHeight(
  weightRow: number[],
  baseIntercept: number,
  liveFeatures: number[],
  axisX: number,
  axisY: number,
  x: number,
  y: number,
): number {
  const offset = computeSgdPhaseShift(weightRow, baseIntercept, liveFeatures, axisX, axisY);
  const wx = axisX < weightRow.length ? weightRow[axisX] : 0;
  const wy = axisY < weightRow.length && axisY !== axisX ? weightRow[axisY] : 0;
  return sgdPlaneHeight(
    Number.isFinite(wx) ? wx : 0,
    Number.isFinite(wy) ? wy : 0,
    offset,
    x,
    y,
  );
}

/**
 * Height colormap for the margin plane: emerald (safe +) → amber (corridor)
 * → rose (safe −). Returns [r, g, b] in 0..1.
 */
export function sgdCorridorColor(t: number): [number, number, number] {
  const u = Math.min(1, Math.max(0, t));
  const safePos: [number, number, number] = [0x10 / 255, 0xb9 / 255, 0x8a / 255];
  const corridor: [number, number, number] = [0xf5 / 255, 0x9e / 255, 0x0b / 255];
  const safeNeg: [number, number, number] = [0xf4 / 255, 0x3f / 255, 0x5e / 255];
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  if (u <= 0.5) {
    const k = u / 0.5;
    return [lerp(safeNeg[0], corridor[0], k), lerp(safeNeg[1], corridor[1], k), lerp(safeNeg[2], corridor[2], k)];
  }
  const k = (u - 0.5) / 0.5;
  return [lerp(corridor[0], safePos[0], k), lerp(corridor[1], safePos[1], k), lerp(corridor[2], safePos[2], k)];
}

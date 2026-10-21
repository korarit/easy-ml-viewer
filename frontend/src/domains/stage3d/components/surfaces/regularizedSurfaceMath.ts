/**
 * Phase 19 — pure math helpers for the regularized tilted hyperplane.
 *
 * Kept dependency-free (no React / Three imports) so the analytical plane
 * formula is unit-testable and reusable by the inspector + WebGL surface.
 *
 * Convention: z(x, y) = w_x * x + w_y * y + interceptOffset,
 * where interceptOffset = b_k + Σ_{m ∉ {x,y}} w_{k,m} * x_m.
 */

export const REGULARIZED_ZERO_TOL = 1e-12;

export function isPrunedWeight(w: number, tol = REGULARIZED_ZERO_TOL): boolean {
  return !Number.isFinite(w) || Math.abs(w) <= tol;
}

export function sparsityRatio(weights: number[], tol = REGULARIZED_ZERO_TOL): number {
  if (weights.length === 0) return 0;
  let zeroed = 0;
  for (const w of weights) if (isPrunedWeight(w, tol)) zeroed += 1;
  return zeroed / weights.length;
}

/**
 * GPU phase-shift uniform for off-axis What-If sliders.
 */
export function computeRegularizedPhaseShift(
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

/** Height of the regularized plane at (x, y). */
export function regularizedPlaneHeight(
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
export function liveRegularizedHeight(
  weightRow: number[],
  baseIntercept: number,
  liveFeatures: number[],
  axisX: number,
  axisY: number,
  x: number,
  y: number,
): number {
  const offset = computeRegularizedPhaseShift(weightRow, baseIntercept, liveFeatures, axisX, axisY);
  const wx = axisX < weightRow.length ? weightRow[axisX] : 0;
  const wy = axisY < weightRow.length && axisY !== axisX ? weightRow[axisY] : 0;
  return regularizedPlaneHeight(
    Number.isFinite(wx) ? wx : 0,
    Number.isFinite(wy) ? wy : 0,
    offset,
    x,
    y,
  );
}

/**
 * Height colormap for the tilted plane: deep slate (low) → sky (mid) →
 * amber (high). Returns [r, g, b] in 0..1. Pruned axes render with a flat
 * slate tint handled by the 3D component (dimming), not here.
 */
export function regularizedHeightColor(t: number): [number, number, number] {
  const u = Math.min(1, Math.max(0, t));
  const low: [number, number, number] = [0x1e / 255, 0x3a / 255, 0x8a / 255];
  const mid: [number, number, number] = [0x38 / 255, 0xbd / 255, 0xf8 / 255];
  const high: [number, number, number] = [0xfa / 255, 0xcc / 255, 0x15 / 255];
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  if (u <= 0.5) {
    const k = u / 0.5;
    return [lerp(low[0], mid[0], k), lerp(low[1], mid[1], k), lerp(low[2], mid[2], k)];
  }
  const k = (u - 0.5) / 0.5;
  return [lerp(mid[0], high[0], k), lerp(mid[1], high[1], k), lerp(mid[2], high[2], k)];
}

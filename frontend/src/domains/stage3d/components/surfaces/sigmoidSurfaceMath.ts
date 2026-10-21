/**
 * Phase 18 — pure math helpers for the 3D Sigmoid S-Surface.
 *
 * Kept in a dependency-free module (no React / Three imports) so the
 * What-If phase-shift formula is unit-testable and reusable by both the
 * inspector panel and the WebGL surface component.
 *
 * Convention: z(x, y) = sigmoid(w_x * x + w_y * y + interceptOffset),
 * where interceptOffset = b_k + Σ_{m ∉ {x,y}} w_{k,m} * x_m.
 */

export function sigmoid(z: number): number {
  // Overflow-safe: branch before calling Math.exp.
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export function logitFromThreshold(tau: number): number {
  const t = Math.min(0.999999, Math.max(0.000001, tau));
  return Math.log(t / (1 - t));
}

/**
 * GPU phase-shift uniform for off-axis What-If sliders.
 *
 * @param weights full coefficient row w_k (length D)
 * @param intercept scalar b_k
 * @param features current full feature vector (length D)
 * @param axisX / axisY indices pinned to the 3D plane axes
 */
export function computeSigmoidPhaseShift(
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

/** Height of the S-surface at a plane point (probability space). */
export function sigmoidSurfaceHeight(
  wx: number,
  wy: number,
  interceptOffset: number,
  x: number,
  y: number,
): number {
  return sigmoid(wx * x + wy * y + interceptOffset);
}

/**
 * Live What-If height without API recall: off-axis features come from the
 * live vector while (x, y) sweep the plane.
 */
export function liveSigmoidHeight(
  weightRow: number[],
  baseIntercept: number,
  liveFeatures: number[],
  axisX: number,
  axisY: number,
  x: number,
  y: number,
): number {
  let offset = Number.isFinite(baseIntercept) ? baseIntercept : 0;
  for (let j = 0; j < weightRow.length; j += 1) {
    if (j === axisX || j === axisY) continue;
    const w = weightRow[j];
    const f = j < liveFeatures.length ? liveFeatures[j] : 0;
    if (!Number.isFinite(w) || !Number.isFinite(f)) continue;
    offset += w * f;
  }
  const wx = axisX < weightRow.length ? weightRow[axisX] : 0;
  const wy = axisY < weightRow.length ? weightRow[axisY] : 0;
  return sigmoid(wx * x + wy * y + offset);
}

/**
 * Diverging probability colormap: deep blue (0.0) → near-white (0.5) →
 * vivid orange-red (1.0). Returns [r, g, b] in 0..1.
 */
export function probabilityDivergingColor(p: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, p));
  // Anchors sampled from the Phase 18 visual spec (#1e3a8a / #f8fafc / #ea580c).
  const low: [number, number, number] = [0x1e / 255, 0x3a / 255, 0x8a / 255];
  const mid: [number, number, number] = [0xf8 / 255, 0xfa / 255, 0xfc / 255];
  const high: [number, number, number] = [0xea / 255, 0x58 / 255, 0x0c / 255];
  const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
  if (t <= 0.5) {
    const u = t / 0.5;
    return [lerp(low[0], mid[0], u), lerp(low[1], mid[1], u), lerp(low[2], mid[2], u)];
  }
  const u = (t - 0.5) / 0.5;
  return [lerp(mid[0], high[0], u), lerp(mid[1], high[1], u), lerp(mid[2], high[2], u)];
}

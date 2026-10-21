/**
 * Phase 20 — pure math helpers for the ElasticNet tilted hyperplane +
 * L1/L2 superellipse constraint ring.
 *
 * Kept dependency-free (no React / Three imports) so the analytical plane
 * formula and the ring solver are unit-testable and reusable by the
 * inspector + WebGL surface.
 *
 * Plane convention: z(x, y) = w_x * x + w_y * y + interceptOffset,
 * where interceptOffset = b_m + Σ_{j ∉ {x,y}} w_{m,j} * x_j.
 *
 * Ring convention (blueprint §5.4): the constraint aura on the base plane
 * follows  ρ(|u| + |v|) + (1 − ρ)(u² + v²) = R².  Along direction θ
 * (u = s·cosθ, v = s·sinθ) this is the quadratic  a·s² + b·s − R² = 0
 * with  a = (1 − ρ),  b = ρ(|cosθ| + |sinθ|).  At ρ = 1 (pure Lasso) the
 * quadratic degenerates to the sharp-diamond line  s = R² / b.
 */

export const ELASTICNET_ZERO_TOL = 1e-12;

export function isPrunedWeight(w: number, tol = ELASTICNET_ZERO_TOL): boolean {
  return !Number.isFinite(w) || Math.abs(w) <= tol;
}

export function sparsityRatio(weights: number[], tol = ELASTICNET_ZERO_TOL): number {
  if (weights.length === 0) return 0;
  let zeroed = 0;
  for (const w of weights) if (isPrunedWeight(w, tol)) zeroed += 1;
  return zeroed / weights.length;
}

/**
 * GPU phase-shift uniform for off-axis What-If sliders.
 */
export function computeElasticNetPhaseShift(
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

/** Height of the elastic-net plane at (x, y). */
export function elasticNetPlaneHeight(
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
export function liveElasticNetHeight(
  weightRow: number[],
  baseIntercept: number,
  liveFeatures: number[],
  axisX: number,
  axisY: number,
  x: number,
  y: number,
): number {
  const offset = computeElasticNetPhaseShift(weightRow, baseIntercept, liveFeatures, axisX, axisY);
  const wx = axisX < weightRow.length ? weightRow[axisX] : 0;
  const wy = axisY < weightRow.length && axisY !== axisX ? weightRow[axisY] : 0;
  return elasticNetPlaneHeight(
    Number.isFinite(wx) ? wx : 0,
    Number.isFinite(wy) ? wy : 0,
    offset,
    x,
    y,
  );
}

/**
 * Solve the constraint-ring radius along one direction θ.
 * Returns the non-negative root s (0 when degenerate).
 */
export function solveConstraintRadius(theta: number, l1Ratio: number, radius = 1): number {
  const rho = Math.min(1, Math.max(0, l1Ratio));
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const b = rho * (Math.abs(c) + Math.abs(s));
  const rhs = radius * radius;
  const a = 1 - rho;
  if (Math.abs(a) < 1e-9) {
    // Pure-Lasso sharp diamond: s = R² / (|cosθ| + |sinθ|).
    if (b <= 1e-12) return 0;
    return rhs / b;
  }
  const disc = b * b + 4 * a * rhs;
  if (!Number.isFinite(disc) || disc < 0) return 0;
  return (-b + Math.sqrt(disc)) / (2 * a);
}

/**
 * Sample the L1/L2 superellipse constraint ring.
 * Returns `segments` 3D-ready [x, y] pairs (z = 0 handled by the caller).
 * ρ = 0 → circle, 0 < ρ < 1 → rounded diamond, ρ = 1 → sharp diamond.
 */
export function elasticNetConstraintRing(
  l1Ratio: number,
  radius = 1,
  segments = 256,
): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  const n = Math.max(8, Math.floor(segments));
  for (let i = 0; i < n; i += 1) {
    const theta = (i / n) * Math.PI * 2;
    const r = solveConstraintRadius(theta, l1Ratio, radius);
    const safe = Number.isFinite(r) && r >= 0 ? r : 0;
    pts.push([safe * Math.cos(theta), safe * Math.sin(theta)]);
  }
  return pts;
}

/**
 * Ring stroke color for the tug-of-war gauge: sky blue at ρ = 0 (Ridge)
 * → violet-pink at ρ = 1 (Lasso). Returns a CSS hex string.
 */
export function constraintRingColor(l1Ratio: number): string {
  const u = Math.min(1, Math.max(0, l1Ratio));
  // #38bdf8 (sky) → #e879f9 (fuchsia)
  const lo = [0x38, 0xbd, 0xf8];
  const hi = [0xe8, 0x79, 0xf9];
  const mix = (a: number, b: number) => Math.round(a + (b - a) * u);
  const toHex = (v: number) => v.toString(16).padStart(2, '0');
  return `#${toHex(mix(lo[0], hi[0]))}${toHex(mix(lo[1], hi[1]))}${toHex(mix(lo[2], hi[2]))}`;
}

/**
 * Grouping-arc endpoints for the correlated-feature effect: when both plane
 * axes carry same-sign non-zero weight, ElasticNet keeps them together.
 * Returns null when the arc should not render.
 */
export function groupingArcEndpoints(
  wx: number,
  wy: number,
  tol = ELASTICNET_ZERO_TOL,
): { from: [number, number]; to: [number, number] } | null {
  if (Math.abs(wx) <= tol || Math.abs(wy) <= tol) return null;
  if (Math.sign(wx) !== Math.sign(wy)) return null;
  const peak = 0.9;
  return {
    from: [Math.sign(wx) * peak, 0],
    to: [0, Math.sign(wy) * peak],
  };
}

/**
 * Height colormap for the tilted plane: slate (low) → sky (mid) →
 * fuchsia (high, Lasso-leaning). Returns [r, g, b] in 0..1.
 */
export function elasticNetHeightColor(l1Ratio: number, t: number): [number, number, number] {
  const u = Math.min(1, Math.max(0, t));
  const rho = Math.min(1, Math.max(0, l1Ratio));
  const low: [number, number, number] = [0x1e / 255, 0x3a / 255, 0x8a / 255];
  const mid: [number, number, number] = [0x38 / 255, 0xbd / 255, 0xf8 / 255];
  // High stop drifts sky → fuchsia as ρ → 1.
  const high: [number, number, number] = [
    (0xfa + (0xe8 - 0xfa) * rho) / 255,
    (0xcc + (0x79 - 0xcc) * rho) / 255,
    (0x15 + (0xf9 - 0x15) * rho) / 255,
  ];
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  if (u <= 0.5) {
    const k = u / 0.5;
    return [lerp(low[0], mid[0], k), lerp(low[1], mid[1], k), lerp(low[2], mid[2], k)];
  }
  const k = (u - 0.5) / 0.5;
  return [lerp(mid[0], high[0], k), lerp(mid[1], high[1], k), lerp(mid[2], high[2], k)];
}

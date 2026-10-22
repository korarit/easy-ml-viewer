/**
 * Phase 31 — pure math helpers for the sklearn GBM residual spiral ladder.
 *
 * Dependency-free (no React / Three imports) so spiral rung heights,
 * stage colors and residual normalization stay unit-testable.
 */

/** Spiral rung direction bucket for a marginal raw residual delta. */
export function spiralDirection(delta: number): 'up' | 'down' | 'flat' {
  if (!Number.isFinite(delta) || Math.abs(delta) < 1e-12) return 'flat';
  return delta > 0 ? 'up' : 'down';
}

/** Rung color ramp: cyan/blue (push up) vs amber/rose (pull down). */
export function spiralColor(delta: number): [number, number, number] {
  const d = spiralDirection(delta);
  if (d === 'up') return [0.2, 0.75, 1.0];
  if (d === 'down') return [1.0, 0.45, 0.25];
  return [0.55, 0.6, 0.68];
}

/** Cumulative raw ladder heights from a base raw score and raw deltas. */
export function spiralHeights(base: number, deltas: number[]): number[] {
  const out: number[] = [];
  let acc = Number.isFinite(base) ? base : 0;
  for (const d of deltas) {
    acc += Number.isFinite(d) ? d : 0;
    out.push(acc);
  }
  return out;
}

/** Sigmoid link (binary GBM log-odds → probability), overflow-safe. */
export function gbmSigmoid(z: number): number {
  if (!Number.isFinite(z)) return 0;
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** First 1-based stage where the staged probability crosses threshold. */
export function crossedGbmStage(trajectory: number[], threshold = 0.5): number | null {
  for (let m = 0; m < trajectory.length; m += 1) {
    if (trajectory[m] >= threshold) return m + 1;
  }
  return null;
}

/** Normalize a regression terraces height into the display range [0, 2]. */
export function normalizeGbmHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function gbmPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Map a world value on an axis grid into the [-2.5, 2.5] stage space. */
export function gridToStageGbm(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

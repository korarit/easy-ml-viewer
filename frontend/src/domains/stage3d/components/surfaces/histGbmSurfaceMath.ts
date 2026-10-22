/**
 * Phase 32 — pure math helpers for the HistGBM quantized bin terraces.
 *
 * Dependency-free (no React / Three imports) so bin lookups, spiral rung
 * heights and residual normalization stay unit-testable.
 */

/** Spiral rung direction bucket for a marginal raw residual delta. */
export function histSpiralDirection(delta: number): 'up' | 'down' | 'flat' {
  if (!Number.isFinite(delta) || Math.abs(delta) < 1e-12) return 'flat';
  return delta > 0 ? 'up' : 'down';
}

/** Rung color ramp: teal/cyan (push up) vs amber/rose (pull down). */
export function histSpiralColor(delta: number): [number, number, number] {
  const d = histSpiralDirection(delta);
  if (d === 'up') return [0.2, 0.85, 0.75];
  if (d === 'down') return [1.0, 0.45, 0.3];
  return [0.55, 0.6, 0.68];
}

/** Cumulative raw ladder heights from a base raw score and raw deltas. */
export function histSpiralHeights(base: number, deltas: number[]): number[] {
  const out: number[] = [];
  let acc = Number.isFinite(base) ? base : 0;
  for (const d of deltas) {
    acc += Number.isFinite(d) ? d : 0;
    out.push(acc);
  }
  return out;
}

/** Sigmoid link (binary HistGBM log-odds → probability), overflow-safe. */
export function histSigmoid(z: number): number {
  if (!Number.isFinite(z)) return 0;
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** First 1-based stage where the staged probability crosses threshold. */
export function crossedHistStage(trajectory: number[], threshold = 0.5): number | null {
  for (let m = 0; m < trajectory.length; m += 1) {
    if (trajectory[m] >= threshold) return m + 1;
  }
  return null;
}

/** Normalize a regression terraces height into the display range [0, 2]. */
export function normalizeHistHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function histPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Map a world value on an axis grid into the [-2.5, 2.5] stage space. */
export function histGridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

/**
 * Locate the histogram bin for a raw value given sorted upper-bound
 * thresholds (mirrors `np.searchsorted(thresholds, x, side='left')`).
 */
export function findHistBin(thresholds: number[], v: number): number {
  if (!Number.isFinite(v)) return -1;
  let lo = 0;
  let hi = thresholds.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (v <= thresholds[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** Human-readable bounds of a bin: [lower, upper] with open ends. */
export function histBinRange(thresholds: number[], bin: number): [number | null, number | null] {
  if (bin < 0 || bin > thresholds.length) return [null, null];
  const lo = bin === 0 ? null : thresholds[bin - 1];
  const hi = bin >= thresholds.length ? null : thresholds[bin];
  return [lo, hi];
}

/** Downsample bin edges for grid rendering (cap lines per axis). */
export function downsampleBinEdges(thresholds: number[], cap = 64): number[] {
  if (thresholds.length <= cap) return thresholds.slice();
  const step = Math.ceil(thresholds.length / cap);
  return thresholds.filter((_, i) => i % step === 0);
}

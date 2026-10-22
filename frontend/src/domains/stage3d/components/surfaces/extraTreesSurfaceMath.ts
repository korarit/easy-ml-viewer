/**
 * Phase 29 — pure math helpers for the extra-trees uncertainty halo.
 *
 * Dependency-free (no React / Three imports) so halo radii, colors and
 * rug binning stay unit-testable.
 */

/** Halo size bucket for the badge (blueprint §2.1: halo in [0, 1]). */
export function haloLabel(radius: number): string {
  if (!Number.isFinite(radius)) return 'Unknown';
  if (radius <= 0.05) return 'Tight';
  if (radius <= 0.25) return 'Calm';
  if (radius <= 0.4) return 'Watch';
  return 'High';
}

/** Display scale for the halo shell around the query ball. */
export function haloScale(radius: number): number {
  if (!Number.isFinite(radius)) return 1;
  return 1 + Math.min(1, Math.max(0, radius)) * 2.2;
}

/** Halo color ramp: emerald (agreement) -> amber -> rose (split). */
export function haloColor(radius: number): [number, number, number] {
  const r = Math.min(1, Math.max(0, Number.isFinite(radius) ? radius : 0));
  if (r <= 0.25) {
    const t = r / 0.25;
    return [0.2 + t * 0.79, 0.83 - t * 0.21, 0.59 - t * 0.18];
  }
  const t = (r - 0.25) / 0.75;
  return [0.99, 0.62 - t * 0.29, 0.41 - t * 0.16];
}

/** Vote split for the distribution bar: [positiveShare, negativeShare]. */
export function etVoteShares(treePreds: number[], threshold = 0.5): [number, number] {
  if (treePreds.length === 0) return [0.5, 0.5];
  const pos = treePreds.filter((v) => v >= threshold).length / treePreds.length;
  return [pos, 1 - pos];
}

/** Pillar display height in [0.05, 2]: posterior for classifiers, z-scored mean for regressors. */
export function etPillarHeight(pred: number, zMin: number, zMax: number, isRegressor: boolean): number {
  if (!Number.isFinite(pred)) return 0.05;
  if (!isRegressor) return 0.05 + Math.min(1, Math.max(0, pred)) * 1.95;
  if (![zMin, zMax].every(Number.isFinite) || zMax <= zMin) return 1;
  return 0.05 + Math.min(1, Math.max(0, (pred - zMin) / (zMax - zMin))) * 1.95;
}

/** Normalize a regression surface height into the display range [0, 2]. */
export function normalizeExtraTreesHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function extraTreesPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Uncertainty std (already >= 0) into the display range [0, 2] via halo mapping. */
export function uncertaintyDisplayHeight(halo: number): number {
  if (!Number.isFinite(halo)) return 0;
  return Math.min(1, Math.max(0, halo)) * 2;
}

/** Human-readable breadcrumb for the first-tree reference path. */
export function extraTreesStepLabel(
  featureName: string,
  threshold: number,
  wentLeft: boolean,
  delta: number,
): string {
  const op = wentLeft ? '<=' : '>';
  const sign = delta >= 0 ? '+' : '';
  return `${featureName} ${op} ${threshold} (${wentLeft ? 'True' : 'False'}, Δ=${sign}${delta.toFixed(3)})`;
}

/**
 * Bin split thresholds into `bins` histogram buckets over [lo, hi] for the rug plot
 * (edge #5: binary-feature cut-points would otherwise overplot).
 */
export function binSplits(thresholds: number[], lo: number, hi: number, bins = 50): number[] {
  const out = new Array<number>(Math.max(1, bins)).fill(0);
  const span = hi - lo || 1;
  for (const t of thresholds) {
    if (!Number.isFinite(t)) continue;
    const b = Math.min(bins - 1, Math.max(0, Math.floor(((t - lo) / span) * bins)));
    out[b] += 1;
  }
  return out;
}

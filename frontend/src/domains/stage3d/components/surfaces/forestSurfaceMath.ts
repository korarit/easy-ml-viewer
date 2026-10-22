/**
 * Phase 28 — pure math helpers for the 3D forest voting arena.
 *
 * Dependency-free (no React / Three imports) so consensus labels, vote
 * tallies and pillar heights stay unit-testable.
 */

/** Agreement bucket for the consensus badge (blueprint §2.1). */
export function consensusLabel(ratio: number): string {
  if (!Number.isFinite(ratio)) return 'Unknown';
  if (ratio >= 0.85) return 'High';
  if (ratio >= 0.65) return 'Moderate';
  return 'Split';
}

/** Vote split for the distribution bar: [positiveShare, negativeShare]. */
export function voteShares(treePreds: number[], threshold = 0.5): [number, number] {
  if (treePreds.length === 0) return [0.5, 0.5];
  const pos = treePreds.filter((v) => v >= threshold).length / treePreds.length;
  return [pos, 1 - pos];
}

/** Pillar display height in [0.05, 2]: posterior for classifiers, z-scored mean for regressors. */
export function pillarHeight(pred: number, zMin: number, zMax: number, isRegressor: boolean): number {
  if (!Number.isFinite(pred)) return 0.05;
  if (!isRegressor) return 0.05 + Math.min(1, Math.max(0, pred)) * 1.95;
  if (![zMin, zMax].every(Number.isFinite) || zMax <= zMin) return 1;
  return 0.05 + (Math.min(1, Math.max(0, (pred - zMin) / (zMax - zMin))) * 1.95);
}

/** Pillar color key: winning side for the current frame. */
export function pillarSide(pred: number, threshold = 0.5): 'pos' | 'neg' {
  return pred >= threshold ? 'pos' : 'neg';
}

/** Normalize a forest regression surface height into the display range [0, 2]. */
export function normalizeForestHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function forestPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Human-readable breadcrumb for the first-tree reference path. */
export function forestStepLabel(
  featureName: string,
  threshold: number,
  wentLeft: boolean,
  delta: number,
): string {
  const op = wentLeft ? '<=' : '>';
  const sign = delta >= 0 ? '+' : '';
  return `${featureName} ${op} ${threshold} (${wentLeft ? 'True' : 'False'}, Δ=${sign}${delta.toFixed(3)})`;
}

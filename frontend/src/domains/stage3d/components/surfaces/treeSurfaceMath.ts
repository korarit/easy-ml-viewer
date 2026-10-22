/**
 * Phase 27 — pure math helpers for the 3D stepped voxel terraces.
 *
 * Dependency-free (no React / Three imports) so terrace normalization,
 * breadcrumb text and split-line filtering stay unit-testable.
 *
 * Terrace convention (blueprint §2.1): the surface is piecewise
 * constant over the axis pair — every grid cell renders as one flat
 * voxel box (no smooth interpolation across cliffs).
 */

/** Normalize a regression terrace height into the display range [0, 2]. */
export function normalizeRegressionHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function posteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Split lines at or above a depth clip (shallow splits render thicker). */
export function splitsUpToDepth<T extends { depth: number }>(lines: T[], depthClip: number): T[] {
  return lines.filter((l) => l.depth <= depthClip);
}

/** Line width weight: shallow splits (depth 0–2) render thickest. */
export function splitLineWeight(depth: number, maxDepth: number): number {
  const d = Math.max(0, Math.min(maxDepth, depth));
  return Math.max(0.35, 1 - d / Math.max(1, maxDepth + 1));
}

/** Human-readable breadcrumb for one active-path step. */
export function stepLabel(
  featureName: string,
  threshold: number,
  wentLeft: boolean,
  delta: number,
): string {
  const op = wentLeft ? '<=' : '>';
  const sign = delta >= 0 ? '+' : '';
  return `${featureName} ${op} ${threshold} (${wentLeft ? 'True' : 'False'}, Δ=${sign}${delta.toFixed(3)})`;
}

/** Leaf badge text. */
export function leafBadge(
  leafId: number,
  impurity: number,
  nSamples: number,
  expectation: number,
  isRegressor: boolean,
): string {
  const head = isRegressor ? `ŷ=${expectation.toFixed(3)}` : `E=${expectation.toFixed(3)}`;
  return `Leaf #${leafId}: ${head} · impurity ${impurity.toFixed(3)} · N=${nSamples}`;
}

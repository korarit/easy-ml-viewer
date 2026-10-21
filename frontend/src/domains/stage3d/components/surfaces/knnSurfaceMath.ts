/**
 * Phase 22 — pure math helpers for the KNN expanding radar sphere +
 * spider-web laser lines.
 *
 * Dependency-free (no React / Three imports) so vote shares, radar
 * badges and 3D projection distances stay unit-testable.
 *
 * Blueprint §2: R_t = max neighbor distance (fixed-K) or model.radius;
 * shape follows effective_metric_: sphere (L2), octahedron (L1), cube
 * (chebyshev). Spider opacity/width scale with the vote weight w_k.
 */

export type KnnRadarShape = 'sphere' | 'octahedron' | 'cube';

export function radarShapeLabel(shape: KnnRadarShape | string): string {
  if (shape === 'octahedron') return 'Octahedron (L1 diamond)';
  if (shape === 'cube') return 'Cube (L∞ chebyshev)';
  return 'Sphere (L2 euclidean)';
}

/** Vote-weight share of neighbor k (already normalized server-side). */
export function voteShare(weights: number[], k: number): number {
  if (k < 0 || k >= weights.length) return 0;
  const w = weights[k];
  return Number.isFinite(w) && w >= 0 ? w : 0;
}

/** Spider line width in [1, 4] px scaled by vote weight. */
export function spiderLineWidth(weight: number, maxWeight: number): number {
  if (!Number.isFinite(weight) || !Number.isFinite(maxWeight) || maxWeight <= 0) return 1;
  return 1 + 3 * Math.min(1, Math.max(0, weight / maxWeight));
}

/** Spider line opacity in [0.25, 1] scaled by vote weight. */
export function spiderOpacity(weight: number, maxWeight: number): number {
  if (!Number.isFinite(weight) || !Number.isFinite(maxWeight) || maxWeight <= 0) return 0.35;
  return 0.25 + 0.75 * Math.min(1, Math.max(0, weight / maxWeight));
}

/** 3D-projected euclidean distance on the (axisX, axisY, height) axes. */
export function projected3DDistance(
  query: number[],
  neighborCoord: number[],
  axisX: number,
  axisY: number,
  queryHeight: number,
  neighborHeight: number,
): number {
  const qx = axisX < query.length ? query[axisX] : 0;
  const qy = axisY < query.length ? query[axisY] : 0;
  const nx = axisX < neighborCoord.length ? neighborCoord[axisX] : 0;
  const ny = axisY < neighborCoord.length ? neighborCoord[axisY] : 0;
  const dx = qx - nx;
  const dy = qy - ny;
  const dz = queryHeight - neighborHeight;
  if (![dx, dy, dz].every(Number.isFinite)) return 0;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Normalize a value grid to a display height in [0, 2]. */
export function normalizeKnnHeight(zMin: number, zMax: number, z: number): number {
  const span = zMax - zMin;
  if (!Number.isFinite(span) || span <= 1e-9) return 1;
  const t = (z - zMin) / span;
  return Math.min(1, Math.max(0, t)) * 2;
}

/** Vote-tally bar fractions for the HUD (label -> share of K). */
export function tallyFractions(
  tally: Array<{ label: string; count: number; weight_share: number }>,
  totalK: number,
): Array<{ label: string; count: number; weightShare: number; countShare: number }> {
  const k = totalK > 0 ? totalK : 1;
  return tally.map((t) => ({
    label: t.label,
    count: t.count,
    weightShare: Number.isFinite(t.weight_share) ? t.weight_share : 0,
    countShare: t.count / k,
  }));
}

/**
 * Phase 26 — pure math helpers for the LDA/QDA tilted covariance
 * ellipsoids, Mahalanobis-vs-Euclidean paradox flag and 1/2/3σ shells.
 *
 * Dependency-free (no React / Three imports) so distances, paradox
 * detection and ellipse footprints stay unit-testable.
 *
 * Ellipse convention (blueprint §2.2): the 2×2 sub-covariance on the
 * axis pair eigendecomposes to V·diag(λ)·Vᵀ (det(V) = +1); the m-σ
 * radii are r_i = m·sqrt(λ_i) and columns of V are the principal axes.
 */

export type SigmaLevel = 1 | 2 | 3;

/** Squared Mahalanobis distance via an explicit precision matrix. */
export function mahalanobisSq(diff: number[], precision: number[][]): number {
  let acc = 0;
  for (let i = 0; i < diff.length; i += 1) {
    let row = 0;
    const prow = precision[i] ?? [];
    for (let j = 0; j < diff.length; j += 1) row += (prow[j] ?? 0) * (diff[j] ?? 0);
    acc += (diff[i] ?? 0) * row;
  }
  return Math.max(0, acc);
}

/** Euclidean distance between two points. */
export function euclideanDist(a: number[], b: number[]): number {
  let acc = 0;
  for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    acc += d * d;
  }
  return Math.sqrt(Math.max(0, acc));
}

/** Paradox: Mahalanobis-nearest class differs from Euclidean-nearest. */
export function paradoxDetected(mahal: number[], euclid: number[]): boolean {
  if (mahal.length === 0 || euclid.length === 0) return false;
  const argmin = (arr: number[]): number => {
    let bi = 0;
    for (let i = 1; i < arr.length; i += 1) {
      if ((arr[i] ?? Infinity) < (arr[bi] ?? Infinity)) bi = i;
    }
    return bi;
  };
  return argmin(mahal) !== argmin(euclid);
}

/** Point on an axis-aligned-then-rotated ellipse footprint. */
export function ellipsePoint(
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  angleRad: number,
  t: number,
): [number, number] {
  const ex = Math.max(0, rx) * Math.cos(t);
  const ey = Math.max(0, ry) * Math.sin(t);
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  return [cx + ex * c - ey * s, cy + ex * s + ey * c];
}

/** m-σ radii from 1σ radii. */
export function sigmaRadii(radii1: [number, number], level: SigmaLevel): [number, number] {
  return [radii1[0] * level, radii1[1] * level];
}

/** Class color palette (tab10-ish, colorblind-safe order). */
export function ellipsoidColor(k: number): [number, number, number] {
  const palette: Array<[number, number, number]> = [
    [0x22 / 255, 0xd3 / 255, 0xee / 255],
    [0xfb / 255, 0x92 / 255, 0x3c / 255],
    [0x34 / 255, 0xd3 / 255, 0x99 / 255],
    [0xf5 / 255, 0x9e / 255, 0x0b / 255],
    [0xe8 / 255, 0x79 / 255, 0xf9 / 255],
    [0x81 / 255, 0x8c / 255, 0xf8 / 255],
    [0xf4 / 255, 0x3f / 255, 0x5e / 255],
    [0xa3 / 255, 0xe6 / 255, 0x35 / 255],
  ];
  return palette[Math.abs(k) % palette.length];
}

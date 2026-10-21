/**
 * Phase 24 — pure math helpers for the GaussianNB bivariate bell hills,
 * posterior mode, z-score inspector and vertical density drop-pins.
 *
 * Dependency-free (no React / Three imports) so hill heights, posterior
 * normalization and outlier flags stay unit-testable.
 *
 * Hill convention (blueprint §3.1):
 *   H_k(x, y) = π_k · exp(-((x-μx)²/(2σx²) + (y-μy)²/(2σy²)))
 * Posterior hills: P_k = H_k / Σ_c H_c (computed client-side).
 */

/** Relative bell-hill height of class k at (x, y). */
export function bellHillHeight(
  prior: number,
  muX: number,
  muY: number,
  sigmaX: number,
  sigmaY: number,
  x: number,
  y: number,
): number {
  const sx = Number.isFinite(sigmaX) && sigmaX > 0 ? sigmaX : 1;
  const sy = Number.isFinite(sigmaY) && sigmaY > 0 ? sigmaY : 1;
  if (![prior, muX, muY, x, y].every(Number.isFinite)) return 0;
  const zx = ((x - muX) * (x - muX)) / (2 * sx * sx);
  const zy = ((y - muY) * (y - muY)) / (2 * sy * sy);
  return Math.max(0, prior) * Math.exp(-(zx + zy));
}

/** Normalize posterior hills from generative hills (guard divide-by-zero). */
export function posteriorFromHills(hills: number[]): number[] {
  const total = hills.reduce((a, b) => a + (Number.isFinite(b) ? Math.max(0, b) : 0), 0);
  if (!(total > 0)) return hills.map(() => 0);
  return hills.map((h) => (Number.isFinite(h) ? Math.max(0, h) : 0) / total);
}

/** Z-score distance of x from (mu, sigma) in sigma units. */
export function zScore(x: number, mu: number, sigma: number): number {
  const s = Number.isFinite(sigma) && sigma > 0 ? sigma : 1;
  if (![x, mu].every(Number.isFinite)) return 0;
  return Math.abs(x - mu) / s;
}

/** Low-density outlier flag: far from every class center. */
export function isLowDensityOutlier(perClassMinZ: number[], threshold = 4): boolean {
  if (perClassMinZ.length === 0) return false;
  return Math.min(...perClassMinZ) > threshold;
}

/** Normalize a hill height to display range [0, 2]. */
export function normalizeHillHeight(hMax: number, h: number): number {
  if (!Number.isFinite(hMax) || hMax <= 1e-12) return 0;
  if (!Number.isFinite(h) || h <= 0) return 0;
  return Math.min(1, h / hMax) * 2;
}

/** Class color palette (tab10-ish, colorblind-safe order). */
export function hillColor(k: number): [number, number, number] {
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

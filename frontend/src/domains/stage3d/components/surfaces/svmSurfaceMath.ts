/**
 * Phase 23 — pure math helpers for the SVM curved kernel saddle +
 * support-vector pillars and top-5 force tethers.
 *
 * Dependency-free (no React / Three imports) so kernel badges, tether
 * styling and saddle normalization stay unit-testable.
 *
 * Score convention: f(x) = Σ dual_s·K(x_s, x) + b. Margin contours sit
 * at f(x) ∈ {-1, 0, +1} for classifiers (ε-tube ±ε for regressors,
 * envelope f(x) = 0 for One-Class).
 */

export type SvmKernelName = 'rbf' | 'poly' | 'sigmoid' | 'linear' | string;

/** HUD badge label for the fitted kernel. */
export function kernelBadge(kernel: SvmKernelName, gamma: number | string): string {
  const g = typeof gamma === 'number' && Number.isFinite(gamma) ? `γ=${gamma.toFixed(3)}` : `γ=${String(gamma)}`;
  if (kernel === 'rbf') return `RBF saddle (${g})`;
  if (kernel === 'poly') return `Poly saddle (${g})`;
  if (kernel === 'sigmoid') return `Sigmoid sheet (${g})`;
  if (kernel === 'linear') return 'Linear plane';
  return `Custom kernel (${String(kernel)})`;
}

/** Tether line width in [1, 4] px scaled by |force|. */
export function tetherLineWidth(force: number, maxAbs: number): number {
  if (!Number.isFinite(force) || !Number.isFinite(maxAbs) || maxAbs <= 0) return 1;
  return 1 + 3 * Math.min(1, Math.abs(force) / maxAbs);
}

/** Tether line opacity in [0.25, 1] scaled by |force|. */
export function tetherOpacity(force: number, maxAbs: number): number {
  if (!Number.isFinite(force) || !Number.isFinite(maxAbs) || maxAbs <= 0) return 0.35;
  return 0.25 + 0.75 * Math.min(1, Math.abs(force) / maxAbs);
}

/** Normalize a saddle height to display range [0, 2]. */
export function normalizeSaddleHeight(zMin: number, zMax: number, z: number): number {
  const span = zMax - zMin;
  if (!Number.isFinite(span) || span <= 1e-9) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / span)) * 2;
}

/**
 * Saddle colormap: rose (below margin) → amber (inside corridor) →
 * emerald (safe above). Returns [r, g, b] in 0..1.
 */
export function svmSaddleColor(t: number): [number, number, number] {
  const u = Math.min(1, Math.max(0, t));
  const below: [number, number, number] = [0xf4 / 255, 0x3f / 255, 0x5e / 255];
  const corridor: [number, number, number] = [0xf5 / 255, 0x9e / 255, 0x0b / 255];
  const above: [number, number, number] = [0x10 / 255, 0xb9 / 255, 0x8a / 255];
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  if (u <= 0.5) {
    const k = u / 0.5;
    return [lerp(below[0], corridor[0], k), lerp(below[1], corridor[1], k), lerp(below[2], corridor[2], k)];
  }
  const k = (u - 0.5) / 0.5;
  return [lerp(corridor[0], above[0], k), lerp(corridor[1], above[1], k), lerp(corridor[2], above[2], k)];
}

/** Margin-side label for a raw decision score (binary / one-class). */
export function svmMarginLabel(rawScore: number, isOneClass: boolean): string {
  if (!Number.isFinite(rawScore)) return '—';
  if (isOneClass) return rawScore >= 0 ? 'INLIER (f(x) ≥ 0)' : 'OUTLIER (f(x) < 0)';
  if (Math.abs(rawScore) < 1) return 'INSIDE MARGIN CORRIDOR';
  return rawScore >= 0 ? 'SAFE MARGIN (> +1)' : 'SAFE MARGIN (< −1)';
}

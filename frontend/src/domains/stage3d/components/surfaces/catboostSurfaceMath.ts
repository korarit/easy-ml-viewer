/**
 * Phase 35 — pure math helpers for the CatBoost symmetric oblivious-grid stage.
 *
 * Dependency-free (no React / Three imports) so bitmask decoding, SHAP
 * force-vector scaling and orthogonal grid mapping stay unit-testable.
 */

/** Sigmoid link (binary raw score → probability), overflow-safe. */
export function catboostSigmoid(z: number): number {
  if (!Number.isFinite(z)) return 0;
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** First 1-based iteration where the staged linked prediction crosses threshold. */
export function crossedCatboostRound(trajectory: number[], threshold = 0.5): number | null {
  for (let m = 0; m < trajectory.length; m += 1) {
    if (trajectory[m] >= threshold) return m + 1;
  }
  return null;
}

/** Normalize a regression terrace height into the display range [0, 2]. */
export function normalizeCatboostHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function catboostPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Map a world value on an axis grid into the [-2.5, 2.5] stage space. */
export function catboostGridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

/** Decode a leaf index into a depth-bit path (LSB = depth 0). */
export function catboostBitmask(leaf: number, depth: number): string {
  const d = Math.max(1, Math.floor(depth) || 1);
  const bits: string[] = [];
  for (let i = 0; i < d; i += 1) {
    bits.push((leaf >> i) & 1 ? '1' : '0');
  }
  return bits.reverse().join('');
}

/** Per-depth Yes/No answers from a leaf bitmask (depth 0 first). */
export function catboostBitAnswers(leaf: number, depth: number): Array<'YES' | 'NO'> {
  const d = Math.max(1, Math.floor(depth) || 1);
  const out: Array<'YES' | 'NO'> = [];
  for (let i = 0; i < d; i += 1) {
    out.push((leaf >> i) & 1 ? 'YES' : 'NO');
  }
  return out;
}

/** Scale a SHAP value into a force-arrow length (display units). */
export function catboostForceLength(shap: number, maxAbs: number): number {
  if (![shap, maxAbs].every(Number.isFinite) || maxAbs <= 0) return 0;
  return Math.min(1.2, (Math.abs(shap) / maxAbs) * 1.2);
}

/** Force-arrow color: amber push (positive SHAP) vs rose pull (negative). */
export function catboostForceColor(shap: number): [number, number, number] {
  if (!Number.isFinite(shap) || shap === 0) return [0.55, 0.6, 0.68];
  return shap > 0 ? [1.0, 0.72, 0.2] : [1.0, 0.4, 0.45];
}

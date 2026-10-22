/**
 * Phase 30 — pure math helpers for the adaboost sequential cascade ladder.
 *
 * Dependency-free (no React / Three imports) so ladder rungs, stage
 * colors and curtain placement stay unit-testable.
 */

/** Ladder rung direction bucket for a marginal stage delta. */
export function rungDirection(delta: number): 'up' | 'down' | 'flat' {
  if (!Number.isFinite(delta) || Math.abs(delta) < 1e-12) return 'flat';
  return delta > 0 ? 'up' : 'down';
}

/** Rung color ramp: emerald/cyan (push up) vs amber/rose (pull down). */
export function rungColor(delta: number): [number, number, number] {
  const d = rungDirection(delta);
  if (d === 'up') return [0.13, 0.83, 0.93];
  if (d === 'down') return [0.98, 0.45, 0.3];
  return [0.55, 0.6, 0.68];
}

/** Ring thickness weight from the learner weight α (normalized). */
export function rungWeight(alpha: number, maxAlpha: number): number {
  if (![alpha, maxAlpha].every(Number.isFinite) || maxAlpha <= 0) return 0.5;
  return 0.25 + (Math.min(1, Math.max(0, alpha / maxAlpha)) * 0.75);
}

/** Cumulative ladder heights from a base value and per-stage deltas. */
export function ladderHeights(base: number, deltas: number[]): number[] {
  const out: number[] = [];
  let acc = Number.isFinite(base) ? base : 0;
  for (const d of deltas) {
    acc += Number.isFinite(d) ? d : 0;
    out.push(acc);
  }
  return out;
}

/** First 1-based stage where the trajectory crosses the threshold (binary). */
export function crossedStage(trajectory: number[], threshold = 0.5): number | null {
  for (let m = 0; m < trajectory.length; m += 1) {
    if (trajectory[m] >= threshold) return m + 1;
  }
  return null;
}

/** Normalize a regression terraces height into the display range [0, 2]. */
export function normalizeAdaBoostHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function adaBoostPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Map a world value on an axis grid into the [-2.5, 2.5] stage space. */
export function gridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

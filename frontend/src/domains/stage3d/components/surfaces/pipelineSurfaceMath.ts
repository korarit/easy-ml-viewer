/**
 * Phase 37 — pure math helpers for the pipeline tunnel + delegated surface.
 *
 * Dependency-free (no React / Three imports) so lineage aggregation and
 * stage scaling stay unit-testable.
 */

/** Sum transformed contributions back onto raw columns (Shapley additivity). */
export function aggregateLineageContributions(
  transformed: Record<string, number>,
  lineage: Record<string, string[]>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [raw, cols] of Object.entries(lineage)) {
    let total = 0;
    for (const c of cols) total += transformed[c] ?? 0;
    out[raw] = Number.isFinite(total) ? total : 0;
  }
  return out;
}

/** Partition check: every transformed column must have exactly one raw owner. */
export function lineageCoversAll(
  lineage: Record<string, string[]>,
  transformedNames: string[],
): boolean {
  const owned = new Set<string>();
  for (const cols of Object.values(lineage)) for (const c of cols) owned.add(c);
  return transformedNames.every((c) => owned.has(c));
}

/** Map a raw-space grid value into the [-2.5, 2.5] stage space. */
export function pipelineGridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

/** Posterior height (already in [0, 1]) into the display range [0, 2]. */
export function pipelinePosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Normalize a regression terrace height into the display range [0, 2]. */
export function normalizePipelineHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Top-N non-zero entries of a transformed row slice (telemetry truncation). */
export function topNonZeroEntries(
  row: Record<string, number>,
  topN = 16,
): Array<[string, number]> {
  return Object.entries(row)
    .filter(([, v]) => Number.isFinite(v) && Math.abs(v) > 1e-12)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, Math.max(1, topN));
}

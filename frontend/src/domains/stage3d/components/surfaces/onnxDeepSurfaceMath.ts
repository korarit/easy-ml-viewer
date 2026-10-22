/**
 * Phase 39 — pure math helpers for the ONNX deep dual 3D stage.
 *
 * Dependency-free (no React / Three imports) so latent-cloud normalization,
 * layer-pulse scaling and telemetry formatting stay unit-testable.
 */

/** Normalize a latent cloud (N×3 rows) into the [-2.5, 2.5] stage cube. */
export function normalizeLatentCloud(points: number[][]): number[][] {
  if (points.length === 0) return [];
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const p of points) {
    for (const v of p) {
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {
    return points.map(() => [0, 0, 0]);
  }
  const span = hi - lo;
  return points.map((p) =>
    [0, 1, 2].map((k) => {
      const v = p[k];
      if (!Number.isFinite(v)) return 0;
      return Math.min(2.5, Math.max(-2.5, ((v - lo) / span) * 5 - 2.5));
    }),
  );
}

/** Layer pulse height for a mean-activation bar (compressed to [0.02, 1]). */
export function latentPulseHeight(meanActivation: number, sigma: number): number {
  if (![meanActivation, sigma].every(Number.isFinite) || sigma <= 0) return 0.02;
  const t = Math.tanh(Math.abs(meanActivation) / sigma);
  return 0.02 + 0.98 * Math.min(1, Math.max(0, t));
}

/** Sparsity fraction → bar shade in [0, 1] (denser layers read brighter). */
export function latentSparsityShade(sparsity: number): number {
  if (!Number.isFinite(sparsity)) return 0.5;
  return Math.min(1, Math.max(0, 1 - sparsity));
}

/** Compact provider badge label, e.g. `DmlExecutionProvider (GPU)`. */
export function providerBadgeLabel(provider: string): string {
  if (provider.includes('CUDA') || provider.includes('Dml') || provider.includes('CoreML')) {
    return `${provider} (GPU)`;
  }
  return `${provider} (CPU)`;
}

/** Batch-mode pill label for the telemetry strip. */
export function batchModeLabel(batchMode: string, staticBatchSize: number | null): string {
  if (batchMode === 'static_unrolled') {
    return `Static batch_size=${staticBatchSize ?? 1} Auto-Unroll Active`;
  }
  return 'Dynamic Batch';
}

/** Format a latent coordinate triple for the HUD, e.g. `[+2.10, -0.80, +1.40]`. */
export function formatLatentCoords(coords: number[]): string {
  const triple = [coords[0] ?? 0, coords[1] ?? 0, coords[2] ?? 0];
  return `[${triple.map((v) => (Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}` : '+0.00')).join(', ')}]`;
}

/** Map a surface grid value into the [-2.5, 2.5] stage space. */
export function deepGridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

/** Posterior height (already in [0, 1]) into the display range [0.02, 2]. */
export function deepPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0.02;
  return Math.min(2, Math.max(0.02, p * 2));
}

/**
 * Phase 36 — pure math helpers for the MLP floating synapse-brain stage.
 *
 * Dependency-free (no React / Three imports) so flux polarity, node-radius
 * compression and stage layout stay unit-testable.
 */

/** Sigmoid link (binary raw score → probability), overflow-safe. */
export function mlpSigmoid(z: number): number {
  if (!Number.isFinite(z)) return 0;
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Synapse polarity: positive flux = excitatory, negative = inhibitory. */
export function mlpFluxPolarity(flux: number): 'excitatory' | 'inhibitory' | 'silent' {
  if (!Number.isFinite(flux) || flux === 0) return 'silent';
  return flux > 0 ? 'excitatory' : 'inhibitory';
}

/** Synapse line color: cyan pulse (excitatory) vs crimson (inhibitory). */
export function mlpFluxColor(flux: number): [number, number, number] {
  if (!Number.isFinite(flux) || flux === 0) return [0.45, 0.5, 0.58];
  return flux > 0 ? [0.25, 0.9, 1.0] : [1.0, 0.3, 0.35];
}

/**
 * Compress an unbounded activation magnitude into the display radius range
 * [rMin, rMax] via tanh(|a| / sigma) (blueprint edge #4: ReLU explosions).
 */
export function mlpNodeRadius(absActivation: number, sigma: number, rMin = 0.06, rMax = 0.22): number {
  if (![absActivation, sigma].every(Number.isFinite) || sigma <= 0) return rMin;
  const t = Math.tanh(Math.abs(absActivation) / sigma);
  return rMin + (rMax - rMin) * Math.min(1, Math.max(0, t));
}

/** Layer column X position across the stage span for L layers. */
export function mlpLayerX(layer: number, layerCount: number, span = 5): number {
  if (![layer, layerCount].every(Number.isFinite) || layerCount <= 1) return 0;
  const clamped = Math.min(Math.max(layer, 0), layerCount - 1);
  return -span / 2 + (clamped / (layerCount - 1)) * span;
}

/** Vertical slot of neuron i within its column of height `height`. */
export function mlpNodeY(index: number, count: number, height = 4.4): number {
  if (![index, count].every(Number.isFinite) || count <= 0) return 0;
  if (count === 1) return 0;
  const clamped = Math.min(Math.max(index, 0), count - 1);
  return height / 2 - (clamped / (count - 1)) * height;
}

/** Map a surface grid value into the [-2.5, 2.5] stage space. */
export function mlpGridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

/** Normalize a regression terrace height into the display range [0, 2]. */
export function normalizeMlpHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function mlpPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Robust activation scale: mean absolute activation over a layer sample. */
export function mlpActivationSigma(sample: number[]): number {
  const finite = sample.filter(Number.isFinite).map(Math.abs);
  if (finite.length === 0) return 1;
  const mean = finite.reduce((a, b) => a + b, 0) / finite.length;
  return mean > 1e-9 ? mean : 1;
}

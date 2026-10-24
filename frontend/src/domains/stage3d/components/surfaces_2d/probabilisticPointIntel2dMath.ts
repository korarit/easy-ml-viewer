/** Phase 52 — point-intelligence math for probabilistic & discriminant models. */

import { wy } from '../DecisionBoundaryContour2D';
import { hexPath } from './pointIntelligence2dMath';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';

/** Ambiguity-zone glow: neon magenta (spec §2.1, H → log2 C). */
export const ENTROPY_MAGENTA = '#e879f9';
/** Centroid spoke toward the true class (spec §2.3). */
export const SPOKE_TRUE_COLOR = '#22c55e';
/** Centroid spoke toward the predicted class (spec §2.3). */
export const SPOKE_PRED_COLOR = '#fb923c';
/** Over-dense bins: model underestimates the empirical mass. */
export const DIVERGENCE_OVER_COLOR = '#f43f5e';
/** Under-dense bins: data desert vs the Gaussian assumption. */
export const DIVERGENCE_UNDER_COLOR = '#f59e0b';

export interface ClassGaussian2D {
  meanRaw: [number, number];
  stdRaw: [number, number];
  rotationRad: number;
  prior: number | null;
  label: string | number | null;
}

/** Shannon entropy in bits (spec §2.1); NaN rows stay neutral. */
export function shannonEntropy(proba: number[]): number {
  const row = proba.filter((p) => Number.isFinite(p) && p >= 0);
  const total = row.reduce((s, p) => s + p, 0);
  if (row.length === 0 || total <= 1e-12) return NaN;
  let h = 0;
  for (const p of row) {
    const q = p / total;
    if (q > 0) h -= q * Math.log2(q);
  }
  return h;
}

/** Rebuild the 2×2 covariance from a Phase-44 ellipse spec. */
export function covFromStdRotation(std: [number, number], rotation: number): [[number, number], [number, number]] {
  // The extractor stores std ASCENDING [minor, major] with rotation = the
  // MAJOR-axis direction, so Σ = R(θ)·diag(sMajor², sMinor²)·R(θ)ᵀ. Plain
  // R·diag(s0², s1²)·Rᵀ would transpose the axes (proven err 1.0 vs 4e-16).
  const sMinor = Number.isFinite(std[0]) && std[0] > 0 ? std[0] : 1;
  const sMajor = Number.isFinite(std[1]) && std[1] > 0 ? std[1] : 1;
  const th = Number.isFinite(rotation) ? rotation : 0;
  const c = Math.cos(th);
  const s = Math.sin(th);
  // R·diag(sMajor², sMinor²)·Rᵀ
  const a = sMajor * sMajor;
  const b = sMinor * sMinor;
  return [
    [c * c * a + s * s * b, c * s * (a - b)],
    [c * s * (a - b), s * s * a + c * c * b],
  ];
}

function regularizedLogPdf(
  px: number,
  py: number,
  mean: [number, number],
  cov: [[number, number], [number, number]],
): number {
  if (!Number.isFinite(px) || !Number.isFinite(py)) return NaN;
  // Symmetric + floored eigenvalues (mirrors the backend EIG_FLOOR).
  const a = (cov[0][0] + cov[0][0]) / 2 + 1e-9;
  const b = (cov[0][1] + cov[1][0]) / 2;
  const d = (cov[1][1] + cov[1][1]) / 2 + 1e-9;
  const tr = a + d;
  const det = Math.max(a * d - b * b, 1e-18);
  const disc = Math.sqrt(Math.max(tr * tr - 4 * det, 0));
  const l1 = Math.max((tr + disc) / 2, 1e-9);
  const l2 = Math.max((tr - disc) / 2, 1e-9);
  const logDet = Math.log(l1) + Math.log(l2);
  // Σ⁻¹ via adjugate.
  const inv: [[number, number], [number, number]] = [
    [d / det, -b / det],
    [-b / det, a / det],
  ];
  const dx = px - mean[0];
  const dy = py - mean[1];
  const m2 = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
  return -0.5 * (m2 + logDet + 2 * Math.log(2 * Math.PI));
}

/**
 * Exact 2D-slice posterior P(c|x) ∝ prior·N(x; μ, Σ), log-sum-exp
 * normalized (mirrors `gaussian_posterior_2d`). Rows sum to 1.
 */
export function gaussianPosterior2D(px: number, py: number, classes: ClassGaussian2D[]): number[] {
  if (classes.length === 0) return [];
  const n = classes.length;
  const priors = classes.map((g) =>
    g.prior !== null && Number.isFinite(g.prior) && g.prior > 0 ? g.prior : 1 / n,
  );
  const pSum = priors.reduce((s, p) => s + p, 0);
  const logJoint = classes.map((g, k) => {
    const cov = covFromStdRotation(g.stdRaw, g.rotationRad);
    return regularizedLogPdf(px, py, g.meanRaw, cov) + Math.log(Math.max(priors[k]! / pSum, 1e-15));
  });
  if (logJoint.some((v) => !Number.isFinite(v))) return classes.map(() => NaN);
  const m = Math.max(...logJoint);
  const exps = logJoint.map((v) => Math.exp(v - m));
  const denom = exps.reduce((s, v) => s + v, 0);
  if (!(denom > 0)) return classes.map(() => NaN);
  return exps.map((v) => v / denom);
}

/** Dual-ring scale 0.5 + 0.5·maxP (spec §2.2). */
export function posteriorRingScale(maxProba: number): number {
  const p = Number.isFinite(maxProba) ? Math.min(1, Math.max(0, maxProba)) : 0.5;
  return 0.5 + 0.5 * p;
}

/** Ambiguity glow when H/log2(C) exceeds the threshold (spec §2.1). */
export function isAmbiguous(entropyBits: number, nClasses: number, threshold = 0.8): boolean {
  if (!Number.isFinite(entropyBits) || nClasses < 2) return false;
  return entropyBits / Math.log2(nClasses) > threshold;
}

/**
 * Trail color for one frame (DoD #1): the predicted class color when the
 * model is sure, neon magenta inside the ambiguity zone.
 */
export function entropyTrailColor(
  entropyBits: number,
  nClasses: number,
  confidentColor: string,
  threshold = 0.8,
): string {
  return isAmbiguous(entropyBits, nClasses, threshold) ? ENTROPY_MAGENTA : confidentColor;
}

/** Mahalanobis distance of a raw point to one class Gaussian. */
export function mahalanobis2D(
  px: number,
  py: number,
  mean: [number, number],
  cov: [[number, number], [number, number]],
): number {
  if (!Number.isFinite(px) || !Number.isFinite(py)) return NaN;
  // Direct precision path (stable even far in the tails).
  const a = (cov[0][0] + cov[0][0]) / 2 + 1e-9;
  const b = (cov[0][1] + cov[1][0]) / 2;
  const d = (cov[1][1] + cov[1][1]) / 2 + 1e-9;
  const det = Math.max(a * d - b * b, 1e-18);
  const dx = px - mean[0];
  const dy = py - mean[1];
  const m2 = (dx * (d * dx - b * dy) + dy * (-b * dx + a * dy)) / det;
  return Math.sqrt(Math.max(m2, 0));
}

/** Mixture theory density Σ prior·N(x) at one raw point (spec §2.4). */
export function gaussianMixturePdf(px: number, py: number, classes: ClassGaussian2D[]): number {
  if (classes.length === 0) return NaN;
  const n = classes.length;
  const priors = classes.map((g) =>
    g.prior !== null && Number.isFinite(g.prior) && g.prior > 0 ? g.prior : 1 / n,
  );
  const pSum = priors.reduce((s, p) => s + p, 0);
  let mix = 0;
  classes.forEach((g, k) => {
    const cov = covFromStdRotation(g.stdRaw, g.rotationRad);
    const lp = regularizedLogPdf(px, py, g.meanRaw, cov);
    if (Number.isFinite(lp)) mix += (priors[k]! / pSum) * Math.exp(Math.min(lp, 0));
  });
  return mix;
}

/** Bayes-paradox explainer LLR = ℓ(true) − ℓ(pred) (spec §2.3). */
export function logLikelihoodRatio(logPdfTrue: number, logPdfPred: number): number {
  if (!Number.isFinite(logPdfTrue) || !Number.isFinite(logPdfPred)) return NaN;
  return logPdfTrue - logPdfPred;
}

/**
 * Invert the world mapping X = ((x−xmin)/spanX)·10 − 5 back to raw units
 * (centres need raw coords for the Gaussian theory path).
 */
export function worldToRaw(w: number, min: number, max: number): number {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 1e-12) return NaN;
  return min + ((w + 5) / 10) * span;
}

/** Raw-space flat-top hex area from the world radius under anisotropic spans. */
export function hexAreaRaw(radiusWorld: number, spanX: number, spanY: number): number {
  if (!Number.isFinite(radiusWorld) || radiusWorld <= 0) return NaN;
  if (!Number.isFinite(spanX) || spanX <= 0 || !Number.isFinite(spanY) || spanY <= 0) return NaN;
  const rx = (radiusWorld / 10) * spanX;
  const ry = (radiusWorld / 10) * spanY;
  return ((3 * Math.sqrt(3)) / 2) * rx * ry;
}

/** Outer ring radius from the posterior confidence (spec §2.2). */
export function confidenceRingRadius(maxProba: number, base = RING_OUTER_RADIUS): number {
  return base * posteriorRingScale(maxProba);
}

/** SVG Y-flip helper for spoke segments. */
export function bayesSpokeToSvg(s: { x1: number; y1: number; x2: number; y2: number }): string {
  return `M${s.x1.toFixed(3)} ${wy(s.y1).toFixed(3)}L${s.x2.toFixed(3)} ${wy(s.y2).toFixed(3)}`;
}

export { hexPath };

/** Phase 55 — point-intelligence math for neural networks, pipelines & ONNX. */

import { wy } from '../DecisionBoundaryContour2D';
import { hexPath } from './pointIntelligence2dMath';

/** Saturated confident ring: Δz > 3.0 (spec §2.2). */
export const LOGIT_CONFIDENT_COLOR = '#22d3ee';
/** Tempting-tie neon halo: Δz < 0.5 (spec §2.2). */
export const LOGIT_DIFFUSE_COLOR = '#e879f9';
/** Decisive mid-gap ring tone. */
export const LOGIT_DECIDIVE_COLOR = '#94a3b8';
/** Dual-space trail tone (spec §2.1). */
export const DUAL_SPACE_TRAIL_COLOR = '#67e8f9';
/** Latent-geodesic trail tone for deep ONNX. */
export const LATENT_TRAIL_COLOR = '#a78bfa';
/** Adversarial gradient spoke tone (spec §2.3). */
export const LOSS_GRAD_COLOR = '#fb923c';
/** Data-desert hazard badge tone (spec §2.4). */
export const DESERT_HAZARD_COLOR = '#f43d5e';

/** Saturated-confidence floor (DoD #2, Δz > 3.0). */
export const LOGIT_CONFIDENT_GAP = 3.0;
/** Tie/diffuse ceiling (DoD #2, Δz < 0.5). */
export const LOGIT_DIFFUSE_GAP = 0.5;
/** Desert-hazard probability floor (DoD #3, Hazard > 0.85). */
export const DESERT_HAZARD_THRESHOLD = 0.85;
/** World-space cap for normalized gradient spokes (DoD #4 companion). */
export const GRAD_SPOKE_MAX_WORLD = 1.5;

export type LogitGapStatus = 'confident' | 'diffuse' | 'decisive' | 'unknown';

/** Exact affine map z = (x − mu) / sigma (spec §2.1, DoD #1). */
export function dualSpaceValue(value: number, mu: number, sigma: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(mu) || !Number.isFinite(sigma)) return NaN;
  if (Math.abs(sigma) <= 1e-12) return NaN;
  return (value - mu) / sigma;
}

/** Raw world tick → scaled world tick via the exact scaler frame (spec §2.1). */
export function dualSpaceWorld(
  w: number,
  rawMin: number,
  rawMax: number,
  mu: number,
  sigma: number,
  scaledMin: number,
  scaledMax: number,
): number {
  const raw = rawMin + ((w + 5) / 10) * (rawMax - rawMin);
  const z = dualSpaceValue(raw, mu, sigma);
  if (!Number.isFinite(z)) return NaN;
  const span = scaledMax - scaledMin;
  if (!(span > 1e-12)) return NaN;
  return ((z - scaledMin) / span) * 10 - 5;
}

/** Logit margin Δz = z_(1) − z_(2); single-logit binary heads map to |logit|. */
export function logitGap(logits: number[]): number {
  const fin = logits.filter((v) => Number.isFinite(v));
  if (fin.length === 0 || fin.length !== logits.length) return NaN;
  if (fin.length === 1) return Math.abs(fin[0] as number);
  const sorted = [...fin].sort((a, b) => (a as number) - (b as number));
  return (sorted[sorted.length - 1] as number) - (sorted[sorted.length - 2] as number);
}

/** Ring status from Δz: confident / diffuse / decisive / unknown. */
export function logitGapStatus(gap: number): LogitGapStatus {
  if (!Number.isFinite(gap)) return 'unknown';
  if (gap > LOGIT_CONFIDENT_GAP) return 'confident';
  if (gap < LOGIT_DIFFUSE_GAP) return 'diffuse';
  return 'decisive';
}

/** Recover the binary logit log(p / (1 − p)) with clipped p (DoD #2). */
export function binaryLogit(proba: number | null): number {
  if (proba === null || !Number.isFinite(proba)) return NaN;
  const p = Math.min(1 - 1e-12, Math.max(1e-12, proba as number));
  return Math.log(p / (1 - p));
}

/** Peak-normalized spoke lengths; peak ‖g‖ → GRAD_SPOKE_MAX_WORLD. */
export function gradSpokeLengths(norms: number[]): { lengths: number[]; live: boolean[] } {
  const mag = norms.map((r) => (Number.isFinite(r) ? Math.abs(r as number) : 0));
  const peak = mag.reduce((m, v) => Math.max(m, v), 0);
  if (!(peak > 1e-12)) {
    return { lengths: norms.map(() => 0), live: norms.map(() => false) };
  }
  return {
    lengths: norms.map((r) => (Number.isFinite(r) ? (Math.abs(r as number) / peak) * GRAD_SPOKE_MAX_WORLD : 0)),
    live: norms.map((r) => Number.isFinite(r) && Math.abs(r as number) > 1e-12),
  };
}

/** Unit direction g / ‖g‖; zero / non-finite rows collapse to (0, 0). */
export function gradUnit(gx: number, gy: number): { ux: number; uy: number } {
  if (!Number.isFinite(gx) || !Number.isFinite(gy)) return { ux: 0, uy: 0 };
  const n = Math.hypot(gx as number, gy as number);
  if (!(n > 1e-12)) return { ux: 0, uy: 0 };
  return { ux: (gx as number) / n, uy: (gy as number) / n };
}

/** Hazard(u,v) = 1[count==0] · max P; desert badge when Hazard > 0.85. */
export function desertHazard(count: number, maxProba: number | null): { hazard: number; isDesert: boolean } {
  if (maxProba === null || !Number.isFinite(maxProba) || !Number.isFinite(count)) {
    return { hazard: NaN, isDesert: false };
  }
  const p = Math.min(1, Math.max(0, maxProba as number));
  const hazard = count <= 0 ? p : 0;
  return { hazard, isDesert: Number.isFinite(hazard) && hazard > DESERT_HAZARD_THRESHOLD };
}

/** Mean |activation| tone for the diffuse halo (spec §2.2). */
export function haloTone(activations: number[]): number {
  const fin = activations.filter((v) => Number.isFinite(v)).map((v) => Math.abs(v as number));
  if (fin.length === 0) return NaN;
  return fin.reduce((s, v) => s + (v as number), 0) / fin.length;
}

/** SVG Y-flip helper for gradient spoke segments. */
export function gradSpokeToSvg(s: { x1: number; y1: number; x2: number; y2: number }): string {
  return `M${s.x1.toFixed(3)} ${wy(s.y1).toFixed(3)}L${s.x2.toFixed(3)} ${wy(s.y2).toFixed(3)}`;
}

export { hexPath };

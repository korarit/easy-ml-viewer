/** Phase 54 — point-intelligence math for gradient boosting heavyweights. */

import { wy } from '../DecisionBoundaryContour2D';
import { hexPath } from './pointIntelligence2dMath';

/** Safe margin: solid class-color ring (spec §2.2, M > 1.5). */
export const MARGIN_SAFE_COLOR = '#22c55e';
/** Critical margin: pulsing orange ring (spec §2.2, 0 ≤ M ≤ 1.5). */
export const MARGIN_CRITICAL_COLOR = '#fb923c';
/** Negative margin: red cross — boosting's next-round target (spec §2.2). */
export const MARGIN_NEGATIVE_COLOR = '#f43d5e';
/** Staged SHAP push trail tone (spec §2.1). */
export const PUSH_TRAIL_COLOR = '#f43d5e';
/** Pseudo-residual spokes: up (needs push) vs down (needs pull). */
export const GRAD_UP_COLOR = '#22c55e';
export const GRAD_DOWN_COLOR = '#fb923c';
/** Quantized hot cell (hard region, high |Σg|) vs lattice traffic. */
export const QUANT_HOT_COLOR = '#f43d5e';
export const QUANT_TRAFFIC_COLOR = '#38bdf8';

/** Functional-margin safe threshold (DoD #2, M > 1.5). */
export const MARGIN_SAFE_THRESHOLD = 1.5;
/** World-space cap for normalized pseudo-residual spokes (DoD #3). */
export const SPOKE_MAX_WORLD = 1.5;
/** World units per unit |phi| before clamping (mirrors SHAP_ARROW_SCALE). */
export const PUSH_WORLD_SCALE = 0.5;
/** Clamp for the whole staged walk on the ±5 stage. */
export const PUSH_WALK_MAX_LEN = 2.5;

export type MarginStatus = 'safe' | 'critical' | 'negative' | 'unknown';

/** Staged walk P_0=0, P_m = Σ_{k≤m} φ_k (spec §2.1, NaN stages step zero). */
export function cumulativePush(stages: Array<{ phiX: number; phiY: number }>): Array<{ x: number; y: number }> {
  const path: Array<{ x: number; y: number }> = [{ x: 0, y: 0 }];
  let cx = 0;
  let cy = 0;
  for (const s of stages) {
    const ok = Number.isFinite(s.phiX) && Number.isFinite(s.phiY);
    if (ok) {
      cx += s.phiX as number;
      cy += s.phiY as number;
    }
    path.push({ x: cx, y: cy });
  }
  return path;
}

/** |φ_m| per stage; NaN stages stay NaN (hardest-stage finder). */
export function pushStepMagnitudes(stages: Array<{ phiX: number; phiY: number }>): number[] {
  return stages.map((s) =>
    Number.isFinite(s.phiX) && Number.isFinite(s.phiY) ? Math.hypot(s.phiX, s.phiY) : NaN,
  );
}

/** World-per-phi factor so the whole staged walk fits the stage. */
export function pushWorldScale(stages: Array<{ phiX: number; phiY: number }>): number {
  const total = stages.reduce(
    (s, v) => s + (Number.isFinite(v.phiX) && Number.isFinite(v.phiY) ? Math.hypot(v.phiX, v.phiY) : 0),
    0,
  );
  if (!(total > 1e-12)) return PUSH_WORLD_SCALE;
  return Math.min(PUSH_WORLD_SCALE, PUSH_WALK_MAX_LEN / total);
}

/** Map a label to +1/−1 against the positive class (spec §2.2). */
export function toSignedLabel(label: string | number | null, positiveLabel: string | number): number {
  if (label === null || label === undefined) return NaN;
  if (typeof label === 'number' && !Number.isFinite(label)) return NaN;
  return String(label) === String(positiveLabel) ? 1 : -1;
}

/** Functional margin M = y·F(x) (spec §2.2, DoD #2). */
export function functionalMargin(ySigned: number, fx: number): number {
  if (!Number.isFinite(ySigned) || !Number.isFinite(fx) || Math.abs(ySigned) <= 1e-12) return NaN;
  return ySigned * fx;
}

/** Ring status from M: safe / critical / negative / unknown. */
export function marginStatus(margin: number): MarginStatus {
  if (!Number.isFinite(margin)) return 'unknown';
  if (margin > MARGIN_SAFE_THRESHOLD) return 'safe';
  if (margin >= 0) return 'critical';
  return 'negative';
}

/** Least-squares pseudo-residual r = y − F (spec §2.3). */
export function pseudoResidualLS(yTrue: number | null, fx: number | null): number {
  if (yTrue === null || fx === null) return NaN;
  if (!Number.isFinite(yTrue) || !Number.isFinite(fx)) return NaN;
  return yTrue - fx;
}

/** Binary-deviance pseudo-residual r = y − p with clipped p (spec §2.3). */
export function pseudoResidualLogloss(y01: number | null, proba: number | null): number {
  if (y01 === null || proba === null) return NaN;
  if (!Number.isFinite(y01) || !Number.isFinite(proba)) return NaN;
  const p = Math.min(1 - 1e-12, Math.max(1e-12, proba));
  const y = Math.min(1, Math.max(0, y01));
  return y - p;
}

/** Peak-normalized spoke lengths; peak |r| → SPOKE_MAX_WORLD (DoD #3). */
export function spokeLengths(residuals: number[]): { lengths: number[]; sides: number[] } {
  const mag = residuals.map((r) => (Number.isFinite(r) ? Math.abs(r as number) : 0));
  const peak = mag.reduce((m, v) => Math.max(m, v), 0);
  if (!(peak > 1e-12)) {
    return { lengths: residuals.map(() => 0), sides: residuals.map(() => 0) };
  }
  return {
    lengths: residuals.map((r, i) => (Number.isFinite(r) ? (mag[i]! / peak) * SPOKE_MAX_WORLD : 0)),
    sides: residuals.map((r) =>
      !Number.isFinite(r) ? 0 : r! > 1e-12 ? 1 : r! < -1e-12 ? -1 : 0,
    ),
  };
}

/** Snap one raw value to the quantized lattice (searchsorted-left). */
export function quantizeToBin(value: number, edges: number[]): number {
  if (!Number.isFinite(value) || edges.length === 0) return -1;
  for (let i = 0; i < edges.length; i += 1) {
    if (value <= (edges[i] as number)) return i;
  }
  return edges.length;
}

/** Flat quantized address bx·B + by; −1 where unknown (−1 mirrors backend). */
export function quantizedCellId(binX: number, binY: number, nBins = 256): number {
  if (!Number.isInteger(binX) || !Number.isInteger(binY)) return -1;
  if (binX < 0 || binY < 0 || binX >= nBins || binY >= nBins) return -1;
  return binX * nBins + binY;
}

/** SVG Y-flip helper for spoke segments. */
export function boostSpokeToSvg(s: { x1: number; y1: number; x2: number; y2: number }): string {
  return `M${s.x1.toFixed(3)} ${wy(s.y1).toFixed(3)}L${s.x2.toFixed(3)} ${wy(s.y2).toFixed(3)}`;
}

export { hexPath };

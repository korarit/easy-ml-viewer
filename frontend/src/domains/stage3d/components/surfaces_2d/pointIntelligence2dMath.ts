/** Phase 49 — shared math for the Universal 2D Point Intelligence engine. */

import { wy } from '../DecisionBoundaryContour2D';

/** Ghost-dot constants (spec §2.1): out-of-window frames collapse here. */
export const GHOST_ALPHA = 0.15;
export const GHOST_RADIUS = 0.06;

/** Dual-ring geometry in world units (spec §2.2). */
export const RING_INNER_RADIUS = 0.08;
export const RING_OUTER_RADIUS = 0.14;
export const RING_THICKNESS = 0.04;
export const RING_CONTRAST_WIDTH = 0.025;

export const COMET_ALPHA_MIN = 0.05;
export const COMET_ALPHA_MAX = 1.0;

export type GlyphShape = 'circle' | 'rect' | 'polygon';

/** Gaussian width σ = W / 2.5 (spec §2.1). */
export function cometSigma(window: number): number {
  return Math.max(1, Math.round(window)) / 2.5;
}

/** Opacity weight α(f) for frame f given the current frame t. */
export function cometAlpha(frame: number, current: number, window: number): number {
  const sigma = cometSigma(window);
  const d = current - frame;
  const decay = Math.exp(-(d * d) / (2 * sigma * sigma));
  return COMET_ALPHA_MIN + (COMET_ALPHA_MAX - COMET_ALPHA_MIN) * decay;
}

/** Per-frame opacities for a queue of n frames; out-of-window → ghost. */
export function cometTrailAlphas(nFrames: number, current: number, window: number): number[] {
  const n = Math.max(0, Math.round(nFrames));
  const w = Math.max(1, Math.round(window));
  const out: number[] = [];
  for (let f = 0; f < n; f += 1) {
    if (f < current - w || f > current + w) out.push(GHOST_ALPHA);
    else if (f <= current) out.push(Number(cometAlpha(f, current, w).toFixed(4)));
    else out.push(Number(cometAlpha(2 * current - f, current, w).toFixed(4)));
  }
  return out;
}

/** Trail stroke width growing toward the head frame (spec §2.1). */
export function cometStrokeWidth(frame: number, current: number, window: number, base = 0.03): number {
  const w = Math.max(1, Math.round(window));
  const t = Math.min(1, Math.max(0, (frame - (current - w)) / w));
  return Number((base * (1 + 2 * t)).toFixed(4));
}

/** High-legibility perimeter color (spec §2.2). */
export function contrastStroke(isLight: boolean): string {
  return isLight ? '#0b0f14' : '#ffffff';
}

/** Indices of mispredicted frames: label mismatch or |residual| > eps. */
export function errorsOnlyFrames(
  residuals: Array<number | null>,
  eps = 0.5,
): number[] {
  const out: number[] = [];
  for (let f = 0; f < residuals.length; f += 1) {
    const r = residuals[f];
    if (r === null || r === undefined || !Number.isFinite(r)) {
      out.push(f);
    } else if (Math.abs(r) > eps) {
      out.push(f);
    }
  }
  return out;
}

/** 2D error-spoke endpoint from (x, y) along angle scaled by |residual|. */
export function residualSpokeEnd(
  x: number,
  y: number,
  residual: number,
  angleRad = Math.PI / 2,
  scale = 0.25,
): [number, number] {
  const e = Number.isFinite(residual) ? residual : 0;
  const len = Math.min(2.5, Math.max(0, Math.abs(e) * scale));
  const sgn = e >= 0 ? 1 : -1;
  return [x + Math.cos(angleRad) * len * sgn, y + Math.sin(angleRad) * len * sgn];
}

/** Flat-top hexagon vertices in SVG coords (Y flipped once, here). */
export function hexVertices(cx: number, cy: number, r: number): Array<[number, number]> {
  const pts: Array<[number, number]> = [];
  for (let k = 0; k < 6; k += 1) {
    const a = (Math.PI / 3) * k + Math.PI / 6;
    pts.push([cx + r * Math.cos(a), wy(cy + r * Math.sin(a))]);
  }
  return pts;
}

/** One hexagon as an SVG path string (groupable into batched paths). */
export function hexPath(cx: number, cy: number, r: number): string {
  const v = hexVertices(cx, cy, r);
  return `M${v.map(([x, y]) => `${x.toFixed(3)} ${y.toFixed(3)}`).join('L')}Z`;
}

/** Fill color for a hexbin cell under the active display mode. */
export function hexbinFill(
  mode: 'count' | 'error' | 'class',
  cell: { count: number; maxBinCount: number; errorRate: number; dominantClass: number | null },
  classColor: (cls: number | null) => string,
): string {
  if (mode === 'error') {
    // White → amber → red ramp by misclassification concentration.
    const t = Math.min(1, Math.max(0, cell.errorRate));
    const r = 255;
    const g = Math.round(255 - 120 * t);
    const b = Math.round(237 - 180 * t);
    return `rgb(${r},${g},${b})`;
  }
  if (mode === 'class') return classColor(cell.dominantClass);
  // Density ramp: deep slate → cyan by sqrt(count / max).
  const denom = Math.max(1, cell.maxBinCount);
  const t = Math.sqrt(cell.count / denom);
  const r = Math.round(15 + 40 * t);
  const g = Math.round(23 + 160 * t);
  const b = Math.round(42 + 170 * t);
  return `rgb(${r},${g},${b})`;
}

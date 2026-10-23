/** Phase 50 — point-intelligence math for linear & generalized-linear models. */

import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { wy } from '../DecisionBoundaryContour2D';

/** Trail ascent colors: toward positive (green) vs negative (orange-red). */
export const TRAIL_UP_COLOR = '#22c55e';
export const TRAIL_DOWN_COLOR = '#fb923c';
export const TRAIL_FLAT_COLOR = '#22d3ee';

/** Amber pulsing ring for margin-corridor violators. */
export const MARGIN_RING_COLOR = '#fbbf24';

export interface WorldLine {
  W1: number;
  W2: number;
  B: number;
  degenerate: boolean;
}

/**
 * Convert the raw-space plane `w1·x + w2·y + b = 0` to world coords.
 *
 * World mapping is per-axis affine (`X = ((x−xmin)/spanX)·10 − 5`), so the
 * line stays linear: `W1 = w1·spanX/10`, `W2 = w2·spanY/10`,
 * `B = b + w1·midX + w2·midY`.
 */
export function worldLine(
  w1: number,
  w2: number,
  bEff: number,
  xRangeRaw: [number, number],
  yRangeRaw: [number, number],
): WorldLine {
  const spanX = xRangeRaw[1] - xRangeRaw[0];
  const spanY = yRangeRaw[1] - yRangeRaw[0];
  if (!Number.isFinite(spanX) || !Number.isFinite(spanY) || spanX <= 1e-12 || spanY <= 1e-12) {
    return { W1: 0, W2: 0, B: 0, degenerate: true };
  }
  const W1 = w1 * (spanX / 10);
  const W2 = w2 * (spanY / 10);
  const B = bEff + w1 * (xRangeRaw[0] + spanX / 2) + w2 * (yRangeRaw[0] + spanY / 2);
  const degenerate = Math.hypot(W1, W2) <= 1e-12;
  return { W1, W2, B, degenerate };
}

/** Raw-space decision score `z = w1·x + w2·y + b`. */
export function zScoreRaw(x: number, y: number, w1: number, w2: number, b: number): number {
  return w1 * x + w2 * y + b;
}

/** Numerically stable sigmoid (log-odds → P(y=1)). */
export function expit(z: number): number {
  if (!Number.isFinite(z)) return 0.5;
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Per-segment trail ascent signs `Δz = w12ᵀ(pf − pf−1)` (raw space). */
export function trailSigns(
  points: Array<{ x: number; y: number }>,
  w1: number,
  w2: number,
): number[] {
  const out: number[] = [];
  for (let f = 1; f < points.length; f += 1) {
    const dz = (points[f]!.x - points[f - 1]!.x) * w1 + (points[f]!.y - points[f - 1]!.y) * w2;
    out.push(dz > 1e-12 ? 1 : dz < -1e-12 ? -1 : 0);
  }
  return out;
}

/** Segment color from its ascent sign. */
export function trailSignColor(sign: number, fallback = TRAIL_FLAT_COLOR): string {
  if (sign > 0) return TRAIL_UP_COLOR;
  if (sign < 0) return TRAIL_DOWN_COLOR;
  return fallback;
}

/** Margin-corridor hit: `|z| ≤ 1` (DoD #2). */
export function isMarginViolator(z: number, level = 1.0): boolean {
  return Number.isFinite(z) && Math.abs(z) <= level;
}

/** Logistic confidence-ring thickness (spec §2.2): thick at P≈0.5. */
export function logisticRingThickness(proba: number): number {
  const p = Number.isFinite(proba) ? Math.min(1, Math.max(0, proba)) : 0.5;
  return 0.02 + 0.06 * (1 - 2 * Math.abs(p - 0.5));
}

export interface OrthogonalSpoke {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  dPerp: number;
}

/**
 * Orthogonal projection of a world point onto the world line (spec §2.3).
 * Returns null for degenerate lines — no spoke instead of a wrong one.
 */
export function orthogonalSpoke(px: number, py: number, line: WorldLine): OrthogonalSpoke | null {
  if (line.degenerate) return null;
  const denom = line.W1 * line.W1 + line.W2 * line.W2;
  if (denom <= 1e-24) return null;
  const signed = (line.W1 * px + line.W2 * py + line.B) / denom;
  const ex = px - signed * line.W1;
  const ey = py - signed * line.W2;
  const dPerp = Math.abs(line.W1 * px + line.W2 * py + line.B) / Math.sqrt(denom);
  return { x1: px, y1: py, x2: ex, y2: ey, dPerp };
}

/** World-space perpendicular distance (for hexbin centers). */
export function worldDistToLine(cx: number, cy: number, line: WorldLine): number {
  const norm = Math.hypot(line.W1, line.W2);
  if (norm <= 1e-12) return NaN;
  return Math.abs(line.W1 * cx + line.W2 * cy + line.B) / norm;
}

/** Flag hexbins intersected by the decision line (spec §2.4). */
export function hexbinBoundaryFlags(cells: HexbinCell2D[], line: WorldLine, inflate = 1.1): boolean[] {
  if (line.degenerate) return cells.map(() => false);
  return cells.map((c) => worldDistToLine(c.centerX, c.centerY, line) <= c.radius * inflate);
}

/** Pearson r between bin-centre distance and bin error rate (spec §2.4). */
export function errorDistanceCorrelation(distances: number[], errorRates: number[]): number {
  const pairs = distances
    .map((d, i) => [d, errorRates[i]!] as const)
    .filter(([d, e]) => Number.isFinite(d) && Number.isFinite(e));
  if (pairs.length < 3) return NaN;
  const md = pairs.reduce((s, [d]) => s + d, 0) / pairs.length;
  const me = pairs.reduce((s, [, e]) => s + e, 0) / pairs.length;
  let cov = 0;
  let vd = 0;
  let ve = 0;
  for (const [d, e] of pairs) {
    cov += (d - md) * (e - me);
    vd += (d - md) * (d - md);
    ve += (e - me) * (e - me);
  }
  if (vd <= 1e-24 || ve <= 1e-24) return NaN;
  return cov / Math.sqrt(vd * ve);
}

/** SVG Y-flip helper for spoke/stem segments. */
export function segToSvg(s: { x1: number; y1: number; x2: number; y2: number }): string {
  return `M${s.x1.toFixed(3)} ${wy(s.y1).toFixed(3)}L${s.x2.toFixed(3)} ${wy(s.y2).toFixed(3)}`;
}

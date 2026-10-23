/** Phase 51 — point-intelligence math for distance & kernel models (KNN & SVM). */

import { wy } from '../DecisionBoundaryContour2D';
import { hexPath } from './pointIntelligence2dMath';

/** Free SV on the margin edge: cyan diamond (spec §2.2, bounded 0<|α|<C). */
export const SV_HALO_ON_MARGIN_COLOR = '#7dd3fc';
/** Bound SV violating the margin: orange diamond (spec §2.2, unbounded |α|≈C). */
export const SV_HALO_VIOLATES_COLOR = '#fb923c';
/** Neutral support halo when C is unknown. */
export const SV_HALO_SUPPORT_COLOR = '#94a3b8';

/** Slack colors: ξ (actual above tube) vs ξ* (pred above tube). */
export const SVR_SLACK_XI_COLOR = '#fb923c';
export const SVR_SLACK_XI_STAR_COLOR = '#f43f5e';

/** KNN conflict discordance spoke (spec §2.3). */
export const KNN_DISCORDANCE_COLOR = '#ef4444';

export interface KnnNeighborPoint {
  x: number;
  y: number;
  voteWeight: number;
  label?: string | number | null;
  refIndex?: number;
}

/** Weighted centroid x̄_K of the K neighbors (elastic-web anchor, spec §2.1). */
export function knnCentroid2D(neighbors: KnnNeighborPoint[]): { x: number; y: number } | null {
  if (neighbors.length === 0) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const n of neighbors) {
    if (!Number.isFinite(n.x) || !Number.isFinite(n.y)) continue;
    const w = Number.isFinite(n.voteWeight) && n.voteWeight > 0 ? n.voteWeight : 0;
    sw += w;
    sx += w * n.x;
    sy += w * n.y;
  }
  if (sw <= 1e-12) {
    const xs = neighbors.filter((n) => Number.isFinite(n.x)).map((n) => n.x);
    const ys = neighbors.filter((n) => Number.isFinite(n.y)).map((n) => n.y);
    if (xs.length === 0 || ys.length === 0) return null;
    return {
      x: xs.reduce((a, b) => a + b, 0) / xs.length,
      y: ys.reduce((a, b) => a + b, 0) / ys.length,
    };
  }
  return { x: sx / sw, y: sy / sw };
}

/** Swap sets between consecutive neighbor id lists (spec §2.1). */
export function detectNeighborSwaps(
  prevIds: number[],
  curIds: number[],
): { entered: number[]; exited: number[] } {
  const prev = new Set(prevIds);
  const cur = new Set(curIds);
  return {
    entered: [...cur].filter((i) => !prev.has(i)).sort((a, b) => a - b),
    exited: [...prev].filter((i) => !cur.has(i)).sort((a, b) => a - b),
  };
}

/** Halo status → diamond color (DoD #1). */
export function haloColor(status: string): string {
  if (status === 'violates_margin') return SV_HALO_VIOLATES_COLOR;
  if (status === 'on_margin') return SV_HALO_ON_MARGIN_COLOR;
  return SV_HALO_SUPPORT_COLOR;
}

/** Diamond radius from the |α|/C halo scale (mirrors Phase-43 sizing). */
export function haloDiamondRadius(scale: number): number {
  const t = Math.min(1, Math.max(0, Number.isFinite(scale) ? scale : 0));
  return 0.1 + 0.16 * t;
}

/** One diamond glyph as an SVG path (world coords, Y flipped once here). */
export function diamondPath(cx: number, cyWorld: number, r: number): string {
  const y = wy(cyWorld);
  return (
    `M${(cx - r).toFixed(3)} ${y.toFixed(3)}` +
    `L${cx.toFixed(3)} ${(y - r).toFixed(3)}` +
    `L${(cx + r).toFixed(3)} ${y.toFixed(3)}` +
    `L${cx.toFixed(3)} ${(y + r).toFixed(3)}Z`
  );
}

export interface VoteArcSegment {
  label: string;
  share: number;
  /** Fraction of a full turn where this segment starts. */
  start: number;
  /** Fraction of a full turn swept (Σ sweep === 1). */
  sweep: number;
  color: string;
}

/** Vote-breakdown pie shares from a votes tally (DoD #3, spec §2.2). */
export function voteArcSegments(
  votes: Record<string, number>,
  colorFor: (label: string, rank: number) => string,
): VoteArcSegment[] {
  const entries = Object.entries(votes).filter(([, w]) => Number.isFinite(w) && w > 0);
  const total = entries.reduce((s, [, w]) => s + w, 0);
  if (entries.length === 0 || total <= 1e-12) return [];
  const ranked = entries.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  let start = 0;
  return ranked.map(([label, w], rank) => {
    const share = w / total;
    const seg = { label, share, start, sweep: share, color: colorFor(label, rank) };
    start += share;
    return seg;
  });
}

/** Donut arc for one vote segment (SVG coords — caller flips Y once). */
export function arcDonutPath(
  cx: number,
  cySvg: number,
  rOuter: number,
  rInner: number,
  startTurn: number,
  sweepTurn: number,
): string {
  const a0 = startTurn * Math.PI * 2 - Math.PI / 2;
  const a1 = (startTurn + Math.max(sweepTurn, 1e-6)) * Math.PI * 2 - Math.PI / 2;
  const large = sweepTurn > 0.5 ? 1 : 0;
  const p = (r: number, a: number): string => `${(cx + r * Math.cos(a)).toFixed(3)} ${(cySvg + r * Math.sin(a)).toFixed(3)}`;
  // Full-turn guard: a single arc cannot close 360°, split into two halves.
  if (sweepTurn >= 1 - 1e-9) {
    const mid = a0 + Math.PI;
    return `M${p(rOuter, a0)}A${rOuter.toFixed(3)} ${rOuter.toFixed(3)} 0 1 1 ${p(rOuter, mid)}A${rOuter.toFixed(3)} ${rOuter.toFixed(3)} 0 1 1 ${p(rOuter, a0)}Z`;
  }
  return (
    `M${p(rOuter, a0)}A${rOuter.toFixed(3)} ${rOuter.toFixed(3)} 0 ${large} 1 ${p(rOuter, a1)}` +
    `L${p(rInner, a1)}A${rInner.toFixed(3)} ${rInner.toFixed(3)} 0 ${large} 0 ${p(rInner, a0)}Z`
  );
}

export interface SvrSlackSpoke {
  xi: number;
  xiStar: number;
  outside: boolean;
}

/** SVR ε-tube slack for one point (DoD #2, spec §2.3). */
export function svrSlackSpoke(yTrue: number | null, yPred: number, epsilon: number): SvrSlackSpoke {
  const eps = Number.isFinite(epsilon) && epsilon >= 0 ? epsilon : 0;
  if (yTrue === null || !Number.isFinite(yTrue) || !Number.isFinite(yPred)) {
    return { xi: 0, xiStar: 0, outside: false };
  }
  const xi = Math.max(0, yTrue - yPred - eps);
  const xiStar = Math.max(0, yPred - yTrue - eps);
  return { xi, xiStar, outside: Math.abs(yTrue - yPred) > eps + 1e-12 };
}

/** Local manifold KNN density ρ = K / (π·R̄_K²) (spec §2.4). */
export function knnDensity(k: number, meanRadius: number): number {
  if (!Number.isFinite(k) || !Number.isFinite(meanRadius) || k <= 0 || meanRadius <= 1e-12) {
    return NaN;
  }
  return k / (Math.PI * meanRadius * meanRadius);
}

/** Neighbor ranks voting against the true class (spec §2.3 discordance). */
export function discordantRanks(
  neighbors: Array<{ rank: number; label: string | number | null }>,
  trueLabel: string | number | null,
): number[] {
  const truth = String(trueLabel);
  return neighbors.filter((n) => String(n.label) !== truth).map((n) => n.rank);
}

/** Tier hexbin cells by density into dense-island / sparse-desert paths. */
export function densityTiers(
  densities: Array<number | null>,
): { dense: number[]; sparse: number[] } {
  const finite = densities
    .map((d, i) => ({ d, i }))
    .filter((e): e is { d: number; i: number } => typeof e.d === 'number' && Number.isFinite(e.d));
  if (finite.length === 0) return { dense: [], sparse: [] };
  const sorted = [...finite].sort((a, b) => a.d - b.d);
  const cut = Math.max(1, Math.floor(sorted.length / 3));
  return {
    dense: sorted.slice(-cut).map((e) => e.i),
    sparse: sorted.slice(0, cut).map((e) => e.i),
  };
}

/** SVG Y-flip helper for spoke segments (mirrors linear segToSvg). */
export function spokeToSvg(s: { x1: number; y1: number; x2: number; y2: number }): string {
  return `M${s.x1.toFixed(3)} ${wy(s.y1).toFixed(3)}L${s.x2.toFixed(3)} ${wy(s.y2).toFixed(3)}`;
}

export { hexPath };

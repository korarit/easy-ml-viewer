/** Phase 53 — point-intelligence math for trees & bagging ensembles. */

import { wy } from '../DecisionBoundaryContour2D';
import { hexPath } from './pointIntelligence2dMath';
import { RING_OUTER_RADIUS } from './pointIntelligence2dMath';

/** Unanimous forest: thin solid ring (spec §2.2, σ = 0). */
export const CONSENSUS_SOLID_COLOR = '#22d3ee';
/** Split vote below 60%: double-dashed fuzzy ring (DoD #1). */
export const CONSENSUS_FUZZY_COLOR = '#f59e0b';
/** AdaBoost halo scales with √ω (spec §2.2, DoD #2). */
export const ADABOOST_HALO_COLOR = '#c084fc';
/** Minority strand inside an impure leaf (spec §2.3, DoD #3). */
export const LEAF_MINORITY_COLOR = '#f43d5e';
/** Pure leaf block vs mixed-leaf warning (spec §2.4). */
export const LEAF_PURE_COLOR = '#22c55e';
export const LEAF_MIXED_COLOR = '#f59e0b';

/** Agreement floor: rings turn double-dashed below 60% (DoD #1). */
export const CONSENSUS_FUZZY_THRESHOLD = 0.6;

/** Elbow vertex of the L-shaped orthogonal hop prev → cur (spec §2.1). */
export function manhattanElbow(
  prev: { x: number; y: number },
  cur: { x: number; y: number },
  xFirst = true,
): { x: number; y: number } | null {
  if (![prev.x, prev.y, cur.x, cur.y].every(Number.isFinite)) return null;
  return xFirst ? { x: cur.x, y: prev.y } : { x: prev.x, y: cur.y };
}

/** L1 hop length |dx| + |dy| (spec §2.1). */
export function manhattanLength(
  prev: { x: number; y: number },
  cur: { x: number; y: number },
): number {
  if (![prev.x, prev.y, cur.x, cur.y].every(Number.isFinite)) return NaN;
  return Math.abs(cur.x - prev.x) + Math.abs(cur.y - prev.y);
}

/** One batched orthogonal trail path through world points (Y flipped once). */
export function manhattanTrailPath(
  points: Array<{ x: number; y: number }>,
  lo: number,
  hi: number,
  xFirst = true,
): string {
  let d = '';
  const n = points.length;
  const a = Math.max(0, lo);
  const b = Math.min(hi, n - 1);
  for (let f = a; f < b; f += 1) {
    const p = points[f];
    const c = points[f + 1];
    if (!p || !c) continue;
    const e = manhattanElbow(p, c, xFirst);
    if (!e) continue;
    d += `M${p.x.toFixed(3)} ${wy(p.y).toFixed(3)}L${e.x.toFixed(3)} ${wy(e.y).toFixed(3)}L${c.x.toFixed(3)} ${wy(c.y).toFixed(3)}`;
  }
  return d;
}

/** Per-point agreement max_c v_c over T tree votes (spec §2.1). */
export function forestAgreement(votes: Array<Array<string | number>>): number[] {
  return votes.map((row) => {
    const labels = row.map(String).filter((s) => s !== 'nan' && s !== 'None' && s !== '');
    if (labels.length === 0) return NaN;
    const tally = new Map<string, number>();
    for (const l of labels) tally.set(l, (tally.get(l) ?? 0) + 1);
    return Math.max(...tally.values()) / labels.length;
  });
}

/** True where the ring must render double-dashed (DoD #1: agreement < 60%). */
export function isFuzzyConsensus(maxShare: number, threshold = CONSENSUS_FUZZY_THRESHOLD): boolean {
  if (!Number.isFinite(maxShare)) return false;
  return maxShare < threshold;
}

/** Ring thickness 1.0 (unanimous) → 2.0 (coin flip) (spec §2.2). */
export function consensusThickness(maxShare: number): number {
  if (!Number.isFinite(maxShare)) return 1;
  const t = Math.min(1, Math.max(0, maxShare));
  return 1 + (1 - t);
}

/** Outer-ring scale t_base·√(w/w̄) for one AdaBoost sample weight (DoD #2). */
export function adaboostRingScale(weight: number, meanWeight: number, base = 1): number {
  const b = Number.isFinite(base) ? base : 1;
  if (!Number.isFinite(weight) || weight < 0 || !Number.isFinite(meanWeight) || meanWeight <= 1e-12) {
    return b;
  }
  return b * Math.sqrt(weight / meanWeight);
}

/** True where a point disagrees with its leaf majority (spec §2.3). */
export function isLeafMinority(yTrue: number | null, leafPred: number | string | null): boolean {
  if (yTrue === null || leafPred === null) return true;
  const t = Number(yTrue);
  const p = Number(leafPred);
  if (!Number.isFinite(t) || !Number.isFinite(p)) return true;
  return Math.round(t) !== Math.round(p);
}

/** Regressor residual e = y − y_leaf vs the leaf constant (spec §2.3). */
export function stepwiseResidual(yTrue: number | null, yLeaf: number | null): number {
  if (yTrue === null || yLeaf === null) return NaN;
  if (!Number.isFinite(yTrue) || !Number.isFinite(yLeaf)) return NaN;
  return yTrue - yLeaf;
}

/** Hexbin purity max_c count / total (spec §2.4). */
export function hexPurity(dominantCount: number, totalCount: number): number {
  if (!Number.isFinite(dominantCount) || !Number.isFinite(totalCount)) return NaN;
  if (!(totalCount > 1e-12) || dominantCount < 0) return NaN;
  return Math.min(1, Math.max(0, dominantCount / totalCount));
}

/** Outer ring radius from the AdaBoost scale (spec §2.2). */
export function adaboostHaloRadius(scale: number, base = RING_OUTER_RADIUS): number {
  const s = Number.isFinite(scale) && scale > 0 ? scale : 1;
  return base * s;
}

/** SVG Y-flip helper for spoke segments. */
export function treeSpokeToSvg(s: { x1: number; y1: number; x2: number; y2: number }): string {
  return `M${s.x1.toFixed(3)} ${wy(s.y1).toFixed(3)}L${s.x2.toFixed(3)} ${wy(s.y2).toFixed(3)}`;
}

export { hexPath };

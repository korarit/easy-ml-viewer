/**
 * Phase 34 — pure math helpers for the LightGBM leaf-wise asymmetric stage.
 *
 * Dependency-free (no React / Three imports) so SHAP force-vector
 * scaling, staged heights, depth tints and leaf-path lookups stay
 * unit-testable.
 */

/** Sigmoid link (binary raw score → probability), overflow-safe. */
export function lgbmSigmoid(z: number): number {
  if (!Number.isFinite(z)) return 0;
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** First 1-based iteration where the staged linked prediction crosses threshold. */
export function crossedLgbmRound(trajectory: number[], threshold = 0.5): number | null {
  for (let m = 0; m < trajectory.length; m += 1) {
    if (trajectory[m] >= threshold) return m + 1;
  }
  return null;
}

/** Normalize a regression terraces height into the display range [0, 2]. */
export function normalizeLgbmHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function lgbmPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Map a world value on an axis grid into the [-2.5, 2.5] stage space. */
export function lgbmGridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

/** Leaf-depth heat tint: shallow leaves are pale gold, deep leaf-wise
 *  micro-leaves ramp toward burnt orange (asymmetric-resolution cue). */
export function lgbmDepthShade(depth: number, maxDepth: number): number {
  if (![depth, maxDepth].every(Number.isFinite) || maxDepth <= 0) return 0;
  return Math.min(1, Math.max(0, depth / maxDepth));
}

export interface LgbmFlatNode {
  id: number;
  leaf_index?: number;
  leaf_value?: number;
  leaf_count?: number;
  split_feature_idx?: number;
  threshold?: number | null;
  categories?: number[];
  gain?: number;
  depth?: number;
  yes?: number;
  no?: number;
}

/** Build a child→parent lookup from an id-linked flat node list. */
export function lgbmParentMap(nodes: LgbmFlatNode[]): Map<number, number> {
  const parent = new Map<number, number>();
  for (const n of nodes) {
    if (n.leaf_index !== undefined) continue;
    if (typeof n.yes === 'number' && n.yes >= 0) parent.set(n.yes, n.id);
    if (typeof n.no === 'number' && n.no >= 0) parent.set(n.no, n.id);
  }
  return parent;
}

/** Walk from a node id back to the root, returning the node-id path leaf→root. */
export function lgbmLeafPath(leafId: number, nodes: LgbmFlatNode[]): number[] {
  const parent = lgbmParentMap(nodes);
  const path: number[] = [leafId];
  const seen = new Set<number>([leafId]);
  let cur: number | undefined = leafId;
  for (let i = 0; i < 512; i += 1) {
    const p: number | undefined = cur === undefined ? undefined : parent.get(cur);
    if (p === undefined || seen.has(p)) break;
    path.push(p);
    seen.add(p);
    cur = p;
    if (p === 0) break;
  }
  return path;
}

/** Scale a SHAP value into a force-arrow length (display units). */
export function lgbmForceLength(shap: number, maxAbs: number): number {
  if (![shap, maxAbs].every(Number.isFinite) || maxAbs <= 0) return 0;
  return Math.min(1.2, (Math.abs(shap) / maxAbs) * 1.2);
}

/** Force-arrow color: amber push (positive SHAP) vs rose pull (negative). */
export function lgbmForceColor(shap: number): [number, number, number] {
  if (!Number.isFinite(shap) || shap === 0) return [0.55, 0.6, 0.68];
  return shap > 0 ? [1.0, 0.72, 0.2] : [1.0, 0.4, 0.45];
}

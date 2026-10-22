/**
 * Phase 33 — pure math helpers for the XGBoost boosted leaf-flow stage.
 *
 * Dependency-free (no React / Three imports) so SHAP force-vector
 * scaling, staged heights and leaf-path lookups stay unit-testable.
 */

/** Sigmoid link (binary margin → probability), overflow-safe. */
export function xgbSigmoid(z: number): number {
  if (!Number.isFinite(z)) return 0;
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** First 1-based round where the staged linked prediction crosses threshold. */
export function crossedXgbRound(trajectory: number[], threshold = 0.5): number | null {
  for (let m = 0; m < trajectory.length; m += 1) {
    if (trajectory[m] >= threshold) return m + 1;
  }
  return null;
}

/** Normalize a regression terraces height into the display range [0, 2]. */
export function normalizeXgbHeight(zMin: number, zMax: number, z: number): number {
  if (![zMin, zMax, z].every(Number.isFinite)) return 0;
  if (zMax <= zMin) return 1;
  return Math.min(1, Math.max(0, (z - zMin) / (zMax - zMin))) * 2;
}

/** Posterior height (already in [0, 1]) into the display range. */
export function xgbPosteriorHeight(p: number): number {
  if (!Number.isFinite(p)) return 0;
  return Math.min(1, Math.max(0, p)) * 2;
}

/** Map a world value on an axis grid into the [-2.5, 2.5] stage space. */
export function xgbGridToStage(v: number, lo: number, hi: number): number {
  if (![v, lo, hi].every(Number.isFinite) || hi <= lo) return 0;
  return Math.min(2.5, Math.max(-2.5, ((v - lo) / (hi - lo)) * 5 - 2.5));
}

export interface XgbFlatNode {
  id: number;
  leaf?: number;
  cover?: number;
  split_feature_idx?: number;
  threshold?: number | null;
  categories?: number[];
  yes?: number;
  no?: number;
  missing?: number;
  gain?: number;
}

/** Build a child→parent lookup from a flat node list. */
export function xgbParentMap(nodes: XgbFlatNode[]): Map<number, number> {
  const parent = new Map<number, number>();
  for (const n of nodes) {
    if (n.leaf !== undefined) continue;
    if (typeof n.yes === 'number' && n.yes >= 0) parent.set(n.yes, n.id);
    if (typeof n.no === 'number' && n.no >= 0) parent.set(n.no, n.id);
  }
  return parent;
}

/** Walk from a leaf id back to the root, returning the node-id path leaf→root. */
export function xgbLeafPath(leafId: number, nodes: XgbFlatNode[]): number[] {
  const parent = xgbParentMap(nodes);
  const path: number[] = [leafId];
  const seen = new Set<number>([leafId]);
  let cur: number | undefined = leafId;
  for (let i = 0; i < 256; i += 1) {
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
export function xgbForceLength(shap: number, maxAbs: number): number {
  if (![shap, maxAbs].every(Number.isFinite) || maxAbs <= 0) return 0;
  return Math.min(1.2, (Math.abs(shap) / maxAbs) * 1.2);
}

/** Force-arrow color: emerald push (positive SHAP) vs rose pull (negative). */
export function xgbForceColor(shap: number): [number, number, number] {
  if (!Number.isFinite(shap) || shap === 0) return [0.55, 0.6, 0.68];
  return shap > 0 ? [0.2, 0.85, 0.55] : [1.0, 0.4, 0.45];
}

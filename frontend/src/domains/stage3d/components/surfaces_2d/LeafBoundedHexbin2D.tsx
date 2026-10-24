import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { hexPath } from './pointIntelligence2dMath';
import { LEAF_MIXED_COLOR, LEAF_PURE_COLOR } from './treesPointIntel2dMath';

export interface LeafBoundedHexbin2DProps {
  cells: HexbinCell2D[];
  /** Parallel purity 1−impurity of the containing leaf; NaN bins are skipped. */
  purities: Array<number | null>;
  visible?: boolean;
}

/**
 * Phase 53 (mechanism D, DoD #4) — leaf-partitioned hexbin aggregation.
 *
 * Bins fully inside a pure leaf render as a faint solid fill; the most
 * impure third of mixed-leaf bins warns with an amber dashed outline.
 * NaN bins (no containing leaf, e.g. forest families without leaf boxes)
 * never warn. Two batched paths hold 60 FPS.
 */
export function LeafBoundedHexbin2D({ cells, purities, visible = true }: LeafBoundedHexbin2DProps) {
  if (!visible || cells.length === 0 || purities.length === 0) return null;
  const n = Math.min(cells.length, purities.length);
  const scored = purities
    .slice(0, n)
    .map((p, i) => ({ p, i }))
    .filter((e): e is { p: number; i: number } => typeof e.p === 'number' && Number.isFinite(e.p));
  if (scored.length === 0) return null;
  const pure = scored.filter((e) => e.p >= 0.999);
  const mixed = scored.filter((e) => e.p < 0.999).sort((a, b) => a.p - b.p);
  const cut = Math.max(1, Math.floor(mixed.length / 3));
  const warned = mixed.slice(0, cut);
  if (pure.length === 0 && warned.length === 0) return null;
  const pathOf = (idx: number[]): string =>
    idx.map((i) => hexPath(cells[i]!.centerX, cells[i]!.centerY, cells[i]!.radius)).join('');
  return (
    <g className="leaf-hexbin-2d pointer-events-none" aria-label="Leaf-bounded hexbin tiers">
      {pure.length > 0 && (
        <path
          d={pathOf(pure.map((e) => e.i))}
          fill={LEAF_PURE_COLOR}
          fillOpacity={0.14}
          stroke={LEAF_PURE_COLOR}
          strokeWidth={0.02}
          opacity={0.8}
        >
          <title>{`Pure-leaf blocks: ${pure.length} bins (purity 1.0)`}</title>
        </path>
      )}
      {warned.length > 0 && (
        <path
          d={pathOf(warned.map((e) => e.i))}
          fill="none"
          stroke={LEAF_MIXED_COLOR}
          strokeWidth={0.035}
          strokeDasharray="0.2 0.12"
          opacity={0.9}
        >
          <title>{`Mixed-leaf blocks: ${warned.length} bins (lowest purity first)`}</title>
        </path>
      )}
    </g>
  );
}

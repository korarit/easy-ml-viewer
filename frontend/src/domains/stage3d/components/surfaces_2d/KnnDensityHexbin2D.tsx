import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { hexPath } from './pointIntelligence2dMath';
import { densityTiers } from './distancePointIntel2dMath';

export interface KnnDensityHexbin2DProps {
  cells: HexbinCell2D[];
  /** Parallel ρ = K/(π·R̄_K²) values; null/NaN bins are skipped honestly. */
  densities: Array<number | null>;
  visible?: boolean;
}

/**
 * Phase 51 (mechanism D) — local manifold density hexbin.
 *
 * Bins tier by ρ into dense islands (solid indigo fill — the top tercile)
 * and data-desert borders (dashed outline — the bottom tercile), so the eye
 * separates trustworthy dense regions from sparse outliers. Two batched
 * paths regardless of bin count.
 */
export function KnnDensityHexbin2D({ cells, densities, visible = true }: KnnDensityHexbin2DProps) {
  if (!visible || cells.length === 0 || densities.length === 0) return null;
  const n = Math.min(cells.length, densities.length);
  const { dense, sparse } = densityTiers(densities.slice(0, n).map((d) => (typeof d === 'number' ? d : null)));
  if (dense.length === 0 && sparse.length === 0) return null;
  const pathOf = (idx: number[]): string =>
    idx.map((i) => hexPath(cells[i]!.centerX, cells[i]!.centerY, cells[i]!.radius)).join('');
  return (
    <g className="knn-density-hexbin-2d pointer-events-none" aria-label="KNN manifold density tiers">
      {dense.length > 0 && (
        <path
          d={pathOf(dense)}
          fill="#6366f1"
          fillOpacity={0.4}
          stroke="#a5b4fc"
          strokeWidth={0.03}
        >
          <title>{`Dense islands: top-tercile ρ=K/(π·R̄²) over ${dense.length} bins`}</title>
        </path>
      )}
      {sparse.length > 0 && (
        <path
          d={pathOf(sparse)}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={0.035}
          strokeDasharray="0.2 0.12"
          opacity={0.9}
        >
          <title>{`Sparse borders: bottom-tercile ρ over ${sparse.length} bins (outlier country)`}</title>
        </path>
      )}
    </g>
  );
}

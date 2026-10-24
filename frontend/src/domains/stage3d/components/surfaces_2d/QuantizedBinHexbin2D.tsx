import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { hexPath } from './pointIntelligence2dMath';
import { QUANT_HOT_COLOR, QUANT_TRAFFIC_COLOR } from './boostingPointIntel2dMath';

export interface QuantizedBinHexbin2DProps {
  cells: HexbinCell2D[];
  /** Heat per cell: |Σg| when gradient stats are available, else queue count. */
  heat: Array<number | null>;
  /** Which statistic drives the tiers (shown in the title — honest badge). */
  mode: 'gradient' | 'traffic';
  visible?: boolean;
}

/**
 * Phase 54 (mechanism D, DoD #4) — quantized-bin hexbin aggregation.
 *
 * Hexbins snap to the HistGBM / LightGBM 256-bin lattice upstream; the
 * hottest third by heat (|Σg| gradient mass, or queue traffic when the
 * endpoint has not shipped gradient stats yet) fills solid while the
 * rest stays faint. NaN cells never warn. Two batched paths hold 60 FPS.
 */
export function QuantizedBinHexbin2D({ cells, heat, mode, visible = true }: QuantizedBinHexbin2DProps) {
  if (!visible || cells.length === 0 || heat.length === 0) return null;
  const n = Math.min(cells.length, heat.length);
  const scored = heat
    .slice(0, n)
    .map((h, i) => ({ h, i }))
    .filter((e): e is { h: number; i: number } => typeof e.h === 'number' && Number.isFinite(e.h) && e.h > 0);
  if (scored.length === 0) return null;
  const cut = Math.max(1, Math.floor(scored.length / 3));
  const ranked = [...scored].sort((a, b) => b.h - a.h);
  const hot = ranked.slice(0, cut);
  const rest = ranked.slice(cut);
  const pathOf = (idx: number[]): string =>
    idx.map((i) => hexPath(cells[i]!.centerX, cells[i]!.centerY, cells[i]!.radius)).join('');
  const noun = mode === 'gradient' ? '|Σg| gradient mass' : 'queue traffic on the lattice';
  return (
    <g className="quant-bin-hexbin-2d pointer-events-none" aria-label="Quantized bin hexbin tiers">
      {rest.length > 0 && (
        <path
          d={pathOf(rest.map((e) => e.i))}
          fill={QUANT_TRAFFIC_COLOR}
          fillOpacity={0.1}
          stroke={QUANT_TRAFFIC_COLOR}
          strokeWidth={0.02}
          opacity={0.7}
        >
          <title>{`Quantized cells: ${rest.length} calm by ${noun}`}</title>
        </path>
      )}
      {hot.length > 0 && (
        <path
          d={pathOf(hot.map((e) => e.i))}
          fill={QUANT_HOT_COLOR}
          fillOpacity={0.32}
          stroke={QUANT_HOT_COLOR}
          strokeWidth={0.03}
          opacity={0.95}
        >
          <title>{`Quantized hot cells: ${hot.length} bins by ${noun} (hard region)`}</title>
        </path>
      )}
    </g>
  );
}

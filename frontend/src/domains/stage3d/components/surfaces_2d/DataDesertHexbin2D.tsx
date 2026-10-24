import { wy } from '../DecisionBoundaryContour2D';
import type { HexbinCell2D } from '../../../projection3d/types/projection.types';
import { hexPath } from './pointIntelligence2dMath';
import { DESERT_HAZARD_COLOR } from './neuralPointIntel2dMath';

export interface DataDesertHexbin2DProps {
  cells: HexbinCell2D[];
  /** Hazard per cell: 1[count==0] · max P (mirrors the backend kernel). */
  hazard: Array<number | null>;
  visible?: boolean;
}

/**
 * Phase 55 (mechanism D, DoD #3) — data-desert & overconfident hexbins.
 *
 * Cells the network predicts with high confidence yet hold no training
 * mass (Hazard > 0.85) earn a dashed hazard badge with a ▲ marker: the
 * model is extrapolating in a data desert. Two batched paths hold 60 FPS.
 */
export function DataDesertHexbin2D({ cells, hazard, visible = true }: DataDesertHexbin2DProps) {
  if (!visible || cells.length === 0 || hazard.length === 0) return null;
  const n = Math.min(cells.length, hazard.length);
  const desert: number[] = [];
  for (let i = 0; i < n; i += 1) {
    const h = hazard[i];
    if (typeof h === 'number' && Number.isFinite(h) && (h as number) > 0.85) desert.push(i);
  }
  if (desert.length === 0) return null;
  const d = desert.map((i) => hexPath(cells[i]!.centerX, cells[i]!.centerY, cells[i]!.radius)).join('');
  return (
    <g className="data-desert-hexbin-2d pointer-events-none" aria-label="Data desert hazard hexbins">
      <path
        d={d}
        fill={DESERT_HAZARD_COLOR}
        fillOpacity={0.22}
        stroke={DESERT_HAZARD_COLOR}
        strokeWidth={0.04}
        strokeDasharray="0.16 0.1"
        opacity={0.95}
      >
        <title>{`Data desert (${desert.length} cells): empty yet overconfident — extrapolation hazard`}</title>
        <animate attributeName="opacity" values="0.95;0.45;0.95" dur="2.2s" repeatCount="indefinite" />
      </path>
      {desert.slice(0, 40).map((i) => (
        <text
          key={i}
          x={cells[i]!.centerX}
          y={wy(cells[i]!.centerY) + 0.1}
          fontSize={0.32}
          fill={DESERT_HAZARD_COLOR}
          textAnchor="middle"
          opacity={0.95}
        >
          ▲<title>{`Desert cell · Hazard=${(hazard[i] as number).toFixed(3)}`}</title>
        </text>
      ))}
    </g>
  );
}

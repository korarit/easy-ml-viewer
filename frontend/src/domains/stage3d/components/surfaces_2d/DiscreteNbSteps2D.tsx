import type { DiscreteNb2D } from '../../../projection3d/types/projection.types';
import { ellipseColor } from './probabilistic2dMath';

export interface DiscreteNbSteps2DProps {
  discrete: DiscreteNb2D | null;
  xFeature: string;
  yFeature: string;
  visible?: boolean;
}

const BAR_W = 1.1;
const BAR_H = 0.16;
const GAP = 0.24;

/**
 * Phase 44 — discrete-NB step legend (Multinomial / Bernoulli / Complement).
 *
 * The dense heatmap already shows the accumulated log-likelihood; this
 * legend pins the per-axis evidence `x·log p` slices that build it, one
 * row per class (≤8), anchored to the bottom-left of the world frame.
 * Bernoulli rows split presence vs absence via the binarize threshold
 * recorded in the tooltip.
 */
export function DiscreteNbSteps2D({
  discrete,
  xFeature,
  yFeature,
  visible = true,
}: DiscreteNbSteps2DProps) {
  if (!visible || !discrete || discrete.axisLogProb.length === 0) return null;
  const rows = discrete.axisLogProb.slice(0, 8);
  const all = rows.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo < 1e-12 ? 1 : hi - lo;
  const originX = -4.9;
  const originY = -4.9;
  const norm = (v: number) => (v - lo) / span;

  return (
    <g
      className="discrete-nb-steps-2d pointer-events-none"
      aria-label={`Discrete NB evidence steps (${discrete.variant})`}
    >
      {rows.map((pair, ci) => {
        const color = ellipseColor(ci);
        const yRow = originY + (rows.length - 1 - ci) * (BAR_H * 2 + GAP);
        const label = discrete.classes[ci] === undefined ? `#${ci}` : String(discrete.classes[ci]);
        return (
          <g key={`nb-step-${ci}`}>
            <text x={originX} y={yRow - 0.06} fontSize={0.26} fill={color} textAnchor="start">
              {label}
              <title>{`Class ${label} · ${discrete.formula}`}</title>
            </text>
            {/* X-axis evidence bar */}
            <rect
              x={originX}
              y={yRow}
              width={Math.max(norm(pair[0]) * BAR_W, 0.02)}
              height={BAR_H}
              fill={color}
              opacity={0.85}
            >
              <title>{`Class ${label} · ${xFeature}: log p = ${pair[0].toPrecision(4)}`}</title>
            </rect>
            {/* Y-axis evidence bar */}
            <rect
              x={originX}
              y={yRow + BAR_H + 0.04}
              width={Math.max(norm(pair[1] ?? pair[0]) * BAR_W, 0.02)}
              height={BAR_H}
              fill={color}
              opacity={0.55}
            >
              <title>{`Class ${label} · ${yFeature}: log p = ${(pair[1] ?? pair[0]).toPrecision(4)}`}</title>
            </rect>
          </g>
        );
      })}
    </g>
  );
}

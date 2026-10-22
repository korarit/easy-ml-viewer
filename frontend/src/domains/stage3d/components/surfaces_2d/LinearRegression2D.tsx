import { wy } from '../DecisionBoundaryContour2D';
import { deriveExactElevationLines } from './linear2dMath';

export interface LinearRegression2DProps {
  w1: number;
  w2: number;
  bEff: number;
  /** Sampled elevation thresholds from the contour payload (ŷ levels). */
  elevationThresholds: number[];
  xRangeRaw: [number, number];
  yRangeRaw: [number, number];
  slopeW1: number | null;
  visible?: boolean;
}

/**
 * Phase 42 — exact regression guides for OLS / Ridge / Lasso / ElasticNet /
 * SGD-regressor planes.
 *
 * For a linear plane the ŷ = t isolines are exactly parallel straight lines;
 * deriving them analytically keeps them razor-sharp where Marching Squares
 * would stair-step. (In-plane "residual bars" are deliberately *not* drawn:
 * residuals live in the ŷ dimension, which this planar view encodes as
 * color — an in-plane connector would mislead.)
 */
export function LinearRegression2D({
  w1,
  w2,
  bEff,
  elevationThresholds,
  xRangeRaw,
  yRangeRaw,
  slopeW1,
  visible = true,
}: LinearRegression2DProps) {
  if (!visible) return null;
  const guides = deriveExactElevationLines(w1, w2, bEff, elevationThresholds, xRangeRaw, yRangeRaw);
  if (guides.length === 0) return null;
  return (
    <g className="linear-regression-2d pointer-events-none" aria-label="Exact regression elevation guides">
      {guides.map((g) => (
        <line
          key={g.threshold}
          x1={g.segment[0]}
          y1={wy(g.segment[1])}
          x2={g.segment[2]}
          y2={wy(g.segment[3])}
          stroke="#fbbf24"
          strokeWidth={0.05}
          strokeDasharray="0.3 0.12"
          opacity={0.85}
          strokeLinecap="round"
        >
          <title>
            {`Exact elevation ŷ = ${g.threshold}${slopeW1 !== null ? ` · slope dŷ/dx1 = ${slopeW1.toPrecision(4)}` : ''}`}
          </title>
        </line>
      ))}
    </g>
  );
}

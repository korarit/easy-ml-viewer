import { wy } from '../DecisionBoundaryContour2D';
import type { ClassEllipse2D } from '../../../projection3d/types/projection.types';
import { ellipseColor, sigmaCoverage } from './probabilistic2dMath';

export interface GaussianEllipses2DProps {
  ellipses: ClassEllipse2D[];
  showCentroids?: boolean;
  visible?: boolean;
}

const CROSS = 0.14;

/**
 * Phase 44 — bivariate equiprobability ellipses (1σ / 2σ / 3σ) per class.
 *
 * World coords (SVG Y flipped via `wy`); the data-space rotation is negated
 * to survive the flip. GNB ellipses are axis-aligned (rotation 0); LDA/QDA
 * ellipses tilt along the covariance eigenvectors. Centroid crosshairs stay
 * axis-aligned in world space so (μx, μy) reads exactly.
 */
export function GaussianEllipses2D({
  ellipses,
  showCentroids = true,
  visible = true,
}: GaussianEllipses2DProps) {
  if (!visible || ellipses.length === 0) return null;
  return (
    <g className="gaussian-ellipses-2d pointer-events-none" aria-label="Class probability ellipses">
      {ellipses.map((e) => {
        const color = ellipseColor(e.classIndex);
        const [cx, cy] = e.centerWorld;
        // Phase-44 stores radii ASCENDING [minor, major] with rotationRad =
        // the MAJOR-axis direction; rotate(−θ) maps local-x onto that
        // direction, so local-x must carry the MAJOR radius (verified: the
        // unswapped mapping transposes rotated LDA/QDA ellipses).
        const [rMinor, rMajor] = e.radiiWorld1Sigma;
        const rx = rMajor;
        const ry = rMinor;
        const rotDeg = (-e.rotationRad * 180) / Math.PI;
        const label = e.label === null || e.label === undefined ? `#${e.classIndex}` : String(e.label);
        return (
          <g key={`ellipse-${e.classIndex}`}>
            <g transform={`translate(${cx.toFixed(3)} ${wy(cy).toFixed(3)}) rotate(${rotDeg.toFixed(2)})`}>
              {/* 3σ ring (98.9%) */}
              <ellipse
                rx={Math.max(rx * 3, 1e-6)}
                ry={Math.max(ry * 3, 1e-6)}
                fill="none"
                stroke={color}
                strokeWidth={0.03}
                strokeDasharray="0.14 0.12"
                opacity={0.35}
              >
                <title>{`Class ${label} 3σ ring (${sigmaCoverage(3)} mass)`}</title>
              </ellipse>
              {/* 2σ ring (86.5%) */}
              <ellipse
                rx={Math.max(rx * 2, 1e-6)}
                ry={Math.max(ry * 2, 1e-6)}
                fill="none"
                stroke={color}
                strokeWidth={0.045}
                strokeDasharray="0.2 0.1"
                opacity={0.65}
              >
                <title>{`Class ${label} 2σ ring (${sigmaCoverage(2)} mass)`}</title>
              </ellipse>
              {/* 1σ ring (39.3%) */}
              <ellipse
                rx={Math.max(rx, 1e-6)}
                ry={Math.max(ry, 1e-6)}
                fill={color}
                fillOpacity={0.12}
                stroke={color}
                strokeWidth={0.06}
              >
                <title>
                  {`Class ${label} 1σ ellipse (${sigmaCoverage(1)} mass) · μ=(${e.meanRaw[0].toPrecision(4)}, ${e.meanRaw[1].toPrecision(4)}) σ=(${e.stdRaw[0].toPrecision(4)}, ${e.stdRaw[1].toPrecision(4)})`}
                </title>
              </ellipse>
            </g>
            {showCentroids && (
              <g stroke={color} strokeWidth={0.06} strokeLinecap="round">
                <line x1={cx - CROSS} y1={wy(cy)} x2={cx + CROSS} y2={wy(cy)}>
                  <title>{`Class ${label} centroid (μx=${e.meanRaw[0].toPrecision(4)}, μy=${e.meanRaw[1].toPrecision(4)})`}</title>
                </line>
                <line x1={cx} y1={wy(cy) - CROSS} x2={cx} y2={wy(cy) + CROSS}>
                  <title>{`Class ${label} centroid (μx=${e.meanRaw[0].toPrecision(4)}, μy=${e.meanRaw[1].toPrecision(4)})`}</title>
                </line>
              </g>
            )}
          </g>
        );
      })}
    </g>
  );
}

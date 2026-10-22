import { wy } from '../DecisionBoundaryContour2D';
import type { OnnxLatent2D } from '../../../projection3d/types/projection.types';
import { fitLatentToWorld } from './neural2dMath';

/** Inset mini-map geometry in world units (bottom-right of the ±5 frame). */
const INSET_X = 1.6;
const INSET_TOP_WORLD = -1.6;
const INSET_SIZE = 3.0;

export interface OnnxLatentManifold2DProps {
  latent: OnnxLatent2D;
  showManifold?: boolean;
}

/**
 * Phase 47 — ONNX deep 2D latent projection plane (dev-plan §2.3).
 *
 * The penultimate-layer embedding is a different coordinate system from the
 * (X, Y) feature slice, so it renders as an inset mini-map: SVD-projected
 * sample points fitted into a framed 3×3 world-unit panel with the
 * explained-variance readout. Classical `ai.onnx.ml` graphs carry no
 * latent plane (`latentAvailable === false`) and render nothing — the
 * universal contour remains the exact 2D map.
 */
export function OnnxLatentManifold2D({
  latent,
  showManifold = true,
}: OnnxLatentManifold2DProps) {
  if (!showManifold || !latent.latentAvailable || latent.latentPoints2d.length === 0) return null;

  const svgTop = wy(INSET_TOP_WORLD);
  const fitted = fitLatentToWorld(latent.latentPoints2d, INSET_X, svgTop, INSET_SIZE);
  const [ev1, ev2] = latent.latentExplainedVariance;

  return (
    <g className="onnx-latent-manifold-2d pointer-events-none" aria-label="ONNX 2D latent projection plane">
      <rect
        x={INSET_X}
        y={svgTop}
        width={INSET_SIZE}
        height={INSET_SIZE}
        fill="rgba(10,15,22,0.78)"
        stroke="#818cf8"
        strokeWidth={0.04}
        rx={0.12}
      />
      <text x={INSET_X + 0.12} y={svgTop + 0.34} fontSize={0.26} fill="#c7d2fe">
        {`Latent SVD 2D (${latent.latentPoints2d.length} pts${ev1 !== undefined ? ` · ${(ev1 * 100).toFixed(0)}%/${((ev2 ?? 0) * 100).toFixed(0)}%` : ''})`}
      </text>
      {fitted.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={0.055} fill="#a5b4fc" opacity={0.8}>
          {i === 0 && <title>{latent.formula}</title>}
        </circle>
      ))}
    </g>
  );
}

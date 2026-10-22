import { useStage3DStore } from '../../store/useStage3DStore';
import type { Contour2DPayload } from '../../../projection3d/types/projection.types';
import { GaussianEllipses2D } from './GaussianEllipses2D';
import { DiscriminantBoundary2D } from './DiscriminantBoundary2D';
import { DiscreteNbSteps2D } from './DiscreteNbSteps2D';

/**
 * Phase 44 — orchestrator for probabilistic & discriminant 2D overlays.
 *
 * - Gaussian/LDA/QDA ellipses (1σ–3σ) + centroid crosshairs,
 * - LDA exact line + QDA analytic conic curve,
 * - discrete-NB per-axis evidence step legend.
 * Renders nothing when the backend sent no `probabilistic2d` payload
 * (non-probabilistic / pipelined models) — the sampled contour remains.
 */
export function ProbabilisticOverlays2D({
  contourData,
}: {
  contourData: Contour2DPayload;
}) {
  const showProbEllipses = useStage3DStore((s) => s.showProbEllipses);
  const showProbCentroids = useStage3DStore((s) => s.showProbCentroids);
  const showConicBoundary = useStage3DStore((s) => s.showConicBoundary);

  const prob = contourData.probabilistic2d;
  if (!prob) return null;

  return (
    <g aria-label="Probabilistic and discriminant overlays">
      <GaussianEllipses2D
        ellipses={prob.ellipses}
        showCentroids={showProbCentroids}
        visible={showProbEllipses}
      />
      <DiscriminantBoundary2D
        line={prob.discriminantLine}
        conic={prob.conic}
        showLine={showConicBoundary}
        showConic={showConicBoundary}
      />
      <DiscreteNbSteps2D
        discrete={prob.discrete}
        xFeature={prob.xFeature}
        yFeature={prob.yFeature}
        visible={showProbEllipses}
      />
    </g>
  );
}

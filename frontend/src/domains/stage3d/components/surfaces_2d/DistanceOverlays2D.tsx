import { useStage3DStore } from '../../store/useStage3DStore';
import type {
  Contour2DPayload,
  KnnFrame2DPayload,
} from '../../../projection3d/types/projection.types';
import { KnnRadar2D, type KnnNeighbor2D } from './KnnRadar2D';
import { SvmKernelContour2D } from './SvmKernelContour2D';
import { SupportVectorRings2D } from './SupportVectorRings2D';

/**
 * Phase 43 — orchestrator for distance & kernel 2D overlays.
 *
 * - KNN radar + spider lines track the per-frame payload (fetched live as
 *   the VCR scrubs); the static `knn2d` descriptor gates the fetch.
 * - SVM margin parallels come from the score-field contour (`kind margin`);
 *   halos come from the static `svm2d` payload. Renders nothing when the
 *   payloads are absent (non-distance models) — the sampled contour remains.
 */
export function DistanceOverlays2D({
  contourData,
  knnFrame,
}: {
  contourData: Contour2DPayload;
  knnFrame: KnnFrame2DPayload | null;
}) {
  const showKnnRadar = useStage3DStore((s) => s.showKnnRadar);
  const showSpiderLines = useStage3DStore((s) => s.showSpiderLines);
  const showSvHalos = useStage3DStore((s) => s.showSvHalos);
  const showSvmMargins = useStage3DStore((s) => s.showSvmMargins);

  const svm = contourData.svm2d;
  const marginLevels = contourData.contours.filter((c) => c.kind === 'margin');
  // Embedded halos are DTO-capped (≤500) and grouped into ≤3 SVG paths,
  // so even the largest scenes stay at a handful of draw calls (E43.5).
  const halos = svm && !svm.isLinearNoSv ? svm.halos : [];

  const radarVisible = showKnnRadar && knnFrame && !knnFrame.outlier;
  const spiderNeighbors: KnnNeighbor2D[] =
    knnFrame && showSpiderLines && !knnFrame.outlier
      ? knnFrame.neighbors.map((n) => ({
          refIndex: n.trainIndex,
          x: n.xWorld,
          y: n.yWorld,
          distance2D: n.distance2d,
          voteWeight: n.voteWeight,
          targetClass: n.label,
          rank: n.rank,
        }))
      : [];

  return (
    <g aria-label="Distance and kernel overlays">
      {showSvmMargins && <SvmKernelContour2D marginLevels={marginLevels} />}
      {showSvHalos && <SupportVectorRings2D halos={halos} />}
      {radarVisible && (
        <KnnRadar2D
          centerX={knnFrame.queryWorld[0]}
          centerY={knnFrame.queryWorld[1]}
          radiusX={knnFrame.radarRadiusWorld[0]}
          radiusY={knnFrame.radarRadiusWorld[1]}
          neighbors={spiderNeighbors}
          visible={showKnnRadar || showSpiderLines}
        />
      )}
    </g>
  );
}

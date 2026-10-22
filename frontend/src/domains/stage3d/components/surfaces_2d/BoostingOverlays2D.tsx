import { useStage3DStore } from '../../store/useStage3DStore';
import type {
  BoostFrame2DPayload,
  Contour2DPayload,
} from '../../../projection3d/types/projection.types';
import { batchCutPaths } from './trees2dMath';
import { findBoostLeafAt } from './boosting2dMath';
import { XgboostTreeShapVector2D } from './XgboostTreeShapVector2D';
import { HistGbmQuantizedBins2D } from './HistGbmQuantizedBins2D';
import { CatBoostObliviousGrid2D } from './CatBoostObliviousGrid2D';
import { LightgbmLeafWise2D } from './LightgbmLeafWise2D';

const SHAP_TONE: Record<string, string> = {
  xgboost: '#f43f5e',
  lightgbm: '#f59e0b',
  catboost: '#8b5cf6',
};

/**
 * Phase 46 — orchestrator for gradient-boosting heavyweight 2D overlays.
 *
 * - xgboost → tree-0 leaf polygons + live C++ TreeSHAP push vector,
 * - lightgbm → asymmetric tree-0 leaf-wise polygons + live push vector,
 * - catboost → symmetric oblivious checkerboard + live SHAP push vector,
 * - hist_gbm → 255-bin quantized grid + active bin cell,
 * - sklearn_gbm → superimposed top-8 residual-tree cuts (dense field is
 *   the additive log-odds surface, the overlay adds the mosaic structure).
 * Renders nothing when the backend sent no `boosting2d` payload
 * (non-boosting / pipelined models) — the sampled contour remains.
 */
export function BoostingOverlays2D({
  contourData,
  boostFrame,
}: {
  contourData: Contour2DPayload;
  boostFrame: BoostFrame2DPayload | null;
}) {
  const showShapVector = useStage3DStore((s) => s.showShapVector);
  const showLeafPolygons = useStage3DStore((s) => s.showLeafPolygons);
  const showQuantizedBins = useStage3DStore((s) => s.showQuantizedBins);
  const showObliviousGrid = useStage3DStore((s) => s.showObliviousGrid);
  const points = useStage3DStore((s) => s.points);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);

  const boosting = contourData.boosting2d;
  if (!boosting) return null;

  const activePoint =
    points.length > 0 && activeFrameIndex >= 0 && activeFrameIndex < points.length
      ? {
          x: points[activeFrameIndex].world_x,
          y: points[activeFrameIndex].world_y,
          row: points[activeFrameIndex].original_row_index,
        }
      : null;

  // Live per-frame SHAP wins over the static contour snapshot (median
  // pinning ships no row); both carry the same native C++ values.
  const liveShap =
    boostFrame && boostFrame.family === boosting.family ? boostFrame.shap : boosting.shapVector;
  const liveOrigin =
    boostFrame && boostFrame.family === boosting.family && activePoint
      ? { x: boostFrame.queryWorld[0], y: boostFrame.queryWorld[1] }
      : activePoint;
  const tone = SHAP_TONE[boosting.family] ?? '#f43f5e';

  return (
    <g aria-label="Boosting heavyweight overlays">
      {(boosting.family === 'xgboost' || boosting.family === 'lightgbm') && (
        <LightgbmLeafWise2D
          segments={boosting.segments}
          leaves={boosting.leaves}
          activePoint={activePoint}
          activeRowIndex={activePoint?.row ?? null}
          showPolygons={showLeafPolygons}
          showActiveLeaf={showLeafPolygons}
        />
      )}
      {boosting.family === 'sklearn_gbm' && showLeafPolygons && (
        <g className="sklearn-gbm-cuts-2d pointer-events-none" aria-label="Sklearn GBM residual cuts">
          {(() => {
            const paths = batchCutPaths(boosting.segments);
            return (
              <>
                {paths.vertical && (
                  <path d={paths.vertical} fill="none" stroke="#34d399" strokeWidth={0.035} strokeDasharray="0.22 0.14" opacity={0.8} strokeLinecap="round">
                    <title>{`GBM top-${boosting.treesExported} residual cuts (vertical)`}</title>
                  </path>
                )}
                {paths.horizontal && (
                  <path d={paths.horizontal} fill="none" stroke="#34d399" strokeWidth={0.035} strokeDasharray="0.22 0.14" opacity={0.8} strokeLinecap="round">
                    <title>{`GBM top-${boosting.treesExported} residual cuts (horizontal)`}</title>
                  </path>
                )}
              </>
            );
          })()}
        </g>
      )}
      {boosting.family === 'hist_gbm' && (
        <HistGbmQuantizedBins2D
          linesXWorld={boosting.binLinesXWorld}
          linesYWorld={boosting.binLinesYWorld}
          activeBin={boosting.activeBin}
          activeRowIndex={activePoint?.row ?? null}
          showBins={showQuantizedBins}
          showActiveBin={showQuantizedBins}
        />
      )}
      {boosting.family === 'catboost' && (
        <CatBoostObliviousGrid2D
          linesXWorld={boosting.obliviousXLinesWorld}
          linesYWorld={boosting.obliviousYLinesWorld}
          cells={boosting.obliviousCells}
          obliviousRule={boosting.obliviousRule}
          activeRowIndex={activePoint?.row ?? null}
          showGrid={showObliviousGrid}
          showActiveCell={showObliviousGrid}
        />
      )}
      {liveShap && liveOrigin && (
        <XgboostTreeShapVector2D
          originX={liveOrigin.x}
          originY={liveOrigin.y}
          phiX={liveShap.phiX}
          phiY={liveShap.phiY}
          method={liveShap.method}
          tone={tone}
          visible={showShapVector}
        />
      )}
      {boosting.noAxisSplits && (
        <title>Splits occur on non-visualized axes; uniform prediction on this slice</title>
      )}
    </g>
  );
}

// Resolved client-side active leaf for tooltips (point-in-box, zero round-trips).
export function resolveBoostActiveLeaf(
  contourData: Contour2DPayload,
  wx: number,
  wyv: number,
): { leafId: number; leafValue: number } | null {
  const boosting = contourData.boosting2d;
  if (!boosting || (boosting.family !== 'xgboost' && boosting.family !== 'lightgbm')) return null;
  const hit = findBoostLeafAt(boosting.leaves, wx, wyv);
  return hit ? { leafId: hit.leafId, leafValue: hit.leafValue } : null;
}

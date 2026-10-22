import { useStage3DStore } from '../../store/useStage3DStore';
import type { Contour2DPayload } from '../../../projection3d/types/projection.types';
import { TreePartitions2D } from './TreePartitions2D';
import { EnsembleConsensus2D } from './EnsembleConsensus2D';
import { AdaBoostCascade2D } from './AdaBoostCascade2D';

/**
 * Phase 45 — orchestrator for tree & ensemble 2D overlays.
 *
 * - decision_tree → orthogonal cuts + leaf boxes + active-leaf glow,
 * - random_forest / extra_trees → superimposed top-N mosaic cuts
 *   (the heatmap underneath already is the vote consensus),
 * - adaboost → staged weak-learner cuts honouring the cascade slider.
 * Renders nothing when the backend sent no `trees2d` payload
 * (non-tree / pipelined models) — the sampled contour remains.
 */
export function TreesOverlays2D({ contourData }: { contourData: Contour2DPayload }) {
  const showTreeCuts = useStage3DStore((s) => s.showTreeCuts);
  const showActiveLeaf = useStage3DStore((s) => s.showActiveLeaf);
  const showForestCuts = useStage3DStore((s) => s.showForestCuts);
  const adaBoostStage = useStage3DStore((s) => s.adaBoostStage);
  const points = useStage3DStore((s) => s.points);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);

  const trees = contourData.trees2d;
  if (!trees) return null;

  const activePoint =
    points.length > 0 && activeFrameIndex >= 0 && activeFrameIndex < points.length
      ? {
          x: points[activeFrameIndex].world_x,
          y: points[activeFrameIndex].world_y,
          row: points[activeFrameIndex].original_row_index,
        }
      : null;

  return (
    <g aria-label="Tree and ensemble overlays">
      {trees.family === 'decision_tree' && (
        <TreePartitions2D
          segments={trees.segments}
          leaves={trees.leaves}
          activePoint={activePoint}
          activeRowIndex={activePoint?.row ?? null}
          showCuts={showTreeCuts}
          showActiveLeaf={showActiveLeaf}
        />
      )}
      {(trees.family === 'random_forest' || trees.family === 'extra_trees') && (
        <EnsembleConsensus2D
          segments={trees.segments}
          nEstimators={trees.nEstimators}
          treesExported={trees.treesExported}
          showCuts={showForestCuts}
        />
      )}
      {trees.family === 'adaboost' && (
        <AdaBoostCascade2D
          stages={trees.stages}
          stage={adaBoostStage === null ? null : Math.max(0, Math.min(adaBoostStage, trees.nEstimators))}
          baseValue={trees.baseValue}
        />
      )}
    </g>
  );
}

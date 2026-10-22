import type { TreeCutSegment2D } from '../../../projection3d/types/projection.types';
import { batchCutPaths } from './trees2dMath';

export interface EnsembleConsensus2DProps {
  segments: TreeCutSegment2D[];
  nEstimators: number;
  treesExported: number;
  showCuts?: boolean;
  visible?: boolean;
}

/**
 * Phase 45 — forest mosaic: top-N tree cuts superimposed as one faint layer.
 *
 * The vote-fraction heatmap underneath already IS the consensus surface
 * (predict_proba == mean tree vote); this overlay adds the tessellated
 * rectangular structure. Batched into 2 SVG paths regardless of tree count.
 */
export function EnsembleConsensus2D({
  segments,
  nEstimators,
  treesExported,
  showCuts = true,
  visible = true,
}: EnsembleConsensus2DProps) {
  if (!visible || !showCuts || segments.length === 0) return null;
  const paths = batchCutPaths(segments);
  const style = {
    fill: 'none' as const,
    stroke: '#fbbf24',
    strokeWidth: 0.03,
    opacity: 0.5,
    strokeLinecap: 'round' as const,
  };
  return (
    <g className="ensemble-consensus-2d pointer-events-none" aria-label="Forest consensus cuts">
      {paths.vertical && (
        <path d={paths.vertical} {...style}>
          <title>{`Top-${treesExported} of ${nEstimators} trees · vertical mosaic cuts`}</title>
        </path>
      )}
      {paths.horizontal && (
        <path d={paths.horizontal} {...style}>
          <title>{`Top-${treesExported} of ${nEstimators} trees · horizontal mosaic cuts`}</title>
        </path>
      )}
    </g>
  );
}

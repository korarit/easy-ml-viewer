import type { MlpHyperplane2D } from '../../../projection3d/types/projection.types';
import { batchHyperplanePaths } from './neural2dMath';

export interface MlpNeuronHyperplanes2DProps {
  hyperplanes: MlpHyperplane2D[];
  showCuts?: boolean;
  showDead?: boolean;
}

/**
 * Phase 47 — MLP first-layer switching lines (dev-plan §5.2 prototype).
 *
 * Each hidden neuron's `z_k = 0` line is drawn thin cyan; Dead ReLUs
 * (z_k < 0 over the whole slice) render dashed red at lower opacity.
 * Batching keeps the full first layer at 2 draw calls regardless of width.
 */
export function MlpNeuronHyperplanes2D({
  hyperplanes,
  showCuts = true,
  showDead = true,
}: MlpNeuronHyperplanes2DProps) {
  if (!showCuts && !showDead) return null;
  if (hyperplanes.length === 0) return null;
  const paths = batchHyperplanePaths(hyperplanes);

  return (
    <g className="mlp-neuron-hyperplanes-2d pointer-events-none" aria-label="MLP first-layer neuron hyperplanes">
      {showCuts && paths.active && (
        <path
          d={paths.active}
          fill="none"
          stroke="#22d3ee"
          strokeWidth={0.035}
          opacity={0.75}
          strokeLinecap="round"
        >
          <title>{`Layer-1 switching lines z_k=0 (${hyperplanes.filter((h) => !h.isDead && h.segmentWorld).length} active cuts)`}</title>
        </path>
      )}
      {showDead && paths.dead && (
        <path
          d={paths.dead}
          fill="none"
          stroke="#f87171"
          strokeWidth={0.045}
          strokeDasharray="0.28 0.16"
          opacity={0.85}
          strokeLinecap="round"
        >
          <title>{`Dead ReLUs in this 2D slice (${hyperplanes.filter((h) => h.isDead).length} dead)`}</title>
        </path>
      )}
    </g>
  );
}

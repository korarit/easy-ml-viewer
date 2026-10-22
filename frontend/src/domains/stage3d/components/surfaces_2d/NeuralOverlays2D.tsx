import { useStage3DStore } from '../../store/useStage3DStore';
import type { Contour2DPayload } from '../../../projection3d/types/projection.types';
import { MlpNeuronHyperplanes2D } from './MlpNeuronHyperplanes2D';
import { PipelineSpaceToggle2D } from './PipelineSpaceToggle2D';
import { OnnxLatentManifold2D } from './OnnxLatentManifold2D';

/**
 * Phase 47 — orchestrator for neural / pipeline / ONNX 2D overlays.
 *
 * - mlp → first-layer switching lines + dead-ReLU flags,
 * - pipeline → [Raw] <-> [Scaled] space toggle (order-preserving single
 *   scaler only; anything fancier keeps the raw-space contour + badge),
 * - onnx_classical → no planar overlay (the universal contour is exact),
 * - onnx_deep → inset 2D latent SVD projection plane.
 * Renders nothing when the backend sent no `neural2d` payload
 * (non-neural models) — the sampled contour remains.
 */
export function NeuralOverlays2D({ contourData }: { contourData: Contour2DPayload }) {
  const showNeuronCuts = useStage3DStore((s) => s.showNeuronCuts);
  const showDeadReLUs = useStage3DStore((s) => s.showDeadReLUs);
  const showLatentManifold = useStage3DStore((s) => s.showLatentManifold);
  const showSpaceToggle = useStage3DStore((s) => s.showSpaceToggle);
  const pipelineSpace = useStage3DStore((s) => s.pipelineSpace);
  const setPipelineSpace = useStage3DStore((s) => s.setPipelineSpace);

  const neural = contourData.neural2d;
  if (!neural) return null;

  return (
    <g aria-label="Neural and universal overlays">
      {neural.family === 'mlp' && (
        <MlpNeuronHyperplanes2D
          hyperplanes={neural.hyperplanes}
          showCuts={showNeuronCuts}
          showDead={showDeadReLUs}
        />
      )}
      {neural.family === 'pipeline' && neural.pipeline && (
        <PipelineSpaceToggle2D
          pipeline={neural.pipeline}
          mode={pipelineSpace}
          onChange={setPipelineSpace}
          showToggle={showSpaceToggle}
        />
      )}
      {neural.family === 'onnx_deep' && neural.onnx && (
        <OnnxLatentManifold2D latent={neural.onnx} showManifold={showLatentManifold} />
      )}
      {neural.noAxisSplits && <title>Splits occur on non-visualized axes; uniform prediction on this slice</title>}
    </g>
  );
}

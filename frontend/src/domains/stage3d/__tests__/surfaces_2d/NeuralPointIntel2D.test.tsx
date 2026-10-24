import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { useStage3DStore } from '../../store/useStage3DStore';
import { DualSpaceCometTrail2D } from '../../components/surfaces_2d/DualSpaceCometTrail2D';
import { SoftmaxLogitGapRing2D } from '../../components/surfaces_2d/SoftmaxLogitGapRing2D';
import { LossGradientSpoke2D } from '../../components/surfaces_2d/LossGradientSpoke2D';
import { DataDesertHexbin2D } from '../../components/surfaces_2d/DataDesertHexbin2D';
import { NeuralPointIntelOverlays2D } from '../../components/surfaces_2d/NeuralPointIntelOverlays2D';
import {
  binaryLogit,
  desertHazard,
  dualSpaceValue,
  dualSpaceWorld,
  gradSpokeLengths,
  gradUnit,
  haloTone,
  logitGap,
  logitGapStatus,
} from '../../components/surfaces_2d/neuralPointIntel2dMath';

afterEach(() => {
  useStage3DStore.setState({ showHexbins: false, showErrorsOnly: false });
});

describe('neuralPointIntel2dMath (Phase 55 spec §2)', () => {
  it('dual-space affine maps raw units 1-to-1 (DoD #1)', () => {
    expect(dualSpaceValue(1, 1, 2)).toBeCloseTo(0, 12);
    expect(dualSpaceValue(3, 1, 2)).toBeCloseTo(1, 12);
    expect(Number.isNaN(dualSpaceValue(1, 0, 0))).toBe(true);
    expect(Number.isNaN(dualSpaceValue(NaN, 0, 1))).toBe(true);
    // World tick 0 on raw [0, 10] with mu=5 sigma=5 lands on scaled center 0.
    expect(dualSpaceWorld(0, 0, 10, 5, 5, -1, 1)).toBeCloseTo(0, 12);
    expect(Number.isNaN(dualSpaceWorld(0, 0, 10, 0, 0, -1, 1))).toBe(true);
  });

  it('logit gaps badge saturated / tie / decisive exactly (DoD #2)', () => {
    expect(logitGap([3.5, 0.2])).toBeCloseTo(3.3, 12);
    expect(logitGap([2.0])).toBeCloseTo(2, 12);
    expect(Number.isNaN(logitGap([1, NaN]))).toBe(true);
    expect(logitGapStatus(3.5)).toBe('confident');
    expect(logitGapStatus(3.0)).toBe('decisive');
    expect(logitGapStatus(0.49)).toBe('diffuse');
    expect(logitGapStatus(NaN)).toBe('unknown');
    expect(binaryLogit(0.9)).toBeCloseTo(Math.log(9), 12);
    expect(Number.isNaN(binaryLogit(null))).toBe(true);
  });

  it('gradient spokes peak-normalize with live unit directions', () => {
    const { lengths, live } = gradSpokeLengths([5, 0, NaN]);
    expect(lengths[0]).toBeCloseTo(1.5, 12);
    expect(live).toEqual([true, false, false]);
    expect(gradUnit(3, 4)).toEqual({ ux: 0.6, uy: 0.8 });
    expect(gradUnit(0, 0)).toEqual({ ux: 0, uy: 0 });
  });

  it('desert hazard fires only on empty overconfident cells (DoD #3)', () => {
    expect(desertHazard(0, 0.94)).toEqual({ hazard: 0.94, isDesert: true });
    expect(desertHazard(5, 0.99).isDesert).toBe(false);
    expect(desertHazard(0, 0.5).isDesert).toBe(false);
    expect(desertHazard(0, 0.85).isDesert).toBe(false); // strictly greater
    expect(Number.isNaN(desertHazard(0, null).hazard)).toBe(true);
    expect(haloTone([1, -2])).toBeCloseTo(1.5, 12);
    expect(Number.isNaN(haloTone([NaN]))).toBe(true);
  });
});

describe('Neural point-intelligence SVG components', () => {
  it('dual-space trail replays the active window with a head dot (DoD #1)', () => {
    const pts = [
      { x: -2, y: 0 },
      { x: 0, y: 1 },
      { x: 2, y: 0 },
    ];
    const { container } = render(
      <svg>
        <DualSpaceCometTrail2D points={pts} activeFrame={2} spaceLabel="scaled" />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(1);
    expect(container.querySelectorAll('circle').length).toBe(1);
    expect(screen.getByTitle(/scaled, frames 0–2/)).toBeTruthy();
  });

  it('logit-gap rings split confident / decisive / diffuse channels (DoD #2)', () => {
    const { container } = render(
      <svg>
        <SoftmaxLogitGapRing2D
          rings={[
            { x: 0, y: 0, gap: 3.5, status: 'confident', color: '#22d3ee', label: 's' },
            { x: 1, y: 1, gap: 1.5, status: 'decisive', color: '#94a3b8', label: 'd' },
            { x: 2, y: 2, gap: 0.2, status: 'diffuse', color: '#e879f9', label: 't' },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('circle').length).toBe(1); // confident solid
    expect(container.querySelectorAll('path[d]').length).toBe(2); // decisive + diffuse
    expect(container.querySelector('animate')).toBeTruthy(); // diffuse neon pulse
  });

  it('gradient spokes point along ∇ₓL with fragility labels', () => {
    const { container } = render(
      <svg>
        <LossGradientSpoke2D
          spokes={[
            { x: 0, y: 0, ux: 0.6, uy: 0.8, length: 1.5, magnitude: 5 },
            { x: 1, y: 1, ux: 0, uy: 0, length: 0, magnitude: 0 },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('‖g‖=5.00')).toBeTruthy();
  });

  it('desert hexbins badge empty overconfident cells (DoD #3)', () => {
    const cells = [0, 1].map((i) => ({
      centerX: i,
      centerY: 0,
      radius: 0.45,
      count: i === 0 ? 0 : 8,
      dominantClass: 1,
      meanResidual: 0,
      errorRate: 0,
    }));
    const { container } = render(
      <svg>
        <DataDesertHexbin2D cells={cells} hazard={[0.94, 0]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(1);
    expect(screen.getByText('▲')).toBeTruthy();
  });

  it('renders nothing for empty inputs', () => {
    const { container } = render(
      <svg>
        <DualSpaceCometTrail2D points={[]} activeFrame={0} />
        <SoftmaxLogitGapRing2D rings={[]} />
        <LossGradientSpoke2D spokes={[]} />
        <DataDesertHexbin2D cells={[]} hazard={[]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

describe('NeuralPointIntelOverlays2D orchestrator', () => {
  const baseNeural = {
    algorithm: 'MLPClassifier',
    xFeature: 'a',
    yFeature: 'b',
    xFeatureIdx: 0,
    yFeatureIdx: 1,
    activation: 'relu',
    hyperplanes: [],
    hyperplanesTruncated: false,
    deadCount: 0,
    activeCount: 4,
    activeRatio: 1,
    pipeline: null,
    onnx: null,
    noAxisSplits: false,
    formula: 'mlp',
  };
  const pt = (o: Partial<Record<string, number | null>>) => ({
    frame_index: 0,
    original_row_index: 0,
    world_x: 0,
    world_y: 0,
    world_z_pred: 0,
    world_z_actual: 0,
    raw_x: 0,
    raw_y: 0,
    raw_z_pred: 0,
    raw_z_actual: 0,
    residual_error: 0,
    ...o,
  });
  const shell = {
    points: [pt({}), pt({ world_x: 1 })],
    activeFrameIndex: 1,
    hexbinCells: [],
    isClassification: true,
    classColor: () => '#22d3ee',
    pipelineSpace: 'raw' as const,
    xRangeRaw: [0, 1] as [number, number],
    yRangeRaw: [0, 1] as [number, number],
  };

  it('replays the raw trail for mlp and stays ring-dormant without logits', () => {
    const { container } = render(
      <svg>
        <NeuralPointIntelOverlays2D neural2d={{ ...baseNeural, family: 'mlp' }} {...shell} />
      </svg>,
    );
    expect(container.querySelectorAll('.dual-space-trail-2d path[d]').length).toBe(1);
    expect(container.querySelectorAll('.logit-gap-ring-2d').length).toBe(0);
  });

  it('badges saturated / tie gaps from pre-softmax logits (DoD #2)', () => {
    const { container } = render(
      <svg>
        <NeuralPointIntelOverlays2D
          neural2d={{ ...baseNeural, family: 'onnx_deep' }}
          {...shell}
          points={[pt({}), pt({ world_x: 1 }), pt({ world_x: 2 })]}
          logits={[[3.5, 0.2], [0.1, 0.0], [1.0, 0.9, 0.0]]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('.logit-gap-ring-2d circle').length).toBe(1); // confident
    expect(container.querySelectorAll('.logit-gap-ring-2d path[d]').length).toBe(2); // decisive + diffuse
  });

  it('remaps the trail through the exact scaler affine in scaled space (DoD #1)', () => {
    const { container } = render(
      <svg>
        <NeuralPointIntelOverlays2D
          neural2d={{
            ...baseNeural,
            family: 'pipeline',
            pipeline: {
              hasToggle: true,
              scalerKind: 'standard',
              innerFamily: 'MLPClassifier',
              innerAlgorithm: 'MLPClassifier',
              muX: 0.5,
              sigmaX: 0.5,
              muY: 0.5,
              sigmaY: 0.5,
              scaledXRange: [-1, 1],
              scaledYRange: [-1, 1],
              formula: 'scaled',
            },
          }}
          {...shell}
          pipelineSpace="scaled"
          points={[pt({ world_x: -5, world_y: -5 }), pt({ world_x: 5, world_y: 5 })]}
          xRangeRaw={[0, 1]}
          yRangeRaw={[0, 1]}
        />
      </svg>,
    );
    expect(screen.getByTitle(/scaled, frames 0–1/)).toBeTruthy();
    expect(container.querySelectorAll('.dual-space-trail-2d path[d]').length).toBe(1);
  });

  it('draws live gradient spokes and desert badges when endpoints ship', () => {
    useStage3DStore.setState({ showHexbins: true });
    const { container } = render(
      <svg>
        <NeuralPointIntelOverlays2D
          neural2d={{ ...baseNeural, family: 'mlp' }}
          {...shell}
          points={[pt({}), pt({ world_x: 1 })]}
          gradients={[{ gx: 3, gy: 4 }, { gx: 0, gy: 0 }]}
          hexbinCells={[
            { centerX: 0, centerY: 0, radius: 0.45, count: 0, dominantClass: 1, meanResidual: 0, errorRate: 0 },
          ]}
          cellMaxProba={[0.94]}
        />
      </svg>,
    );
    expect(screen.getByText('‖g‖=5.00')).toBeTruthy();
    expect(container.querySelectorAll('.data-desert-hexbin-2d path[d]').length).toBe(1);
  });

  it('renders nothing for non-neural or missing payloads', () => {
    const { container } = render(
      <svg>
        <NeuralPointIntelOverlays2D neural2d={{ ...baseNeural, family: 'xgboost' }} {...shell} />
        <NeuralPointIntelOverlays2D neural2d={null} {...shell} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

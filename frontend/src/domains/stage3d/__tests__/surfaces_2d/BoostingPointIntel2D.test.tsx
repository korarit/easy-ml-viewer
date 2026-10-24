import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { useStage3DStore } from '../../store/useStage3DStore';
import { BoostingShapPushTrail2D } from '../../components/surfaces_2d/BoostingShapPushTrail2D';
import { FunctionalMarginRing2D } from '../../components/surfaces_2d/FunctionalMarginRing2D';
import { PseudoResidualSpoke2D } from '../../components/surfaces_2d/PseudoResidualSpoke2D';
import { QuantizedBinHexbin2D } from '../../components/surfaces_2d/QuantizedBinHexbin2D';
import { BoostingPointIntelOverlays2D } from '../../components/surfaces_2d/BoostingPointIntelOverlays2D';
import {
  cumulativePush,
  functionalMargin,
  marginStatus,
  pseudoResidualLogloss,
  pseudoResidualLS,
  pushStepMagnitudes,
  pushWorldScale,
  quantizeToBin,
  quantizedCellId,
  spokeLengths,
  toSignedLabel,
} from '../../components/surfaces_2d/boostingPointIntel2dMath';

afterEach(() => {
  useStage3DStore.setState({ showHexbins: false, showErrorsOnly: false });
});

describe('boostingPointIntel2dMath (Phase 54 spec §2)', () => {
  it('cumulative pushes walk stage by stage from the origin', () => {
    const path = cumulativePush([{ phiX: 1, phiY: 0 }, { phiX: 0, phiY: 2 }]);
    expect(path).toEqual([{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 2 }]);
    expect(cumulativePush([])).toEqual([{ x: 0, y: 0 }]);
    expect(pushStepMagnitudes([{ phiX: 3, phiY: 4 }])).toEqual([5]);
    expect(Number.isNaN(pushStepMagnitudes([{ phiX: NaN, phiY: 0 }])[0])).toBe(true);
    expect(pushWorldScale([{ phiX: 10, phiY: 0 }])).toBeLessThan(0.5);
  });

  it('margins badge safe / critical / negative exactly (DoD #2)', () => {
    expect(functionalMargin(1, 2)).toBeCloseTo(2, 12);
    expect(functionalMargin(-1, 2)).toBeCloseTo(-2, 12);
    expect(Number.isNaN(functionalMargin(NaN, 1))).toBe(true);
    expect(marginStatus(2.1)).toBe('safe');
    expect(marginStatus(1.5)).toBe('critical');
    expect(marginStatus(0)).toBe('critical');
    expect(marginStatus(-0.01)).toBe('negative');
    expect(marginStatus(NaN)).toBe('unknown');
    expect(toSignedLabel(1, 1)).toBe(1);
    expect(toSignedLabel(0, 1)).toBe(-1);
    expect(Number.isNaN(toSignedLabel(null, 1))).toBe(true);
  });

  it('pseudo-residuals follow the negative gradient (DoD #3)', () => {
    expect(pseudoResidualLS(2, 1.5)).toBeCloseTo(0.5, 12);
    expect(pseudoResidualLogloss(1, 0.9)).toBeCloseTo(0.1, 12);
    expect(Number.isNaN(pseudoResidualLS(null, 1))).toBe(true);
    const { lengths, sides } = spokeLengths([1, -2, 0]);
    expect(lengths[1]).toBeCloseTo(1.5, 12);
    expect(sides).toEqual([1, -1, 0]);
  });

  it('quantization snaps to the 256-bin lattice (DoD #4)', () => {
    expect(quantizeToBin(0.1, [0.25, 0.5, 0.75])).toBe(0);
    expect(quantizeToBin(0.9, [0.25, 0.5, 0.75])).toBe(3);
    expect(quantizeToBin(NaN, [0.5])).toBe(-1);
    expect(quantizedCellId(1, 2)).toBe(1 * 256 + 2);
    expect(quantizedCellId(-1, 0)).toBe(-1);
  });
});

describe('Boosting point-intelligence SVG components', () => {
  const stages = [
    { phiX: 1, phiY: 0 },
    { phiX: 0, phiY: 1 },
    { phiX: 1, phiY: 1 },
  ];

  it('push trail replays played stages solid and ghosts the rest (DoD #1)', () => {
    const { container } = render(
      <svg>
        <BoostingShapPushTrail2D origin={{ x: 0, y: 0 }} stages={stages} uptoStage={1} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2); // played + ghost
    expect(screen.getByText('T=1/3')).toBeTruthy();
  });

  it('margin rings split safe / critical / negative channels (DoD #2)', () => {
    const { container } = render(
      <svg>
        <FunctionalMarginRing2D
          rings={[
            { x: 0, y: 0, margin: 2.1, status: 'safe', color: '#22c55e', label: 's' },
            { x: 1, y: 1, margin: 0.4, status: 'critical', color: '#fb923c', label: 'c' },
            { x: 2, y: 2, margin: -0.5, status: 'negative', color: '#f43d5e', label: 'n' },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('circle').length).toBe(1);
    expect(container.querySelectorAll('path[d]').length).toBe(3); // critical + negative ring + cross
    expect(container.querySelector('animate')).toBeTruthy();
  });

  it('gradient spokes batch up/down with hard-sample labels (DoD #3)', () => {
    const { container } = render(
      <svg>
        <PseudoResidualSpoke2D
          spokes={[
            { x: 0, y: 0, length: 1.5, direction: 1, magnitude: 2 },
            { x: 1, y: 1, length: 0.5, direction: -1, magnitude: 0.5 },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(4); // 2 spokes + 2 markers
    expect(screen.getByText('|r|=2.00')).toBeTruthy();
  });

  it('quantized hexbins split hot cells from calm traffic (DoD #4)', () => {
    const cells = [0, 1, 2].map((i) => ({
      centerX: i,
      centerY: 0,
      radius: 0.45,
      count: 5,
      dominantClass: 0,
      meanResidual: 0,
      errorRate: 0,
    }));
    const { container } = render(
      <svg>
        <QuantizedBinHexbin2D cells={cells} heat={[9, 1, null]} mode="gradient" />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2);
  });

  it('renders nothing for empty inputs', () => {
    const { container } = render(
      <svg>
        <BoostingShapPushTrail2D origin={{ x: 0, y: 0 }} stages={[]} />
        <FunctionalMarginRing2D rings={[]} />
        <PseudoResidualSpoke2D spokes={[]} />
        <QuantizedBinHexbin2D cells={[]} heat={[]} mode="traffic" />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

describe('BoostingPointIntelOverlays2D orchestrator', () => {
  const baseBoosting = {
    algorithm: 'XGBClassifier',
    xFeature: 'a',
    yFeature: 'b',
    xFeatureIdx: 0,
    yFeatureIdx: 1,
    nEstimators: 15,
    treesExported: 1,
    segments: [],
    segmentsTruncated: false,
    leaves: [],
    leavesTruncated: false,
    binEdgesXRaw: [] as number[],
    binEdgesYRaw: [] as number[],
    binLinesXWorld: [-2.5, 0, 2.5],
    binLinesYWorld: [-2.5, 0, 2.5],
    activeBin: null,
    obliviousXCutsRaw: [] as number[],
    obliviousYCutsRaw: [] as number[],
    obliviousXLinesWorld: [] as number[],
    obliviousYLinesWorld: [] as number[],
    obliviousCells: [],
    obliviousTruncated: false,
    obliviousRule: '',
    shapVector: null,
    noAxisSplits: false,
    formula: 'boosted',
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

  it('badges safe / critical / negative margins from scores (DoD #2)', () => {
    const { container } = render(
      <svg>
        <BoostingPointIntelOverlays2D
          boosting2d={{ ...baseBoosting, family: 'xgboost' }}
          boostFrame={null}
          points={[
            pt({ world_x: -2, raw_z_actual: 1, raw_z_pred: 1 }),
            pt({ world_x: 0, raw_z_actual: 1, raw_z_pred: 1 }),
            pt({ world_x: 2, raw_z_actual: 0, raw_z_pred: 1, residual_error: 1 }),
          ]}
          activeFrameIndex={1}
          hexbinCells={[]}
          isClassification
          positiveLabel={1}
          classColor={() => '#22c55e'}
          margins={[2.1, 0.4, 0.5]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('circle').length).toBe(1); // safe
    expect(container.querySelector('path[stroke-dasharray]')).toBeTruthy(); // critical pulse
  });

  it('draws live regressor spokes from signed residuals (DoD #3)', () => {
    const { container } = render(
      <svg>
        <BoostingPointIntelOverlays2D
          boosting2d={{ ...baseBoosting, family: 'sklearn_gbm' }}
          boostFrame={null}
          points={[
            pt({ world_x: -2, residual_error: 1.5 }),
            pt({ world_x: 2, residual_error: -0.5 }),
          ]}
          activeFrameIndex={0}
          hexbinCells={[]}
          isClassification={false}
          positiveLabel={null}
          classColor={() => '#fff'}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('|r|=1.50')).toBeTruthy();
  });

  it('snaps hist_gbm hexbins to the lattice and replays staged pushes', () => {
    useStage3DStore.setState({ showHexbins: true });
    const { container } = render(
      <svg>
        <BoostingPointIntelOverlays2D
          boosting2d={{ ...baseBoosting, family: 'hist_gbm' }}
          boostFrame={null}
          points={[pt({ world_x: -2.5, world_y: 0 }), pt({ world_x: 2.5, world_y: 0 })]}
          activeFrameIndex={1}
          hexbinCells={[
            { centerX: -2.5, centerY: 0, radius: 0.45, count: 8, dominantClass: 0, meanResidual: 0, errorRate: 0 },
            { centerX: 0, centerY: 0, radius: 0.45, count: 2, dominantClass: 0, meanResidual: 0, errorRate: 0 },
            { centerX: 2.5, centerY: 0, radius: 0.45, count: 5, dominantClass: 0, meanResidual: 0, errorRate: 0 },
          ]}
          isClassification
          positiveLabel={1}
          classColor={() => '#fff'}
          activeStagedPush={[
            { phiX: 1, phiY: 0 },
            { phiX: 0, phiY: 1 },
          ]}
        />
      </svg>,
    );
    expect(screen.getByText('T=2/2')).toBeTruthy();
    expect(container.querySelectorAll('.quant-bin-hexbin-2d path[d]').length).toBe(2);
  });

  it('marks the active CatBoost oblivious cell with a symmetry grid', () => {
    const { container } = render(
      <svg>
        <BoostingPointIntelOverlays2D
          boosting2d={{
            ...baseBoosting,
            family: 'catboost',
            obliviousCells: [
              {
                cellId: 3,
                leafIndex: 3,
                bitmask: '11',
                boxRaw: [-5, 5, -5, 5] as [number, number, number, number],
                boxWorld: [-5, 5, -5, 5] as [number, number, number, number],
                isActive: true,
              },
            ],
          }}
          boostFrame={null}
          points={[pt({})]}
          activeFrameIndex={0}
          hexbinCells={[]}
          isClassification
          positiveLabel={1}
          classColor={() => '#fff'}
        />
      </svg>,
    );
    expect(container.querySelectorAll('.oblivious-symmetry-2d line').length).toBe(6);
  });

  it('renders nothing for non-boosting or missing payloads', () => {
    const { container } = render(
      <svg>
        <BoostingPointIntelOverlays2D
          boosting2d={{ ...baseBoosting, family: 'gnb' }}
          boostFrame={null}
          points={[pt({})]}
          activeFrameIndex={0}
          hexbinCells={[]}
          isClassification
          positiveLabel={1}
          classColor={() => '#fff'}
        />
        <BoostingPointIntelOverlays2D
          boosting2d={null}
          boostFrame={null}
          points={[pt({})]}
          activeFrameIndex={0}
          hexbinCells={[]}
          isClassification
          positiveLabel={1}
          classColor={() => '#fff'}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { useStage3DStore } from '../../store/useStage3DStore';
import { ManhattanStepTrail2D } from '../../components/surfaces_2d/ManhattanStepTrail2D';
import { ForestConsensusRing2D } from '../../components/surfaces_2d/ForestConsensusRing2D';
import { AdaBoostWeightHalo2D } from '../../components/surfaces_2d/AdaBoostWeightHalo2D';
import { LeafMinoritySpoke2D } from '../../components/surfaces_2d/LeafMinoritySpoke2D';
import { LeafBoundedHexbin2D } from '../../components/surfaces_2d/LeafBoundedHexbin2D';
import { TreesPointIntelOverlays2D } from '../../components/surfaces_2d/TreesPointIntelOverlays2D';
import {
  adaboostHaloRadius,
  adaboostRingScale,
  consensusThickness,
  forestAgreement,
  hexPurity,
  isFuzzyConsensus,
  isLeafMinority,
  manhattanElbow,
  manhattanLength,
  manhattanTrailPath,
  stepwiseResidual,
} from '../../components/surfaces_2d/treesPointIntel2dMath';

afterEach(() => {
  useStage3DStore.setState({ showHexbins: false, showErrorsOnly: false });
});

describe('treesPointIntel2dMath (Phase 53 spec §2)', () => {
  it('elbows route L-shaped hops and lengths are L1', () => {
    expect(manhattanElbow({ x: 0, y: 0 }, { x: 3, y: 4 })).toEqual({ x: 3, y: 0 });
    expect(manhattanElbow({ x: 0, y: 0 }, { x: 3, y: 4 }, false)).toEqual({ x: 0, y: 4 });
    expect(manhattanElbow({ x: NaN, y: 0 }, { x: 1, y: 1 })).toBeNull();
    expect(manhattanLength({ x: 0, y: 0 }, { x: 3, y: 4 })).toBeCloseTo(7, 12);
    expect(Number.isNaN(manhattanLength({ x: NaN, y: 0 }, { x: 1, y: 1 }))).toBe(true);
    expect(manhattanTrailPath([{ x: 0, y: 0 }, { x: 3, y: 4 }], 0, 1)).toContain('L');
  });

  it('agreement is the top vote share; fuzzy below 60% (DoD #1)', () => {
    expect(forestAgreement([[0, 0, 0, 1, 1]])).toEqual([0.6]);
    expect(forestAgreement([[0, 0, 1, 1]])).toEqual([0.5]);
    expect(Number.isNaN(forestAgreement([[]])[0])).toBe(true);
    expect(isFuzzyConsensus(0.59)).toBe(true);
    expect(isFuzzyConsensus(0.6)).toBe(false);
    expect(isFuzzyConsensus(NaN)).toBe(false);
    expect(consensusThickness(1)).toBeCloseTo(1, 12);
    expect(consensusThickness(0.5)).toBeCloseTo(1.5, 12);
  });

  it('adaboost halos scale with sqrt weight ratio (DoD #2)', () => {
    expect(adaboostRingScale(4, 1)).toBeCloseTo(2, 12);
    expect(adaboostRingScale(1, 1)).toBeCloseTo(1, 12);
    expect(adaboostRingScale(NaN, 1)).toBeCloseTo(1, 12);
    expect(adaboostHaloRadius(2, 0.14)).toBeCloseTo(0.28, 12);
  });

  it('minority flags disagree with the leaf majority (DoD #3)', () => {
    expect(isLeafMinority(0, 1)).toBe(true);
    expect(isLeafMinority(1, 1)).toBe(false);
    expect(isLeafMinority(null, 1)).toBe(true);
    expect(stepwiseResidual(2, 1.5)).toBeCloseTo(0.5, 12);
    expect(Number.isNaN(stepwiseResidual(null, 1))).toBe(true);
  });

  it('hex purity scores leaf blocks; degenerate bins stay NaN (DoD #4)', () => {
    expect(hexPurity(10, 10)).toBeCloseTo(1, 12);
    expect(hexPurity(5, 10)).toBeCloseTo(0.5, 12);
    expect(Number.isNaN(hexPurity(3, 0))).toBe(true);
    expect(Number.isNaN(hexPurity(NaN, 5))).toBe(true);
  });
});

describe('Trees point-intelligence SVG components', () => {
  it('manhattan trail renders two lines per hop plus the pulse box', () => {
    const { container } = render(
      <svg>
        <ManhattanStepTrail2D
          points={[
            { x: 0, y: 0 },
            { x: 3, y: 4 },
          ]}
          activeFrame={1}
          pulseBox={[0, 5, -5, 5]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('line').length).toBe(2);
    expect(container.querySelectorAll('rect').length).toBe(1);
  });

  it('consensus rings batch solid paths and double fuzzy pairs', () => {
    const { container } = render(
      <svg>
        <ForestConsensusRing2D
          rings={[
            { x: 0, y: 0, maxShare: 1, label: 'sure' },
            { x: 1, y: 1, maxShare: 0.52, label: 'split' },
          ]}
        />
      </svg>,
    );
    // 1 solid batched path + 2 fuzzy paths (outer + inner).
    expect(container.querySelectorAll('path[d]').length).toBe(3);
    expect(container.querySelector('path[stroke-dasharray]')).toBeTruthy();
  });

  it('weight halos batch into one path capped for legibility', () => {
    const { container } = render(
      <svg>
        <AdaBoostWeightHalo2D
          halos={[
            { x: 0, y: 0, scale: 1, label: 'a' },
            { x: 1, y: 1, scale: 2.5, label: 'b' },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(1);
  });

  it('minority spokes batch with impurity labels', () => {
    const { container } = render(
      <svg>
        <LeafMinoritySpoke2D
          spokes={[
            { x1: 0, y1: 0, x2: 1, y2: 1, impurity: 0.48 },
            { x1: 0, y1: 0, x2: -1, y2: -1 },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2); // spoke batch + marker
    expect(screen.getByText('G=0.48')).toBeTruthy();
  });

  it('leaf hexbins split pure fills from mixed warnings', () => {
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
        <LeafBoundedHexbin2D cells={cells} purities={[1, 0.5, null]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2);
  });

  it('renders nothing for empty inputs', () => {
    const { container } = render(
      <svg>
        <ManhattanStepTrail2D points={[]} activeFrame={0} />
        <ForestConsensusRing2D rings={[]} />
        <AdaBoostWeightHalo2D halos={[]} />
        <LeafMinoritySpoke2D spokes={[]} />
        <LeafBoundedHexbin2D cells={[]} purities={[]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('line').length).toBe(0);
  });
});

describe('TreesPointIntelOverlays2D orchestrator', () => {
  const trees2d = {
    family: 'decision_tree' as const,
    algorithm: 'DecisionTreeClassifier',
    xFeature: 'a',
    yFeature: 'b',
    xFeatureIdx: 0,
    yFeatureIdx: 1,
    nEstimators: 1,
    treesExported: 1,
    segments: [],
    segmentsTruncated: false,
    leaves: [
      {
        leafId: 0,
        treeIndex: 0,
        boxRaw: [-5, 0, -5, 5] as [number, number, number, number],
        boxWorld: [-5, 0, -5, 5] as [number, number, number, number],
        prediction: 0 as string | number | null,
        samples: 60,
        impurity: 0,
        rule: 'a <= 0',
        isActive: false,
      },
      {
        leafId: 1,
        treeIndex: 0,
        boxRaw: [0, 5, -5, 5] as [number, number, number, number],
        boxWorld: [0, 5, -5, 5] as [number, number, number, number],
        prediction: 1 as string | number | null,
        samples: 40,
        impurity: 0.48,
        rule: 'a > 0',
        isActive: true,
      },
    ],
    leavesTruncated: false,
    stages: [],
    stagesTruncated: false,
    baseValue: null,
    noAxisSplits: false,
    multiOutput: false,
    formula: 'axis-aligned partitions',
  };
  // p1 sits in the impure leaf but belongs to class 0 → minority strand.
  const points = [
    {
      frame_index: 0,
      original_row_index: 0,
      world_x: -2.5,
      world_y: 0,
      world_z_pred: 0,
      world_z_actual: 0,
      raw_x: -2.5,
      raw_y: 0,
      raw_z_pred: 0,
      raw_z_actual: 0,
      residual_error: 0,
    },
    {
      frame_index: 1,
      original_row_index: 7,
      world_x: 2.5,
      world_y: 0,
      world_z_pred: 1,
      world_z_actual: 0,
      raw_x: 2.5,
      raw_y: 0,
      raw_z_pred: 1,
      raw_z_actual: 0,
      residual_error: 1,
    },
  ];
  const hexbinCells = [
    { centerX: -2.5, centerY: 0, radius: 0.45, count: 8, dominantClass: 0, meanResidual: 0, errorRate: 0 },
    { centerX: 2.5, centerY: 0, radius: 0.45, count: 6, dominantClass: 1, meanResidual: 0.4, errorRate: 0.5 },
  ];

  it('renders trail, impurity rings and the minority spoke', () => {
    useStage3DStore.setState({ showHexbins: true });
    const { container } = render(
      <svg>
        <TreesPointIntelOverlays2D
          trees2d={trees2d}
          points={points}
          activeFrameIndex={1}
          hexbinCells={hexbinCells}
          isClassification
          classColor={() => '#22c55e'}
        />
      </svg>,
    );
    // 1 hop → 2 orthogonal lines + pulse rect on the destination leaf.
    expect(container.querySelectorAll('line').length).toBe(2);
    expect(container.querySelectorAll('rect').length).toBe(1);
    // Impurity proxy: pure leaf solid, 0.48 leaf fuzzy (share 0.52 < 60%).
    expect(container.querySelector('path[stroke-dasharray]')).toBeTruthy();
    expect(screen.getByText('G=0.48')).toBeTruthy();
    // Leaf-bounded hexbins: pure fill + mixed warning.
    expect(container.querySelectorAll('.leaf-hexbin-2d path[d]').length).toBe(2);
  });

  it('drives exact-vote rings and weight halos when arrays are provided', () => {
    const { container } = render(
      <svg>
        <TreesPointIntelOverlays2D
          trees2d={{ ...trees2d, family: 'random_forest' }}
          points={points}
          activeFrameIndex={1}
          hexbinCells={[]}
          isClassification
          classColor={() => '#fff'}
          forestVotes={[
            [0, 0, 1, 1],
            [1, 1, 1, 1],
          ]}
          sampleWeights={[0.5, 2.0]}
        />
      </svg>,
    );
    // 50/50 split vote → fuzzy double ring; halos batch into one path.
    expect(container.querySelector('path[stroke-dasharray]')).toBeTruthy();
    expect(container.querySelectorAll('.adaboost-halo-2d path[d]').length).toBe(1);
  });

  it('renders nothing for non-tree or missing payloads', () => {
    const { container } = render(
      <svg>
        <TreesPointIntelOverlays2D
          trees2d={{ ...trees2d, family: 'gnb' }}
          points={points}
          activeFrameIndex={0}
          hexbinCells={[]}
          isClassification
          classColor={() => '#fff'}
        />
        <TreesPointIntelOverlays2D
          trees2d={null}
          points={points}
          activeFrameIndex={0}
          hexbinCells={[]}
          isClassification
          classColor={() => '#fff'}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('line').length).toBe(0);
  });
});

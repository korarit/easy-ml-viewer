import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { SvmSupportVectorHalo2D } from '../../components/surfaces_2d/SvmSupportVectorHalo2D';
import { SvrSlackSpoke2D } from '../../components/surfaces_2d/SvrSlackSpoke2D';
import { KnnVoteArcRing2D } from '../../components/surfaces_2d/KnnVoteArcRing2D';
import { KnnDensityHexbin2D } from '../../components/surfaces_2d/KnnDensityHexbin2D';
import { DistancePointIntelOverlays2D } from '../../components/surfaces_2d/DistancePointIntelOverlays2D';
import {
  arcDonutPath,
  densityTiers,
  detectNeighborSwaps,
  diamondPath,
  discordantRanks,
  haloColor,
  haloDiamondRadius,
  knnCentroid2D,
  knnDensity,
  svrSlackSpoke,
  voteArcSegments,
} from '../../components/surfaces_2d/distancePointIntel2dMath';

describe('distancePointIntel2dMath (Phase 51 spec §2)', () => {
  it('centroid is the vote-weighted mean of the K neighbors', () => {
    const c = knnCentroid2D([
      { x: 0, y: 0, voteWeight: 0.6 },
      { x: 10, y: 0, voteWeight: 0.4 },
    ])!;
    expect(c.x).toBeCloseTo(4, 12);
    expect(c.y).toBeCloseTo(0, 12);
    expect(knnCentroid2D([])).toBeNull();
  });

  it('swap detection flags entered vs exited neighbors', () => {
    expect(detectNeighborSwaps([1, 2, 3], [2, 3, 4])).toEqual({ entered: [4], exited: [1] });
    expect(detectNeighborSwaps([5], [5])).toEqual({ entered: [], exited: [] });
  });

  it('halo colors follow the |α| status (DoD #1)', () => {
    expect(haloColor('violates_margin')).toBe('#fb923c');
    expect(haloColor('on_margin')).toBe('#7dd3fc');
    expect(haloColor('support')).toBe('#94a3b8');
    expect(haloDiamondRadius(1)).toBeGreaterThan(haloDiamondRadius(0));
    expect(haloDiamondRadius(99)).toBeCloseTo(0.26, 12);
  });

  it('vote arcs tile the full turn in descending share order (DoD #3)', () => {
    const segs = voteArcSegments({ a: 0.6, b: 0.4 }, (label) => label);
    expect(segs.map((s) => s.label)).toEqual(['a', 'b']);
    expect(segs.reduce((s, e) => s + e.sweep, 0)).toBeCloseTo(1, 12);
    expect(segs[0]!.start).toBe(0);
    expect(segs[1]!.start).toBeCloseTo(0.6, 12);
    expect(voteArcSegments({}, (label) => label)).toEqual([]);
  });

  it('donut arcs are closed paths with a full-turn guard', () => {
    const d = arcDonutPath(0, 0, 0.19, 0.12, 0, 0.6);
    expect(d.startsWith('M')).toBe(true);
    expect(d.endsWith('Z')).toBe(true);
    expect(arcDonutPath(0, 0, 0.19, 0.12, 0, 1)).toContain('A');
    expect(diamondPath(1, 2, 0.2)).toContain('M');
  });

  it('SVR slack is non-zero only outside the ε-tube (DoD #2)', () => {
    const out = svrSlackSpoke(1.0, 0.0, 0.1);
    expect(out.outside).toBe(true);
    expect(out.xi).toBeCloseTo(0.9, 12);
    expect(out.xiStar).toBe(0);
    const over = svrSlackSpoke(-1.0, 0.0, 0.1);
    expect(over.outside).toBe(true);
    expect(over.xiStar).toBeCloseTo(0.9, 12);
    expect(svrSlackSpoke(0.05, 0.0, 0.1).outside).toBe(false);
    expect(svrSlackSpoke(null, 0.0, 0.1).outside).toBe(false);
  });

  it('KNN density matches ρ = K/(π·R̄²) (spec §2.4)', () => {
    expect(knnDensity(5, 1)).toBeCloseTo(5 / Math.PI, 12);
    expect(knnDensity(10, 1)).toBeCloseTo(2 * knnDensity(5, 1), 12);
    expect(Number.isNaN(knnDensity(5, 0))).toBe(true);
    expect(Number.isNaN(knnDensity(0, 1))).toBe(true);
  });

  it('discordance flags neighbors voting against the truth', () => {
    const nbs = [
      { rank: 1, label: 'a' as const },
      { rank: 2, label: 'b' as const },
      { rank: 3, label: 'b' as const },
    ];
    expect(discordantRanks(nbs, 'a')).toEqual([2, 3]);
    expect(discordantRanks(nbs, 'b')).toEqual([1]);
  });

  it('density tiers split top/bottom terciles', () => {
    const { dense, sparse } = densityTiers([1, 2, 3, 4, 5, 6]);
    expect(dense.length).toBe(2);
    expect(sparse.length).toBe(2);
    expect(dense.every((i) => [4, 5].includes(i))).toBe(true);
    expect(densityTiers([null, NaN])).toEqual({ dense: [], sparse: [] });
  });
});

describe('Distance point-intelligence SVG components', () => {
  it('halos batch diamonds by status with dual rings', () => {
    const { container } = render(
      <svg>
        <SvmSupportVectorHalo2D
          halos={[
            { trainIndex: 1, xWorld: 0, yWorld: 0, dualCoef: 1.0, haloScale: 1, status: 'violates_margin' },
            { trainIndex: 2, xWorld: 1, yWorld: 1, dualCoef: 0.3, haloScale: 0.3, status: 'on_margin' },
          ]}
        />
      </svg>,
    );
    // 2 batched diamond paths + 2 dual-ring circles.
    expect(container.querySelectorAll('path[d]').length).toBe(2);
    expect(container.querySelectorAll('circle').length).toBe(2);
  });

  it('slack spokes split under/over-tube into two batched paths', () => {
    const { container } = render(
      <svg>
        <SvrSlackSpoke2D
          spokes={[
            { x: 0, y: 0, length: 1.0, side: -1, slack: 0.9 },
            { x: 1, y: 1, length: 0.5, side: 1, slack: 0.45 },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(4); // 2 spokes + 2 markers
  });

  it('vote arcs render one donut per class share', () => {
    const segs = voteArcSegments({ a: 0.6, b: 0.4 }, (label) => (label === 'a' ? '#22c55e' : '#ef4444'));
    const { container } = render(
      <svg>
        <KnnVoteArcRing2D x={0} y={0} segments={segs} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2);
    expect(screen.getByTitle('Class a: 60.0% of the K votes')).toBeTruthy();
  });

  it('density hexbins outline dense islands and sparse borders', () => {
    const cells = [0, 1, 2, 3, 4, 5].map((i) => ({
      centerX: i,
      centerY: 0,
      radius: 0.45,
      count: i + 1,
      dominantClass: 0,
      meanResidual: 0,
      errorRate: 0,
    }));
    const { container } = render(
      <svg>
        <KnnDensityHexbin2D cells={cells} densities={[1, 2, 3, 4, 5, 6]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2);
  });

  it('renders nothing for empty inputs', () => {
    const { container } = render(
      <svg>
        <SvmSupportVectorHalo2D halos={[]} />
        <SvrSlackSpoke2D spokes={[]} />
        <KnnVoteArcRing2D x={0} y={0} segments={[]} />
        <KnnDensityHexbin2D cells={[]} densities={[]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

describe('DistancePointIntelOverlays2D orchestrator', () => {
  const points = [
    {
      frame_index: 0,
      original_row_index: 7,
      world_x: 0,
      world_y: 0,
      world_z_pred: 1,
      world_z_actual: 0,
      raw_x: 0.1,
      raw_y: 0.2,
      raw_z_pred: 1,
      raw_z_actual: 0,
      residual_error: 1,
    },
  ];

  it('renders halos, vote arcs and swap ripples for distance models', () => {
    const { container } = render(
      <svg>
        <DistancePointIntelOverlays2D
          svm2d={{
            modelType: 'SVC',
            kernel: 'rbf',
            gamma: 0.5,
            c: 1,
            degree: 3,
            coef0: 0,
            isRegressor: false,
            isBinary: true,
            nClasses: 2,
            nSupportTotal: 1,
            isLinearNoSv: false,
            halos: [{ trainIndex: 3, xWorld: 1, yWorld: 1, dualCoef: 1, haloScale: 1, status: 'violates_margin' }],
            halosTruncated: false,
          }}
          knnFrame={{
            xFeature: 'a',
            yFeature: 'b',
            rowIndex: 0,
            queryWorld: [0, 0],
            radarRadiusRaw: 1,
            radarRadiusWorld: [1, 1],
            neighbors: [
              { trainIndex: 1, xWorld: 0.5, yWorld: 0, distanceFull: 0.5, distance2d: 0.5, voteWeight: 0.6, rank: 1, label: 1 },
              { trainIndex: 2, xWorld: -0.5, yWorld: 0, distanceFull: 0.5, distance2d: 0.5, voteWeight: 0.4, rank: 2, label: 0 },
            ],
            votes: { '1': 0.6, '0': 0.4 },
            predicted: 1,
            k: 2,
            weightsMode: 'uniform',
            metric: 'euclidean',
            isRadiusModel: false,
            outlier: false,
            displayCapped: false,
            totalNeighbors: 2,
            computationTimeMs: 1,
          }}
          points={points}
          activeFrameIndex={0}
          hexbinCells={[]}
          classColor={() => '#22c55e'}
        />
      </svg>,
    );
    // Diamond path + vote arcs + elastic line + ripple circles + discordance path.
    expect(container.querySelectorAll('path[d]').length).toBeGreaterThanOrEqual(3);
    expect(container.querySelectorAll('circle').length).toBeGreaterThanOrEqual(2);
    expect(container.querySelector('line')).toBeTruthy();
  });

  it('renders nothing for non-distance models', () => {
    const { container } = render(
      <svg>
        <DistancePointIntelOverlays2D
          svm2d={null}
          knnFrame={null}
          points={[]}
          activeFrameIndex={0}
          hexbinCells={[]}
          classColor={() => '#fff'}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
    expect(container.querySelectorAll('line').length).toBe(0);
  });
});

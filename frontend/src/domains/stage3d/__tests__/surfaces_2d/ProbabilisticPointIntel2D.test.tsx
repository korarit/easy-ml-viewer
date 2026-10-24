import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { BayesEntropyCometTrail2D } from '../../components/surfaces_2d/BayesEntropyCometTrail2D';
import { PosteriorConfidenceRing2D } from '../../components/surfaces_2d/PosteriorConfidenceRing2D';
import { CentroidAttractorSpoke2D } from '../../components/surfaces_2d/CentroidAttractorSpoke2D';
import { GaussianDivergenceHexbin2D } from '../../components/surfaces_2d/GaussianDivergenceHexbin2D';
import { ProbabilisticPointIntelOverlays2D } from '../../components/surfaces_2d/ProbabilisticPointIntelOverlays2D';
import {
  confidenceRingRadius,
  covFromStdRotation,
  entropyTrailColor,
  gaussianMixturePdf,
  gaussianPosterior2D,
  hexAreaRaw,
  isAmbiguous,
  logLikelihoodRatio,
  mahalanobis2D,
  posteriorRingScale,
  shannonEntropy,
  worldToRaw,
} from '../../components/surfaces_2d/probabilisticPointIntel2dMath';

const GAUSSIANS = [
  { meanRaw: [0, 0] as [number, number], stdRaw: [1, 1] as [number, number], rotationRad: 0, prior: 0.5, label: 0 },
  { meanRaw: [5, 0] as [number, number], stdRaw: [1, 1] as [number, number], rotationRad: 0, prior: 0.5, label: 1 },
];

describe('probabilisticPointIntel2dMath (Phase 52 spec §2)', () => {
  it('entropy is 1 bit at 50/50 and 0 when certain', () => {
    expect(shannonEntropy([0.5, 0.5])).toBeCloseTo(1, 12);
    expect(shannonEntropy([1, 0])).toBeCloseTo(0, 12);
    expect(Number.isNaN(shannonEntropy([0, 0]))).toBe(true);
    expect(Number.isNaN(shannonEntropy([]))).toBe(true);
  });

  it('rebuilds covariances under the extractor convention', () => {
    // std ASCENDING [minor, major], rotation = major-axis direction.
    const c0 = covFromStdRotation([1, 2], 0);
    expect(c0[0][0]).toBeCloseTo(4, 12);
    expect(c0[1][1]).toBeCloseTo(1, 12);
    const c90 = covFromStdRotation([1, 2], Math.PI / 2);
    expect(c90[0][0]).toBeCloseTo(1, 12);
    expect(c90[1][1]).toBeCloseTo(4, 12);
  });

  it('posterior concentrates on the nearby class and sums to 1', () => {
    const atZero = gaussianPosterior2D(0, 0, GAUSSIANS);
    expect(atZero[0]).toBeGreaterThan(0.99);
    expect(atZero[0]! + atZero[1]!).toBeCloseTo(1, 12);
    const mid = gaussianPosterior2D(2.5, 0, GAUSSIANS);
    expect(mid[0]).toBeCloseTo(0.5, 9);
    expect(mid[1]).toBeCloseTo(0.5, 9);
  });

  it('ring scale is full at certainty and half at a coin flip', () => {
    expect(posteriorRingScale(1)).toBeCloseTo(1, 12);
    expect(posteriorRingScale(0.5)).toBeCloseTo(0.75, 12);
    expect(confidenceRingRadius(1, 0.14)).toBeCloseTo(0.14, 12);
  });

  it('ambiguity glow triggers near max entropy (DoD #1)', () => {
    expect(isAmbiguous(0.95, 2)).toBe(true);
    expect(isAmbiguous(0.2, 2)).toBe(false);
    expect(entropyTrailColor(0.95, 2, '#22c55e')).toBe('#e879f9');
    expect(entropyTrailColor(0.2, 2, '#22c55e')).toBe('#22c55e');
  });

  it('mahalanobis is 0 at the centroid and 1 at 1σ', () => {
    expect(mahalanobis2D(0, 0, [0, 0], [[1, 0], [0, 1]])).toBeCloseTo(0, 12);
    expect(mahalanobis2D(3, 0, [0, 0], [[9, 0], [0, 1]])).toBeCloseTo(1, 12);
  });

  it('mixture pdf peaks at centroids and LLR signs the paradox', () => {
    expect(gaussianMixturePdf(0, 0, GAUSSIANS)).toBeGreaterThan(gaussianMixturePdf(20, 20, GAUSSIANS));
    expect(logLikelihoodRatio(2, 1)).toBeCloseTo(1, 12);
    expect(Number.isNaN(logLikelihoodRatio(NaN, 1))).toBe(true);
  });

  it('world/raw mapping inverts and hex areas stay positive', () => {
    expect(worldToRaw(-5, 0, 10)).toBeCloseTo(0, 12);
    expect(worldToRaw(5, 0, 10)).toBeCloseTo(10, 12);
    expect(hexAreaRaw(0.45, 10, 10)).toBeGreaterThan(0);
    expect(Number.isNaN(hexAreaRaw(0, 10, 10))).toBe(true);
  });
});

describe('Probabilistic point-intelligence SVG components', () => {
  it('entropy trail renders one line per in-window segment', () => {
    const { container } = render(
      <svg>
        <BayesEntropyCometTrail2D
          points={[
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
          ]}
          colors={['#22c55e', '#22c55e', '#e879f9']}
          activeFrame={2}
        />
      </svg>,
    );
    const lines = container.querySelectorAll('line');
    expect(lines.length).toBe(2);
    expect(lines[1]?.getAttribute('stroke')).toBe('#e879f9');
  });

  it('confidence rings scale by maxP and pulse when ambiguous', () => {
    const { container } = render(
      <svg>
        <PosteriorConfidenceRing2D
          rings={[
            { x: 0, y: 0, maxProba: 1, ambiguous: false, color: '#22c55e', label: 'sure' },
            { x: 1, y: 1, maxProba: 0.52, ambiguous: true, color: '#e879f9', label: 'unsure' },
          ]}
        />
      </svg>,
    );
    const circles = container.querySelectorAll('circle');
    expect(circles.length).toBe(2);
    expect(Number(circles[0]?.getAttribute('r'))).toBeGreaterThan(Number(circles[1]?.getAttribute('r')));
    expect(container.querySelector('animate')).toBeTruthy();
  });

  it('mini sigma halos echo the class covariance orientation', () => {
    const { container } = render(
      <svg>
        <PosteriorConfidenceRing2D
          rings={[{ x: 0, y: 0, maxProba: 0.9, ambiguous: false, halo: { rx: 1, ry: 0.5, rotDeg: -30 }, color: '#22c55e', label: 'qda' }]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('ellipse').length).toBe(1);
  });

  it('centroid spokes batch true/pred paths with LLR labels', () => {
    const { container } = render(
      <svg>
        <CentroidAttractorSpoke2D
          spokes={[
            { x1: 0, y1: 0, x2: 1, y2: 1, kind: 'true', llr: -2.5 },
            { x1: 0, y1: 0, x2: -1, y2: -1, kind: 'pred' },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(4); // 2 spokes + 2 markers
    expect(screen.getByText('LLR=-2.50')).toBeTruthy();
  });

  it('divergence hexbins split over/under-dense tiers', () => {
    const cells = [0, 1, 2, 3, 4, 5].map((i) => ({
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
        <GaussianDivergenceHexbin2D cells={cells} divergences={[9, 8, 0, 0, -8, -9]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2);
  });

  it('renders nothing for empty inputs', () => {
    const { container } = render(
      <svg>
        <BayesEntropyCometTrail2D points={[]} colors={[]} activeFrame={0} />
        <PosteriorConfidenceRing2D rings={[]} />
        <CentroidAttractorSpoke2D spokes={[]} />
        <GaussianDivergenceHexbin2D cells={[]} divergences={[]} />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('line').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

describe('ProbabilisticPointIntelOverlays2D orchestrator', () => {
  const ellipses = [
    {
      classIndex: 0,
      label: 0 as string | number | null,
      meanRaw: [0, 0] as [number, number],
      stdRaw: [1, 1] as [number, number],
      rotationRad: 0,
      centerWorld: [0, 0] as [number, number],
      radiiWorld1Sigma: [1, 1] as [number, number],
      prior: 0.5,
    },
    {
      classIndex: 1,
      label: 1 as string | number | null,
      meanRaw: [5, 0] as [number, number],
      stdRaw: [1, 1] as [number, number],
      rotationRad: 0,
      centerWorld: [5, 0] as [number, number],
      radiiWorld1Sigma: [1, 1] as [number, number],
      prior: 0.5,
    },
  ];
  const prob2d = {
    family: 'gnb' as const,
    algorithm: 'GaussianNB',
    xFeature: 'a',
    yFeature: 'b',
    xFeatureIdx: 0,
    yFeatureIdx: 1,
    nClasses: 2,
    classes: [0, 1] as Array<string | number>,
    ellipses,
    ellipsesTruncated: false,
    isCovarianceApproximated: false,
    discriminantLine: null,
    conic: null,
    discrete: null,
    formula: 'GNB',
  };
  // Point sits on class-1's centroid but is actually class 0 → paradox case.
  const points = [
    {
      frame_index: 0,
      original_row_index: 3,
      world_x: 5,
      world_y: 0,
      world_z_pred: 1,
      world_z_actual: 0,
      raw_x: 5,
      raw_y: 0,
      raw_z_pred: 1,
      raw_z_actual: 0,
      residual_error: 1,
    },
  ];

  it('renders trail, rings and paradox spokes for Gaussian families', () => {
    const { container } = render(
      <svg>
        <ProbabilisticPointIntelOverlays2D
          prob2d={prob2d}
          points={points}
          activeFrameIndex={0}
          hexbinCells={[]}
          xRangeRaw={[0, 10]}
          yRangeRaw={[0, 10]}
          classColor={() => '#22c55e'}
        />
      </svg>,
    );
    // Entropy trail needs ≥2 points; rings + dual centroid spokes render.
    expect(container.querySelectorAll('circle').length).toBeGreaterThanOrEqual(1);
    expect(container.querySelectorAll('path[d]').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/LLR=/)).toBeTruthy();
  });

  it('renders nothing for discrete NB or missing payloads', () => {
    const { container } = render(
      <svg>
        <ProbabilisticPointIntelOverlays2D
          prob2d={{ ...prob2d, family: 'discrete_nb', ellipses: [] }}
          points={points}
          activeFrameIndex={0}
          hexbinCells={[]}
          xRangeRaw={[0, 10]}
          yRangeRaw={[0, 10]}
          classColor={() => '#fff'}
        />
        <ProbabilisticPointIntelOverlays2D
          prob2d={null}
          points={points}
          activeFrameIndex={0}
          hexbinCells={[]}
          xRangeRaw={[0, 10]}
          yRangeRaw={[0, 10]}
          classColor={() => '#fff'}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

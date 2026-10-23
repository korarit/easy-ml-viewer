import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { LinearOrthogonalSpoke2D } from '../../components/surfaces_2d/LinearOrthogonalSpoke2D';
import { LinearMarginDualRing2D } from '../../components/surfaces_2d/LinearMarginDualRing2D';
import { LinearResidualStem2D } from '../../components/surfaces_2d/LinearResidualStem2D';
import { LinearHexbinAlignment2D } from '../../components/surfaces_2d/LinearHexbinAlignment2D';
import {
  errorDistanceCorrelation,
  expit,
  hexbinBoundaryFlags,
  isMarginViolator,
  logisticRingThickness,
  orthogonalSpoke,
  trailSignColor,
  trailSigns,
  worldLine,
  zScoreRaw,
} from '../../components/surfaces_2d/linearPointIntel2dMath';

describe('linearPointIntel2dMath (Phase 50 spec §2)', () => {
  it('converts the raw plane to world coords preserving the line', () => {
    // Raw: 2x + 4y − 6 = 0 on rects [0,3]×[0,3] → world spans 10.
    const line = worldLine(2, 4, -6, [0, 3], [0, 3]);
    expect(line.degenerate).toBe(false);
    // Raw point (0, 1.5) lies on the line → world point must too.
    const wx = ((0 - 0) / 3) * 10 - 5;
    const wy = ((1.5 - 0) / 3) * 10 - 5;
    expect(line.W1 * wx + line.W2 * wy + line.B).toBeCloseTo(0, 9);
  });

  it('flags degenerate axes instead of a bogus line', () => {
    expect(worldLine(1, 1, 0, [2, 2], [0, 1]).degenerate).toBe(true);
    expect(worldLine(0, 0, 1, [0, 1], [0, 1]).degenerate).toBe(true);
  });

  it('orthogonal foot lands on the line with exact d_perp', () => {
    const line = worldLine(3, 4, -5, [-5, 5], [-5, 5]);
    const s = orthogonalSpoke(1, 1, line)!;
    expect(s).not.toBeNull();
    expect(line.W1 * s.x2 + line.W2 * s.y2 + line.B).toBeCloseTo(0, 9);
    // Raw-space check: w=(3,4), b=−5, p=(1,1) → d = 2/5.
    expect(s.dPerp).toBeCloseTo(0.4, 9);
  });

  it('trail signs follow the weight dot step', () => {
    const pts = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
    ];
    expect(trailSigns(pts, 1, 0)).toEqual([1, 0]);
    expect(trailSigns(pts, -1, 0)).toEqual([-1, 0]);
    expect(trailSignColor(1)).toBe('#22c55e');
    expect(trailSignColor(-1)).toBe('#fb923c');
  });

  it('margin mask is inclusive at |z| = 1 and thickness peaks at P = 0.5', () => {
    expect(isMarginViolator(1.0)).toBe(true);
    expect(isMarginViolator(-1.0)).toBe(true);
    expect(isMarginViolator(1.0001)).toBe(false);
    expect(logisticRingThickness(0.5)).toBeCloseTo(0.08, 9);
    expect(logisticRingThickness(0.0)).toBeCloseTo(0.02, 9);
    expect(logisticRingThickness(1.0)).toBeCloseTo(0.02, 9);
  });

  it('expit inverts the log-odds score', () => {
    expect(expit(0)).toBeCloseTo(0.5, 12);
    expect(expit(zScoreRaw(1, 1, 3, 4, -5))).toBeCloseTo(expit(2), 12);
  });

  it('flags only hexbins crossed by the line', () => {
    const line = worldLine(1, 0, 0, [-5, 5], [-5, 5]); // x = 0
    const cells = [
      { centerX: -5, centerY: 0, radius: 0.45, count: 3, dominantClass: 0, meanResidual: 0, errorRate: 0 },
      { centerX: 0, centerY: 0, radius: 0.45, count: 9, dominantClass: 1, meanResidual: 0, errorRate: 0.5 },
      { centerX: 5, centerY: 0, radius: 0.45, count: 3, dominantClass: 0, meanResidual: 0, errorRate: 0 },
    ];
    expect(hexbinBoundaryFlags(cells, line)).toEqual([false, true, false]);
  });

  it('error gradient is negative for separable bins', () => {
    expect(errorDistanceCorrelation([0.1, 0.5, 1.0, 2.0], [0.9, 0.6, 0.2, 0.0])).toBeLessThan(-0.9);
    expect(Number.isNaN(errorDistanceCorrelation([1, 1, 1], [0.5, 0.2, 0.1]))).toBe(true);
  });
});

describe('Linear point-intelligence SVG components', () => {
  it('orthogonal spokes batch into one path with d_perp labels', () => {
    const { container } = render(
      <svg>
        <LinearOrthogonalSpoke2D
          spokes={[
            { x1: 0, y1: 0, x2: 1, y2: 0, dPerp: 0.28 },
            { x1: 2, y1: 2, x2: 2, y2: 3, dPerp: 0.5 },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(2); // batched + marker
    expect(screen.getByText('d⊥=0.28')).toBeTruthy();
  });

  it('margin rings pulse amber for violators and size logistic rings by thickness', () => {
    const { container } = render(
      <svg>
        <LinearMarginDualRing2D
          rings={[
            { x: 0, y: 0, kind: 'amber' },
            { x: 1, y: 1, kind: 'logistic', thickness: 0.08 },
          ]}
        />
      </svg>,
    );
    const circles = container.querySelectorAll('circle');
    expect(circles.length).toBe(2);
    expect(circles[1]?.getAttribute('stroke-width')).toBe('0.08');
  });

  it('residual stems split over/under into two batched paths', () => {
    const { container } = render(
      <svg>
        <LinearResidualStem2D
          stems={[
            { x: 0, y: 0, length: 1.0, direction: 1 },
            { x: 1, y: 1, length: 0.5, direction: -1 },
          ]}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(4); // 2 stems + 2 markers
  });

  it('hexbin alignment outlines only crossed cells', () => {
    const line = worldLine(1, 0, 0, [-5, 5], [-5, 5]);
    const cells = [
      { centerX: -5, centerY: 0, radius: 0.45, count: 3, dominantClass: 0, meanResidual: 0, errorRate: 0 },
      { centerX: 0, centerY: 0, radius: 0.45, count: 9, dominantClass: 1, meanResidual: 0, errorRate: 0.5 },
    ];
    const { container } = render(
      <svg>
        <LinearHexbinAlignment2D cells={cells} line={line} />
      </svg>,
    );
    const path = container.querySelector('path[d]');
    expect(path?.getAttribute('d')).toContain('M0');
    expect(path?.getAttribute('stroke-dasharray')).toBeTruthy();
  });

  it('renders nothing for degenerate lines or empty inputs', () => {
    const bad = worldLine(0, 0, 1, [0, 1], [0, 1]);
    const { container } = render(
      <svg>
        <LinearOrthogonalSpoke2D spokes={[]} />
        <LinearMarginDualRing2D rings={[]} />
        <LinearResidualStem2D stems={[]} />
        <LinearHexbinAlignment2D
          cells={[{ centerX: 0, centerY: 0, radius: 0.4, count: 1, dominantClass: 0, meanResidual: 0, errorRate: 0 }]}
          line={bad}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path[d]').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

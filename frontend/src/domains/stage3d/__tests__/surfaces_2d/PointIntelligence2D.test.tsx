import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { CometTrail2D } from '../../components/surfaces_2d/CometTrail2D';
import { DualRingPoint2D } from '../../components/surfaces_2d/DualRingPoint2D';
import { HexbinOverlay2D } from '../../components/surfaces_2d/HexbinOverlay2D';
import { PointDisplayToolbar2D } from '../../components/surfaces_2d/PointDisplayToolbar2D';
import { ResidualSpoke2D } from '../../components/surfaces_2d/ResidualSpoke2D';
import {
  GHOST_ALPHA,
  cometAlpha,
  cometStrokeWidth,
  cometTrailAlphas,
  contrastStroke,
  errorsOnlyFrames,
  hexPath,
  hexbinFill,
  residualSpokeEnd,
} from '../../components/surfaces_2d/pointIntelligence2dMath';

const pts = [
  { x: -4, y: -4 },
  { x: -2, y: -2 },
  { x: 0, y: 0 },
  { x: 2, y: 2 },
  { x: 4, y: 4 },
];

describe('pointIntelligence2dMath (Phase 49 spec §2)', () => {
  it('head frame is full opacity, tail decays monotonically', () => {
    const alphas = cometTrailAlphas(11, 5, 4);
    expect(alphas[5]).toBeCloseTo(1.0, 3);
    expect(alphas.slice(1, 6)).toEqual([...alphas.slice(1, 6)].sort((a, b) => a - b));
  });

  it('matches the exponential kernel formula at d=2', () => {
    const sigma = 4 / 2.5;
    const expected = 0.05 + 0.95 * Math.exp(-(2 * 2) / (2 * sigma * sigma));
    expect(cometAlpha(3, 5, 4)).toBeCloseTo(expected, 6);
  });

  it('out-of-window frames collapse to ghosts and strokes widen to the head', () => {
    const alphas = cometTrailAlphas(10, 5, 2);
    expect(alphas[0]).toBe(GHOST_ALPHA);
    expect(alphas[9]).toBe(GHOST_ALPHA);
    expect(cometStrokeWidth(5, 5, 4)).toBeGreaterThan(cometStrokeWidth(1, 5, 4));
    expect(cometStrokeWidth(5, 5, 4)).toBeCloseTo(0.09, 6);
  });

  it('contrast stroke picks dark-on-light and light-on-dark', () => {
    expect(contrastStroke(true)).toBe('#0b0f14');
    expect(contrastStroke(false)).toBe('#ffffff');
  });

  it('errors-only filter keeps mispredictions and non-finite frames', () => {
    expect(errorsOnlyFrames([0, 1, 0.2, null, NaN], 0.5)).toEqual([1, 3, 4]);
    expect(errorsOnlyFrames([0, 0.1], 0.5)).toEqual([]);
  });

  it('spoke length scales with |residual| and NaN collapses to a dot', () => {
    const [ex, ey] = residualSpokeEnd(1, 2, 2.0);
    expect(Math.hypot(ex - 1, ey - 2)).toBeGreaterThan(0.4);
    expect(residualSpokeEnd(0, 0, NaN)).toEqual([0, 0]);
  });

  it('hex path is a closed hexagon and fills vary by mode', () => {
    const d = hexPath(1, 2, 0.5);
    expect(d.split('L').length).toBe(6);
    expect(d.endsWith('Z')).toBe(true);
    const cell = { count: 8, maxBinCount: 10, errorRate: 0.9, dominantClass: 1 as number | null };
    expect(hexbinFill('error', cell, () => '#000')).toContain('255,');
    expect(hexbinFill('class', cell, (c) => (c === 1 ? '#ff0000' : '#000'))).toBe('#ff0000');
    const lo = hexbinFill('count', { ...cell, count: 1 }, () => '#000');
    const hi = hexbinFill('count', cell, () => '#000');
    expect(lo).not.toBe(hi);
  });
});

describe('Point intelligence SVG components', () => {
  it('comet trail renders one line per in-window segment', () => {
    const { container } = render(
      <svg>
        <CometTrail2D points={pts} activeFrame={4} window={2} />
      </svg>,
    );
    // Segments [2,3] and [3,4] only.
    expect(container.querySelectorAll('line').length).toBe(2);
  });

  it('comet trail renders nothing for a single point', () => {
    const { container } = render(
      <svg>
        <CometTrail2D points={[{ x: 0, y: 0 }]} activeFrame={0} />
      </svg>,
    );
    expect(container.querySelectorAll('line').length).toBe(0);
  });

  it('dual-ring renders core + halo + contrast and a cross on error', () => {
    const { container } = render(
      <svg>
        <DualRingPoint2D x={0} y={0} actualColor="#3b82f6" predColor="#ef4444" mispredicted />
      </svg>,
    );
    // Contrast perimeter + prediction halo.
    expect(container.querySelectorAll('circle').length).toBe(3);
    expect(screen.getByText('×')).toBeTruthy();
  });

  it('dual-ring rect glyph renders a rect core', () => {
    const { container } = render(
      <svg>
        <DualRingPoint2D x={0} y={0} actualColor="#3b82f6" predColor="#3b82f6" glyph="rect" />
      </svg>,
    );
    expect(container.querySelectorAll('rect').length).toBe(1);
    expect(container.querySelectorAll('circle').length).toBe(2);
  });

  it('residual spokes batch into a single dashed path', () => {
    const { container } = render(
      <svg>
        <ResidualSpoke2D segments={[[0, 0, 1, 1], [2, 2, 3, 1]]} />
      </svg>,
    );
    const path = container.querySelector('path[d]');
    expect(path).toBeTruthy();
    expect(path?.getAttribute('stroke-dasharray')).toBeTruthy();
  });

  it('hexbin overlay batches cells and respects the cap', () => {
    const cells = Array.from({ length: 10 }, (_, i) => ({
      centerX: i,
      centerY: 0,
      radius: 0.4,
      count: 10 - i,
      dominantClass: i % 2,
      meanResidual: 0,
      errorRate: i / 10,
    }));
    const { container } = render(
      <svg>
        <HexbinOverlay2D cells={cells} maxBinCount={10} mode="error" cellCap={4} />
      </svg>,
    );
    // ≤4 cells rendered, batched into ≤24 fill-bucket paths.
    expect(container.querySelectorAll('path').length).toBeLessThanOrEqual(4);
    expect(container.querySelectorAll('path').length).toBeGreaterThan(0);
  });

  it('toolbar exposes T/R/E/H toggles and fires clicks', () => {
    const { container } = render(<PointDisplayToolbar2D hexbinMode="count" onHexbinMode={() => {}} />);
    expect(screen.getByRole('toolbar', { name: /point display/i })).toBeTruthy();
    for (const label of ['Trails', 'Dual-Ring', 'Errors', 'Hexbin']) {
      expect(screen.getByRole('button', { name: new RegExp(label) })).toBeTruthy();
    }
    const trails = screen.getByRole('button', { name: /trails/i });
    expect(trails.getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(screen.getByRole('button', { name: /hexbin/i }));
    expect(container.querySelector('select')).toBeTruthy();
  });
});

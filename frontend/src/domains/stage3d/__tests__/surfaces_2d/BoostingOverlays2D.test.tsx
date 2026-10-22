import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { HistGbmQuantizedBins2D } from '../../components/surfaces_2d/HistGbmQuantizedBins2D';
import { XgboostTreeShapVector2D } from '../../components/surfaces_2d/XgboostTreeShapVector2D';
import { CatBoostObliviousGrid2D } from '../../components/surfaces_2d/CatBoostObliviousGrid2D';
import {
  scaleShapVector,
  binCellRect,
  findStrongestLeaf,
  SHAP_ARROW_MAX_LEN,
} from '../../components/surfaces_2d/boosting2dMath';

describe('boosting2dMath helpers', () => {
  it('clamps unbounded log-odds magnitudes to the stage', () => {
    const big = scaleShapVector(40, -30);
    expect(Math.hypot(big.dx, big.dy)).toBeLessThanOrEqual(SHAP_ARROW_MAX_LEN + 1e-9);
    expect(big.magnitude).toBeCloseTo(50, 6);
    const zero = scaleShapVector(0, 0);
    expect(zero).toEqual({ dx: 0, dy: 0, magnitude: 0 });
  });

  it('preserves push direction after clamping', () => {
    const v = scaleShapVector(3, 4);
    expect(v.dx / v.dy).toBeCloseTo(3 / 4, 9);
    expect(v.dx).toBeGreaterThan(0);
    expect(v.dy).toBeGreaterThan(0);
  });

  it('bounds the active hist bin cell by neighboring grid lines', () => {
    const rect = binCellRect([-2.5, 0, 2.5], [-2.5, 0, 2.5], 1, 2);
    expect(rect).not.toBeNull();
    expect(rect!.x).toBeCloseTo(-2.5, 9);
    expect(rect!.width).toBeCloseTo(2.5, 9);
  });

  it('picks the best-first leaf by |leaf value|', () => {
    const mk = (leafId: number, leafValue: number) => ({
      leafId,
      treeIndex: 0,
      boxRaw: [0, 1, 0, 1] as [number, number, number, number],
      boxWorld: [0, 1, 0, 1] as [number, number, number, number],
      leafValue,
      rule: '',
      isActive: false,
    });
    expect(findStrongestLeaf([mk(1, 0.2), mk(2, -1.5), mk(3, 0.9)])?.leafId).toBe(2);
    expect(findStrongestLeaf([])).toBeNull();
  });
});

describe('HistGbmQuantizedBins2D Component Rendering', () => {
  it('batches 255-line grids into 2 paths and highlights the active bin', () => {
    const xs = Array.from({ length: 9 }, (_, i) => -5 + i * 1.25);
    const ys = Array.from({ length: 9 }, (_, i) => -5 + i * 1.25);
    const { container } = render(
      <svg>
        <HistGbmQuantizedBins2D
          linesXWorld={xs}
          linesYWorld={ys}
          activeBin={[4, 5]}
          activeRowIndex={7}
          visible={true}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path').length).toBe(2);
    const rects = container.querySelectorAll('rect');
    expect(rects.length).toBe(1);
    expect(rects[0].querySelector('title')?.textContent).toContain('(x=4, y=5)');
    expect(rects[0].querySelector('title')?.textContent).toContain('Row #7');
  });

  it('renders nothing when hidden', () => {
    const { container } = render(
      <svg>
        <HistGbmQuantizedBins2D linesXWorld={[0]} linesYWorld={[0]} activeBin={null} activeRowIndex={null} visible={false} />
      </svg>,
    );
    expect(container.querySelectorAll('path').length).toBe(0);
    expect(container.querySelectorAll('rect').length).toBe(0);
  });
});

describe('XgboostTreeShapVector2D Component Rendering', () => {
  it('draws a push arrow from the origin with exact phi labels', () => {
    const { container } = render(
      <svg>
        <XgboostTreeShapVector2D originX={1} originY={-1} phiX={0.84} phiY={-1.22} visible={true} />
      </svg>,
    );
    const line = container.querySelector('line');
    expect(line).toBeTruthy();
    // +phi_x pushes right, -phi_y pushes down (SVG y is flipped via wy).
    expect(Number(line?.getAttribute('x2') ?? 0)).toBeGreaterThan(1);
    const label = container.querySelector('text')?.textContent ?? '';
    expect(label).toContain('+0.84');
    expect(label).toContain('-1.22');
  });
});

describe('CatBoostObliviousGrid2D Component Rendering', () => {
  const cells = [
    {
      cellId: 0,
      leafIndex: 0,
      bitmask: '00',
      boxRaw: [0, 1, 0, 1] as [number, number, number, number],
      boxWorld: [-5, 0, -5, 0] as [number, number, number, number],
      isActive: false,
    },
    {
      cellId: 1,
      leafIndex: 1,
      bitmask: '01',
      boxRaw: [1, 2, 0, 1] as [number, number, number, number],
      boxWorld: [0, 5, -5, 0] as [number, number, number, number],
      isActive: true,
    },
  ];

  it('renders grid paths, cell rects and the active cell glow', () => {
    const { container } = render(
      <svg>
        <CatBoostObliviousGrid2D
          linesXWorld={[0]}
          linesYWorld={[0]}
          cells={cells}
          obliviousRule="f0 > 1 AND f1 > 2"
          activeRowIndex={3}
          visible={true}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path').length).toBe(2);
    // 2 cells + 1 active glow rect.
    expect(container.querySelectorAll('rect').length).toBe(3);
    const glow = container.querySelector('[aria-label="Active oblivious cell"]');
    expect(glow?.querySelector('title')?.textContent).toContain('bitmask 01');
    expect(glow?.querySelector('title')?.textContent).toContain('Row #3');
  });
});

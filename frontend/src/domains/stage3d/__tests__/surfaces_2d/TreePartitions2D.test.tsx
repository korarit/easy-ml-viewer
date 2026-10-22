import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { TreePartitions2D } from '../../components/surfaces_2d/TreePartitions2D';

const SEGMENTS = [
  {
    axis: 'x' as const,
    threshold: 25.0,
    nodeId: 0,
    treeIndex: 0,
    depth: 0,
    segmentRaw: [25, 0, 25, 200] as [number, number, number, number],
    segmentWorld: [0, -5, 0, 5] as [number, number, number, number],
  },
  {
    axis: 'y' as const,
    threshold: 140.0,
    nodeId: 2,
    treeIndex: 0,
    depth: 1,
    segmentRaw: [0, 140, 100, 140] as [number, number, number, number],
    segmentWorld: [-5, 1, 5, 1] as [number, number, number, number],
  },
];

const LEAVES = [
  {
    leafId: 1,
    treeIndex: 0,
    boxRaw: [0, 25, 0, 140] as [number, number, number, number],
    boxWorld: [-5, 0, -5, 1] as [number, number, number, number],
    prediction: 0,
    samples: 42,
    impurity: 0.12,
    rule: 'feature_0 <= 25 AND feature_1 <= 140',
    isActive: false,
  },
];

describe('TreePartitions2D Component Rendering', () => {
  it('renders batched cut paths per axis and leaf rects with rule tooltips', () => {
    const { container } = render(
      <svg>
        <TreePartitions2D
          segments={SEGMENTS}
          leaves={LEAVES}
          activePoint={null}
          activeRowIndex={null}
          visible={true}
        />
      </svg>,
    );

    // E45.1 — cuts batch by axis into exactly 2 paths (not one line per cut).
    const paths = container.querySelectorAll('path');
    expect(paths.length).toBe(2);
    // One rect per leaf box, carrying the rule in its tooltip.
    const rects = container.querySelectorAll('rect');
    expect(rects.length).toBe(1);
    expect(rects[0].querySelector('title')?.textContent).toContain('feature_0 <= 25');
    // No active glow without an active point.
    expect(container.querySelectorAll('circle').length).toBe(0);
  });

  it('highlights the active leaf box containing the query point', () => {
    const { container } = render(
      <svg>
        <TreePartitions2D
          segments={SEGMENTS}
          leaves={LEAVES}
          activePoint={{ x: -2.5, y: -2 }}
          activeRowIndex={42}
          visible={true}
        />
      </svg>,
    );

    // Leaf rect + glowing active rect.
    expect(container.querySelectorAll('rect').length).toBe(2);
    const glow = container.querySelector('[aria-label="Active leaf box"]');
    expect(glow).toBeTruthy();
    expect(glow?.querySelector('title')?.textContent).toContain('Row #42');
  });

  it('renders nothing when hidden', () => {
    const { container } = render(
      <svg>
        <TreePartitions2D
          segments={SEGMENTS}
          leaves={LEAVES}
          activePoint={{ x: 0, y: 0 }}
          activeRowIndex={0}
          visible={false}
        />
      </svg>,
    );
    expect(container.querySelectorAll('path').length).toBe(0);
    expect(container.querySelectorAll('rect').length).toBe(0);
  });
});

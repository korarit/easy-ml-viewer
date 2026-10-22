import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { KnnRadar2D } from '../../components/surfaces_2d/KnnRadar2D';

describe('KnnRadar2D SVG Geometry Rendering', () => {
  it('renders radar circle and K lines connecting to neighbors', () => {
    const neighbors = [
      { refIndex: 1, x: 10, y: 0, distance2D: 10, voteWeight: 0.6, targetClass: 0, rank: 1 },
      { refIndex: 2, x: 0, y: 15, distance2D: 15, voteWeight: 0.4, targetClass: 1, rank: 2 },
    ];

    const { container } = render(
      <svg>
        <KnnRadar2D
          centerX={0}
          centerY={0}
          radiusX={15}
          radiusY={15}
          neighbors={neighbors}
          visible={true}
        />
      </svg>
    );

    // NOTE (Phase 43 implementation): the radar is an <ellipse>, not a
    // <circle> — world X/Y spans are usually anisotropic, so a raw circle
    // maps to an axis-aligned ellipse. Neighbor markers stay <circle>s.
    // วงกลมจุดเพื่อนบ้าน 2 วง + วงรีเรดาร์ 1 วง
    const circles = container.querySelectorAll('circle');
    expect(circles.length).toBe(2);
    expect(container.querySelectorAll('ellipse').length).toBe(1);

    // เส้นใยเลเซอร์ 2 เส้น
    const lines = container.querySelectorAll('line');
    expect(lines.length).toBe(2);
  });

  it('renders nothing when hidden', () => {
    const { container } = render(
      <svg>
        <KnnRadar2D
          centerX={0}
          centerY={0}
          radiusX={15}
          radiusY={15}
          neighbors={[]}
          visible={false}
        />
      </svg>
    );
    expect(container.querySelectorAll('ellipse').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
    expect(container.querySelectorAll('line').length).toBe(0);
  });
});

import { describe, it, expect } from 'vitest';
import { buildDecisionSvg } from '../components/svgBuilder';

describe('buildDecisionSvg offline vector export', () => {
  it('produces well-formed SVG with decision + point layers', () => {
    const svg = buildDecisionSvg({
      xLabel: 'income',
      yLabel: 'debt_ratio',
      boundarySegments: [[-1, 0, 1, 0]],
      dataPoints: [{ x: 0, y: 0.5, actual: 0 }],
      width: 800,
      height: 600,
    });
    expect(svg).toContain('<svg');
    expect(svg).toContain('id="decision-boundaries"');
    expect(svg).toContain('id="trajectory-points"');
    expect(svg).toContain('width="800"');
    expect(svg).toContain('income');
  });

  it('escapes adversarial column names so the XML stays valid', () => {
    const svg = buildDecisionSvg({
      xLabel: 'a<b>&"c"',
      yLabel: 'y',
      boundarySegments: [],
      dataPoints: [],
    });
    expect(svg).not.toContain('a<b>');
    expect(svg).toContain('&amp;');
    expect(svg).toContain('&lt;');
  });

  it('renders Phase-56 point-intelligence layers as true vectors', () => {
    const svg = buildDecisionSvg({
      xLabel: 'income',
      yLabel: 'debt_ratio',
      boundarySegments: [[-1, 0, 1, 0]],
      dataPoints: [{ x: 0, y: 0.5, actual: 1 }],
      hexbinCells: [
        { cx: 0, cy: 0, r: 0.45, count: 40, errorRate: 0.25, domClass: 1 },
        { cx: 2, cy: 1, r: 0.3, count: 5, errorRate: 1, domClass: 0 },
      ],
      hexbinMode: 'error',
      dualRings: [{ x: 0, y: 0.5, actualColor: '#f43f5e', predColor: '#10b981', isError: true }],
      trailPolyline: [[-2, -2], [0, 0], [1, 1]],
      densitySegments: [[-1, -1, 1, 1]],
    });
    expect(svg).toContain('id="hexbin-grid"');
    expect(svg).toContain('<polygon');
    expect(svg).toContain('id="dual-rings"');
    expect(svg).toContain('id="comet-trail"');
    expect(svg).toContain('id="density-contours"');
  });

  it('omits point-intelligence groups when no layers are provided (legacy parity)', () => {
    const svg = buildDecisionSvg({
      xLabel: 'x',
      yLabel: 'y',
      boundarySegments: [],
      dataPoints: [],
    });
    expect(svg).not.toContain('id="hexbin-grid"');
    expect(svg).not.toContain('id="dual-rings"');
    expect(svg).not.toContain('id="comet-trail"');
  });
});

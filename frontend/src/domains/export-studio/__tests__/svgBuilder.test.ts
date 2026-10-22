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
});

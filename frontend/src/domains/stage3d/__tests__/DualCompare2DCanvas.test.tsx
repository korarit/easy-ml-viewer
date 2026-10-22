import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DualCompare2DCanvas } from '../components/DualCompare2DCanvas';

const boundariesA = [
  { threshold: 0.5, classIndex: 1, kind: 'decision', lineSegments: [[-4, -4, 4, 4]] },
];
const boundariesB = [
  { threshold: 0.5, classIndex: 1, kind: 'decision', lineSegments: [[-4, 4, 4, -4]] },
];

describe('DualCompare2DCanvas Split & Superimposed Modes', () => {
  it('renders both model boundary layers and disagreement overlay', () => {
    const { container } = render(
      <DualCompare2DCanvas
        modelAName="RandomForest"
        modelBName="SVM-RBF"
        boundariesA={boundariesA}
        boundariesB={boundariesB}
        disagreementAreaPct={14.2}
        showConflictHatch
      />,
    );

    expect(screen.getByText(/14\.2% Disagreement Area/i)).toBeTruthy();
    expect(container.querySelector('[data-testid="conflict-hatch-pattern"]')).toBeTruthy();
    // Superimposed: both boundaries share one stage (2 crossing lines).
    expect(container.querySelectorAll('line').length).toBeGreaterThanOrEqual(2);
  });

  it('shows the consensus notice when models fully agree', () => {
    render(
      <DualCompare2DCanvas
        modelAName="A"
        modelBName="B"
        boundariesA={boundariesA}
        boundariesB={boundariesA}
        disagreementAreaPct={0}
        consensusNotice="100% Consensus: Models are completely in agreement"
        conflictFrameIndices={[]}
      />,
    );
    expect(screen.getByText(/100% Consensus/i)).toBeTruthy();
    expect(screen.getByText(/No frame conflicts/i)).toBeTruthy();
  });

  it('renders side-by-side stages and conflict navigation', () => {
    const { container } = render(
      <DualCompare2DCanvas
        modelAName="A"
        modelBName="B"
        boundariesA={boundariesA}
        boundariesB={boundariesB}
        disagreementAreaPct={50}
        layout="side-by-side"
        conflictFrameIndices={[3, 9]}
      />,
    );
    // Two half-width SVG stages.
    expect(container.querySelectorAll('svg').length).toBe(2);
    expect(screen.getByText(/Next Conflict \(Alt\+D\)/i)).toBeTruthy();
  });
});

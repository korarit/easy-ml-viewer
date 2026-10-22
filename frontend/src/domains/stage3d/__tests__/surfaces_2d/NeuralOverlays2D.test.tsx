import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MlpNeuronHyperplanes2D } from '../../components/surfaces_2d/MlpNeuronHyperplanes2D';
import { OnnxLatentManifold2D } from '../../components/surfaces_2d/OnnxLatentManifold2D';
import { PipelineSpaceToggle2D } from '../../components/surfaces_2d/PipelineSpaceToggle2D';
import {
  batchHyperplanePaths,
  fitLatentToWorld,
  scaledRulerLabel,
} from '../../components/surfaces_2d/neural2dMath';

describe('neural2dMath helpers', () => {
  it('batches active and dead cuts into separate paths', () => {
    const hyperplanes = [
      { neuronIdx: 0, w1: 1, w2: 0, bEff: 0, segmentRaw: [0, -5, 0, 5] as [number, number, number, number], segmentWorld: [0, -5, 0, 5] as [number, number, number, number], isDead: false, isAlwaysActive: true },
      { neuronIdx: 1, w1: 0, w2: 1, bEff: 10, segmentRaw: null, segmentWorld: null, isDead: true, isAlwaysActive: false },
      { neuronIdx: 2, w1: 1, w2: 1, bEff: -1, segmentRaw: [-4, 3, 3, -4] as [number, number, number, number], segmentWorld: [-4, 3, 3, -4] as [number, number, number, number], isDead: true, isAlwaysActive: false },
    ];
    const paths = batchHyperplanePaths(hyperplanes);
    // Only segments with world geometry batch; null segments are skipped.
    expect(paths.active).toContain('M0');
    expect(paths.dead).toContain('M-4');
  });

  it('fits latent clusters into the inset while preserving separation', () => {
    const pts: Array<[number, number]> = [
      [-3, 0.1], [-2.9, -0.1],
      [3, 0.1], [2.9, -0.1],
    ];
    const fitted = fitLatentToWorld(pts, 1.6, 2.4, 3.0);
    expect(fitted).toHaveLength(4);
    for (const p of fitted) {
      expect(p.x).toBeGreaterThanOrEqual(1.6);
      expect(p.x).toBeLessThanOrEqual(4.6);
    }
    // Left cluster stays left of the right cluster.
    expect(fitted[0].x).toBeLessThan(fitted[2].x);
    expect(fitted[1].x).toBeLessThan(fitted[3].x);
  });

  it('maps ruler positions through the exact scaler affine', () => {
    // raw range [0, 10], mu=5, sigma=2 → world 0 (raw 5) maps to 0.
    expect(scaledRulerLabel(0, 0, 10, 5, 2)).toBe('0');
    expect(scaledRulerLabel(5, 0, 10, 5, 2)).toBe('2.5');
    expect(scaledRulerLabel(-5, 0, 10, 5, 2)).toBe('-2.5');
  });
});

describe('MlpNeuronHyperplanes2D Component Rendering', () => {
  const hyperplanes = [
    { neuronIdx: 0, w1: 1, w2: 0, bEff: 0, segmentRaw: [0, -5, 0, 5] as [number, number, number, number], segmentWorld: [0, -5, 0, 5] as [number, number, number, number], isDead: false, isAlwaysActive: true },
    { neuronIdx: 1, w1: 0.5, w2: 0.5, bEff: -1, segmentRaw: [-3, 1, 1, -3] as [number, number, number, number], segmentWorld: [-3, 1, 1, -3] as [number, number, number, number], isDead: true, isAlwaysActive: false },
  ];

  it('draws active cuts solid cyan and dead cuts dashed red', () => {
    const { container } = render(
      <svg>
        <MlpNeuronHyperplanes2D hyperplanes={hyperplanes} />
      </svg>,
    );
    const paths = container.querySelectorAll('path');
    expect(paths.length).toBe(2);
    expect(paths[0].getAttribute('stroke')).toBe('#22d3ee');
    expect(paths[1].getAttribute('stroke')).toBe('#f87171');
    expect(paths[1].getAttribute('stroke-dasharray')).toBeTruthy();
  });

  it('renders nothing when both layers are hidden', () => {
    const { container } = render(
      <svg>
        <MlpNeuronHyperplanes2D hyperplanes={hyperplanes} showCuts={false} showDead={false} />
      </svg>,
    );
    expect(container.querySelectorAll('path').length).toBe(0);
  });
});

describe('OnnxLatentManifold2D Component Rendering', () => {
  const latent = {
    operator: 'Gemm',
    chain: ['ai.onnx::Gemm'],
    nNodes: 5,
    latentAvailable: true,
    latentPoints2d: [[-3, 0.1], [3, -0.1]] as Array<[number, number]>,
    latentExplainedVariance: [0.97, 0.02],
    penultimateWidth: 8,
    formula: 'penultimate latent SVD plane (8D → 2D, 2 pts)',
  };

  it('draws an inset frame with one dot per latent point', () => {
    const { container } = render(
      <svg>
        <OnnxLatentManifold2D latent={latent} />
      </svg>,
    );
    expect(container.querySelectorAll('rect').length).toBe(1);
    expect(container.querySelectorAll('circle').length).toBe(2);
    expect(container.querySelector('text')?.textContent).toContain('Latent SVD 2D');
  });

  it('renders nothing when no latent plane is available', () => {
    const { container } = render(
      <svg>
        <OnnxLatentManifold2D latent={{ ...latent, latentAvailable: false, latentPoints2d: [] }} />
      </svg>,
    );
    expect(container.querySelectorAll('rect').length).toBe(0);
    expect(container.querySelectorAll('circle').length).toBe(0);
  });
});

describe('PipelineSpaceToggle2D Component Rendering', () => {
  const pipeline = {
    hasToggle: true,
    scalerKind: 'standard',
    innerFamily: 'LogisticRegression',
    innerAlgorithm: 'LogisticRegression',
    muX: 0,
    sigmaX: 1,
    muY: 0,
    sigmaY: 1,
    scaledXRange: [-2, 2] as [number, number],
    scaledYRange: [-2, 2] as [number, number],
    formula: "x'=(x-0)/1 [standard]",
  };

  it('renders Raw and Scaled pills and fires the switch', () => {
    let mode: string | null = null;
    const { container } = render(
      <svg>
        <PipelineSpaceToggle2D pipeline={pipeline} mode="raw" onChange={(m) => { mode = m; }} />
      </svg>,
    );
    const buttons = container.querySelectorAll('[role="button"]');
    expect(buttons.length).toBe(2);
    (buttons[1] as SVGElement).dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(mode).toBe('scaled');
  });

  it('renders nothing when the prefix cannot toggle exactly', () => {
    const { container } = render(
      <svg>
        <PipelineSpaceToggle2D pipeline={{ ...pipeline, hasToggle: false }} mode="raw" onChange={() => {}} />
      </svg>,
    );
    expect(container.querySelectorAll('[role="button"]').length).toBe(0);
  });
});

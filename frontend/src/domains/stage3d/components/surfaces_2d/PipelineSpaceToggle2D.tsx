import { wy } from '../DecisionBoundaryContour2D';
import type { PipelineSpace2D } from '../../../projection3d/types/projection.types';

export type PipelineSpaceMode = 'raw' | 'scaled';

export interface PipelineSpaceToggle2DProps {
  pipeline: PipelineSpace2D;
  mode: PipelineSpaceMode;
  onChange: (mode: PipelineSpaceMode) => void;
  showToggle?: boolean;
}

/**
 * Phase 47 — [Raw Space] <-> [Scaled Space] viewport switch.
 *
 * A two-pill SVG toggle flips the ruler frame between raw domain units and
 * the exact scaled coordinates `(v - mu) / sigma` shipped by the backend.
 * In scaled mode a dashed frame marks the scaled data extent
 * (`scaledXRange × scaledYRange` mapped back into the world rect).
 * Pipelines without an order-preserving single scaler render nothing —
 * the contour stays in raw space with an honest badge instead.
 */
export function PipelineSpaceToggle2D({
  pipeline,
  mode,
  onChange,
  showToggle = true,
}: PipelineSpaceToggle2DProps) {
  if (!showToggle || !pipeline.hasToggle) return null;

  const pill = (x: number, label: string, active: boolean, next: PipelineSpaceMode) => (
    <g
      role="button"
      aria-label={`Switch to ${label} space`}
      onClick={(e) => {
        e.stopPropagation();
        onChange(next);
      }}
      style={{ cursor: 'pointer' }}
    >
      <rect
        x={x}
        y={wy(5.32)}
        width={1.15}
        height={0.44}
        rx={0.22}
        fill={active ? '#0e7490' : 'rgba(10,15,22,0.72)'}
        stroke="#22d3ee"
        strokeWidth={0.03}
        opacity={active ? 1 : 0.85}
      />
      <text
        x={x + 0.575}
        y={wy(5.32) + 0.31}
        textAnchor="middle"
        fontSize={0.26}
        fill={active ? '#ffffff' : '#a5f3fc'}
      >
        {label}
      </text>
    </g>
  );

  return (
    <g className="pipeline-space-toggle-2d" aria-label="Pipeline raw-scaled space toggle">
      {pill(-1.25, 'Raw', mode === 'raw', 'raw')}
      {pill(0.05, 'Scaled', mode === 'scaled', 'scaled')}
      <title>{pipeline.formula}</title>
    </g>
  );
}

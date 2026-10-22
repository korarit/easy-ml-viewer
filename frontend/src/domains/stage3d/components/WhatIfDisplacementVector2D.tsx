import { useStage3DStore } from '../store/useStage3DStore';
import { useWhatIfStore } from '../../whatif/store/useWhatIfStore';
import { wy } from './DecisionBoundaryContour2D';

function rawToWorld(raw: number, min: number, max: number): number {
  const span = max - min;
  if (!Number.isFinite(raw) || !Number.isFinite(span) || Math.abs(span) < 1e-12) return 0;
  const t = (raw - min) / span;
  return -5 + Math.min(1.15, Math.max(-0.15, t)) * 10;
}

/**
 * Phase 41 — 2D What-If displacement arrow.
 * Ghost dot = original queued position; solid dot = mutated position from
 * the sandbox sliders; arrow shows Δx = (Δx₁, Δx₂) in real time.
 * Renders only while a What-If session has mutated one of the two planar axes.
 */
export function WhatIfDisplacementVector2D() {
  const points = useStage3DStore((s) => s.points);
  const activeFrameIndex = useStage3DStore((s) => s.activeFrameIndex);
  const axes = useStage3DStore((s) => s.axes);
  const axisXFeatureIndex = useWhatIfStore((s) => s.axisXFeatureIndex);
  const axisYFeatureIndex = useWhatIfStore((s) => s.axisYFeatureIndex);
  const baseRowValues = useWhatIfStore((s) => s.baseRowValues);
  const mutatedValues = useWhatIfStore((s) => s.mutatedValues);

  if (points.length === 0 || baseRowValues.length === 0) return null;
  const active = points[Math.max(0, Math.min(activeFrameIndex, points.length - 1))];
  const hasX = mutatedValues[axisXFeatureIndex] !== undefined;
  const hasY = mutatedValues[axisYFeatureIndex] !== undefined;
  if (!hasX && !hasY) return null;
  if (!axes.x || !axes.y) return null;

  const merged = [...baseRowValues];
  for (const [k, v] of Object.entries(mutatedValues)) merged[Number(k)] = v;
  const mutRawX = merged[axisXFeatureIndex] ?? active.raw_x;
  const mutRawY = merged[axisYFeatureIndex] ?? active.raw_y;
  const mx = rawToWorld(mutRawX, axes.x.raw_min, axes.x.raw_max);
  const my = rawToWorld(mutRawY, axes.y.raw_min, axes.y.raw_max);
  const gx = active.world_x;
  const gy = active.world_y;
  if (Math.hypot(mx - gx, my - gy) < 1e-9) return null;

  const dx = mx - gx;
  const dy = wy(my) - wy(gy);
  const len = Math.max(Math.hypot(dx, dy), 1e-9);
  const ux = dx / len;
  const uy = dy / len;
  const head = 0.22;
  const tipX = mx;
  const tipY = wy(my);

  return (
    <g aria-label="What-If displacement vector" pointerEvents="none">
      {/* Ghost dot (original shadow) */}
      <circle cx={gx} cy={wy(gy)} r={0.16} fill="none" stroke="#9ca3af" strokeWidth={0.05} strokeDasharray="0.1 0.08" opacity={0.9} />
      {/* Displacement arrow */}
      <line x1={gx} y1={wy(gy)} x2={tipX} y2={tipY} stroke="#22d3ee" strokeWidth={0.08} opacity={0.95} />
      <polygon
        points={`${tipX},${tipY} ${tipX - ux * head - uy * head * 0.55},${tipY - uy * head + ux * head * 0.55} ${tipX - ux * head + uy * head * 0.55},${tipY - uy * head - ux * head * 0.55}`}
        fill="#22d3ee"
        opacity={0.95}
      />
      {/* Mutated dot */}
      <circle cx={tipX} cy={tipY} r={0.13} fill="#22d3ee" opacity={0.95} />
    </g>
  );
}

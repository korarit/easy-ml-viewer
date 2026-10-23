import type { TrajectoryPoint3D } from '../../store/useStage3DStore';
import { useStage3DStore } from '../../store/useStage3DStore';
import { sampleColormap } from '../../../projection3d/components/colormaps';
import type { ColormapName } from '../../../projection3d/types/projection.types';
import { wy } from '../DecisionBoundaryContour2D';
import { CometTrail2D } from './CometTrail2D';
import { DualRingPoint2D } from './DualRingPoint2D';
import { ResidualSpoke2D } from './ResidualSpoke2D';
import {
  GHOST_ALPHA,
  cometTrailAlphas,
  residualSpokeEnd,
  type GlyphShape,
} from './pointIntelligence2dMath';
import { trailSigns } from './linearPointIntel2dMath';

export interface PointIntelligenceLayer2DProps {
  points: TrajectoryPoint3D[];
  activeFrameIndex: number;
  isClassification: boolean;
  classes: Array<string | number>;
  colormap: ColormapName;
  heatmapMin: number;
  heatmapMax: number;
  trailWindow?: number;
  /** Phase 50 — raw-space plane weights for trail-ascent coloring (linear family). */
  trailWeights?: { w1: number; w2: number } | null;
}

function valueColor(
  raw: number,
  classes: Array<string | number>,
  isClassification: boolean,
  hMin: number,
  hMax: number,
  colormap: ColormapName,
): string {
  if (isClassification && classes.length > 0) {
    const idx = classes.findIndex((c) => c === raw || String(c) === String(raw));
    const t = classes.length < 2 ? 0.5 : Math.max(0, idx) / (classes.length - 1);
    const [r, g, b] = sampleColormap(colormap, Math.min(1, Math.max(0, t)));
    return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
  }
  const span = hMax - hMin;
  const t = span < 1e-12 ? 0.5 : (raw - hMin) / span;
  const [r, g, b] = sampleColormap(colormap, Math.min(1, Math.max(0, t)));
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

function glyphFor(actualRaw: number | null, classes: Array<string | number>): GlyphShape {
  if (classes.length === 0 || actualRaw === null) return 'circle';
  const idx = classes.findIndex((c) => c === actualRaw || String(c) === String(actualRaw));
  if (idx < 0) return 'circle';
  return (['circle', 'rect', 'polygon'] as const)[idx % 3];
}

/**
 * Phase 49 — root point-intelligence layer (mechanisms A + B + C).
 *
 * Owns all trajectory-point rendering in the 2D viewport: comet trails
 * behind the active frame, dual-ring glyphs (actual core + prediction
 * halo), error-lens filtering, and residual spokes. Flags live in the
 * stage3d store so the toolbar + `T/R/E` shortcuts stay in sync.
 */
export function PointIntelligenceLayer2D({
  points,
  activeFrameIndex,
  isClassification,
  classes,
  colormap,
  heatmapMin,
  heatmapMax,
  trailWindow = 8,
  trailWeights = null,
}: PointIntelligenceLayer2DProps) {
  const showCometTrails = useStage3DStore((s) => s.showCometTrails);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);

  if (points.length === 0) return null;
  const active = Math.max(0, Math.min(activeFrameIndex, points.length - 1));
  const alphas = showCometTrails ? cometTrailAlphas(points.length, active, trailWindow) : null;

  const isError = (f: number): boolean => {
    const r = points[f]?.residual_error;
    if (r === null || r === undefined || !Number.isFinite(r)) return true;
    return Math.abs(r) > 0.5;
  };

  const spokes: Array<[number, number, number, number]> = [];
  if (showResidualSpokes) {
    points.forEach((p, f) => {
      if (showErrorsOnly && !isError(f)) return;
      const r = p.residual_error;
      if (r === null || r === undefined || !Number.isFinite(r) || Math.abs(r) < 1e-12) return;
      const [ex, ey] = residualSpokeEnd(p.world_x, p.world_y, r);
      spokes.push([p.world_x, p.world_y, ex, ey]);
    });
  }

  return (
    <g aria-label="Point intelligence layer">
      {showCometTrails &&
        (() => {
          const rendered = points
            .map((p, f) => ({ p, f }))
            .filter(({ f }) => !showErrorsOnly || isError(f) || f === active);
          const signs = trailWeights
            ? trailSigns(
                rendered.map(({ p }) => ({ x: p.raw_x, y: p.raw_y })),
                trailWeights.w1,
                trailWeights.w2,
              )
            : undefined;
          const activeRendered = rendered.findIndex(({ f }) => f === active);
          return (
            <CometTrail2D
              points={rendered.map(({ p }) => ({ x: p.world_x, y: p.world_y }))}
              activeFrame={activeRendered >= 0 ? activeRendered : rendered.length - 1}
              window={trailWindow}
              signs={signs}
              visible
            />
          );
        })()}
      <ResidualSpoke2D segments={spokes} visible={showResidualSpokes} />
      <g aria-label="Trajectory points">
        {points.map((p, f) => {
          const err = isClassification && isError(f);
          if (showErrorsOnly && !err && f !== active) return null;
          const baseOpacity = alphas ? alphas[f]! : f === active ? 1 : 0.88;
          const opacity = showErrorsOnly && !err ? 1 : baseOpacity;
          const actualRaw = p.raw_z_actual ?? p.raw_z_pred;
          const actualColor = valueColor(actualRaw, classes, isClassification, heatmapMin, heatmapMax, colormap);
          const predColor = valueColor(p.raw_z_pred, classes, isClassification, heatmapMin, heatmapMax, colormap);
          if (!showDualRings) {
            const r = f === active ? 0.14 : 0.1;
            return (
              <g key={`${p.original_row_index}-${f}`}>
                <circle
                  cx={p.world_x}
                  cy={wy(p.world_y)}
                  r={r}
                  fill={actualColor}
                  stroke={err ? '#ef4444' : '#0b0f14'}
                  strokeWidth={err ? 0.05 : 0.03}
                  opacity={opacity}
                />
                {err && (
                  <text x={p.world_x} y={wy(p.world_y)} fontSize={0.22} fill="#ef4444" textAnchor="middle" dy="0.08">
                    ×
                  </text>
                )}
              </g>
            );
          }
          return (
            <g key={`${p.original_row_index}-${f}`} opacity={f === active ? 1 : opacity}>
              <DualRingPoint2D
                x={p.world_x}
                y={p.world_y}
                actualColor={actualColor}
                predColor={predColor}
                glyph={glyphFor(p.raw_z_actual, classes)}
                mispredicted={err}
                isActive={f === active}
                opacity={f === active ? 1 : Math.max(opacity, GHOST_ALPHA)}
              />
            </g>
          );
        })}
      </g>
    </g>
  );
}

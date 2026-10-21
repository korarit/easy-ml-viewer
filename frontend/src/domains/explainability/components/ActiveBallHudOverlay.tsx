import type { ReactNode } from 'react';
import { Html } from '@react-three/drei';
import { useExplainabilityStore } from '../store/useExplainabilityStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { formatTickLabel, worldToThree } from '../../stage3d/components/stage3dSpace';

/** Thin wrapper keeping the single `<Html>` call-site in one place. */
function HudHtml({ position, children }: { position: [number, number, number]; children: ReactNode }) {
  return (
    <Html position={position} center distanceFactor={18} zIndexRange={[20, 0]} occlude={false}>
      <div style={{ pointerEvents: 'none' }}>{children}</div>
    </Html>
  );
}

/**
 * Phase 12 — floating 3D telemetry HUD.
 *
 * Single-`<Html>`-instance rule (§5.2): this is the ONE ball-following
 * `<Html>` in the scene (it replaces the legacy per-actor labels).
 * `pointer-events-none` + `[0, 0.65, 0]` offset so the pill never blocks
 * raycast clicks on the point cloud (edge #3). The marker div stays
 * mounted (with `data-hud-hidden`) so `H`!/tests can observe HUD state.
 */
export function ActiveBallHudOverlay() {
  const points = useStage3DStore((s) => s.points);
  const frameIndex = usePlaybackStore((s) => s.currentFrameIndex);
  const explanation = useExplainabilityStore((s) => s.explanation);
  const hudVisible = useExplainabilityStore((s) => s.hudVisible);
  const hudCompact = useExplainabilityStore((s) => s.hudCompact);

  if (points.length === 0) return null;
  const point = points[Math.max(0, Math.min(frameIndex, points.length - 1))];
  if (!point) return null;

  const [hx, hy, hz] = worldToThree(point.world_x, point.world_y, point.world_z_pred);
  const pos: [number, number, number] = [hx, hy + 0.65, hz];

  const topDriver =
    explanation && explanation.attributions.length > 0
      ? [...explanation.attributions].sort(
          (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution),
        )[0]
      : null;

  const matchBadge =
    point.raw_z_actual !== null && point.residual_error !== null ? (
      <span
        style={{
          color: Math.abs(point.residual_error) <= 1e-9 ? '#34d399' : '#f43f5e',
          marginLeft: 6,
        }}
      >
        {Math.abs(point.residual_error) <= 1e-9 ? '[✓ MATCH]' : '[✗ MISCLASSIFIED]'}
      </span>
    ) : null;

  return (
    <HudHtml position={pos}>
      <div data-testid="active-ball-hud" data-hud-hidden={hudVisible ? 'false' : 'true'}>
        {hudVisible && !hudCompact && (
          <div
            className="font-mono-tabular rounded px-2 py-1 text-[11px] leading-4"
            style={{
              background: 'rgba(10,15,22,0.85)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
            }}
          >
            <span style={{ color: '#fb923c' }}>●</span> FRAME #{point.frame_index} (Row #
            {point.original_row_index})
            {matchBadge}
            <br />
            Pred ŷ = {formatTickLabel(point.raw_z_pred)}
            {point.raw_z_actual !== null && <> · Actual = {formatTickLabel(point.raw_z_actual)}</>}
            <br />
            {topDriver && (
              <span style={{ color: 'var(--text-secondary)' }}>
                Top: {topDriver.contribution >= 0 ? '▲' : '▼'} {topDriver.feature_name} (
                {topDriver.contribution >= 0 ? '+' : ''}
                {topDriver.contribution.toFixed(3)})
              </span>
            )}
          </div>
        )}
        {hudVisible && hudCompact && (
          <div
            className="font-mono-tabular rounded px-2 py-1 text-[11px]"
            style={{
              background: 'rgba(10,15,22,0.85)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
            }}
          >
            {formatTickLabel(point.raw_z_pred)}
          </div>
        )}
      </div>
    </HudHtml>
  );
}

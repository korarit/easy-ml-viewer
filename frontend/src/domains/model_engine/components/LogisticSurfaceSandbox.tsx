import { useEffect, useMemo } from 'react';
import { useLogisticSliceStore } from '../store/useLogisticSliceStore';
import { SigmoidSurface3D } from '../../stage3d/components/surfaces/SigmoidSurface3D';

/** Phase 18 — axis selectors, threshold slider, z-mode toggle and mesh telemetry. */
export function LogisticSurfaceSandbox() {
  const {
    featureNames,
    axisX,
    axisY,
    surface,
    threshold,
    zMode,
    isSurfacing,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setThreshold,
    setZMode,
  } = useLogisticSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useLogisticSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useLogisticSliceStore((s) => s.rowInput);

  const liveFeatures = useMemo(() => {
    const vals = rowInput
      .split(/[\s,;]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map(Number);
    if (vals.length !== featureNames.length || vals.some((v) => !Number.isFinite(v))) return null;
    return vals;
  }, [rowInput, featureNames.length]);

  if (featureNames.length === 0) return null;
  const telemetry = surface?.telemetry;
  const zeroAxes = surface?.extra_overlays?.zero_influence_axes === true;

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D sigmoid S-surface
        </span>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Axis X
          <select
            aria-label="Axis X"
            value={axisX}
            onChange={(e) => setAxisX(Number(e.target.value))}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {featureNames.map((n, i) => (
              <option key={n} value={i}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Axis Y
          <select
            aria-label="Axis Y"
            value={axisY}
            onChange={(e) => setAxisY(Number(e.target.value))}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {featureNames.map((n, i) => (
              <option key={n} value={i}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
          Threshold τ
          <input
            aria-label="Surface threshold"
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <span className="font-mono-tabular">{threshold.toFixed(2)}</span>
        </label>
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
          60×60 grid · mode: {zMode === 'probability' ? 'σ(z) S-curve' : 'linear z plane'}
        </span>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      <div className="mt-2 flex flex-wrap gap-2">
        <button
          onClick={() => setZMode('probability')}
          className="rounded px-2 py-1 text-[11px]"
          style={{
            border: '1px solid var(--border-strong)',
            background: zMode === 'probability' ? 'var(--ml-primary)' : 'transparent',
            color: zMode === 'probability' ? '#fff' : 'var(--text-primary)',
          }}
        >
          Probability S-Curve
        </button>
        <button
          onClick={() => setZMode('logodds')}
          className="rounded px-2 py-1 text-[11px]"
          style={{
            border: '1px solid var(--border-strong)',
            background: zMode === 'logodds' ? 'var(--ml-primary)' : 'transparent',
            color: zMode === 'logodds' ? '#fff' : 'var(--text-primary)',
          }}
        >
          Linear Log-Odds (z)
        </button>
      </div>

      {zeroAxes && (
        <p
          className="mt-2 rounded px-2 py-1.5"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          Both selected axes carry zero weight (e.g. L1 pruned them) — the S-surface is flat and
          the P = τ boundary tube is hidden.
        </p>
      )}

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} float32 vertices ready | P Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      {surface && <SigmoidSurface3D liveFeatures={liveFeatures} />}

      {showTelemetry && surface && (
        <pre
          className="font-mono-tabular mt-2 max-h-48 overflow-auto rounded p-2 text-[11px]"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
        >
          {JSON.stringify(
            {
              visual_type: surface.visual_type,
              resolution: surface.resolution,
              threshold,
              z_mode: zMode,
              x_range: [surface.x_grid[0], surface.x_grid[surface.x_grid.length - 1]],
              y_range: [surface.y_grid[0], surface.y_grid[surface.y_grid.length - 1]],
              extra_overlays: surface.extra_overlays,
              telemetry: surface.telemetry,
              z_sample: surface.z_values.slice(0, 8),
            },
            null,
            2,
          )}
        </pre>
      )}
      <p className="font-mono-tabular mt-1" style={{ color: 'var(--text-muted)' }}>
        Tip: Ctrl+Shift+S toggles the JSON telemetry drawer.
      </p>
    </div>
  );
}

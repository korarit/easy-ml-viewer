import { useEffect, useMemo } from 'react';
import { useElasticNetSliceStore } from '../store/useElasticNetSliceStore';
import { ElasticNetPlane3D } from '../../stage3d/components/surfaces/ElasticNetPlane3D';

/** Phase 20 — axis selectors, task switcher, ring note and mesh telemetry. */
export function ElasticNetSurfaceSandbox() {
  const {
    featureNames,
    axisX,
    axisY,
    taskIdx,
    taskCount,
    surface,
    isSurfacing,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setTaskIdx,
  } = useElasticNetSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useElasticNetSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useElasticNetSliceStore((s) => s.rowInput);

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
  const overlays = surface?.extra_overlays;
  const flat = overlays?.zero_influence_axes === true;
  const rho = typeof overlays?.l1_ratio === 'number' ? overlays.l1_ratio : null;

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D elastic-net plane + constraint ring
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
        {taskCount > 1 && (
          <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            Task
            <select
              aria-label="Task output"
              value={taskIdx}
              onChange={(e) => setTaskIdx(Number(e.target.value))}
              className="rounded px-1.5 py-0.5"
              style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
            >
              {Array.from({ length: taskCount }, (_, i) => (
                <option key={i} value={i}>
                  task {i}
                </option>
              ))}
            </select>
          </label>
        )}
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
          60×60 grid · tilted plane + L1/L2 aura
        </span>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      {rho !== null && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Constraint aura:{' '}
          {Math.abs(rho) < 1e-9
            ? '○ Ridge circle — L2 ball, no sparsity'
            : Math.abs(1 - rho) < 1e-9
              ? '◇ Lasso diamond — sharp corners prune weights to 0'
              : `▢ Rounded diamond (ρ=${rho.toFixed(2)}) — corners prune, curvature groups correlated features`}
        </p>
      )}

      {flat && (
        <p
          className="mt-2 rounded px-2 py-1.5"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          Both selected axes carry zero weight — the plane is flat and What-If sliders on pruned
          features cannot change the prediction.
        </p>
      )}

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} float32 vertices ready | Z Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      {surface && <ElasticNetPlane3D liveFeatures={liveFeatures} />}

      {showTelemetry && surface && (
        <pre
          className="font-mono-tabular mt-2 max-h-48 overflow-auto rounded p-2 text-[11px]"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
        >
          {JSON.stringify(
            {
              visual_type: surface.visual_type,
              resolution: surface.resolution,
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

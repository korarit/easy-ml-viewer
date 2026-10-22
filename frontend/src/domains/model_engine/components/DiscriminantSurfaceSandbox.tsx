import { useEffect, useMemo } from 'react';
import { useDiscriminantSliceStore } from '../store/useDiscriminantSliceStore';
import { CovarianceEllipsoids3D } from '../../stage3d/components/surfaces/CovarianceEllipsoids3D';
import type { SigmaLevel } from '../../stage3d/components/surfaces/discriminantSurfaceMath';

/** Phase 26 — axis selectors, class switcher, σ-shell level, axes/tether toggles and mesh telemetry. */
export function DiscriminantSurfaceSandbox() {
  const {
    featureNames,
    axisX,
    axisY,
    classIdx,
    classCount,
    surface,
    isSurfacing,
    sigmaLevel,
    showAxes,
    showTethers,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setClassIdx,
    setSigmaLevel,
    setShowAxes,
    setShowTethers,
  } = useDiscriminantSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useDiscriminantSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useDiscriminantSliceStore((s) => s.rowInput);

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
  const ells = overlays?.ellipsoids_2d ?? [];

  const levels: Array<{ key: SigmaLevel; label: string }> = [
    { key: 1, label: '1σ (68%)' },
    { key: 2, label: '2σ (95%)' },
    { key: 3, label: '3σ (99.7%)' },
  ];

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          Ellipsoid shells
        </span>
        {levels.map((m) => (
          <button
            key={m.key}
            onClick={() => setSigmaLevel(m.key)}
            className="font-mono-tabular rounded px-2 py-1 text-[11px]"
            style={{
              border: '1px solid var(--border-strong)',
              background: sigmaLevel === m.key ? 'var(--ml-primary)' : 'var(--surface-elevated)',
              color: sigmaLevel === m.key ? '#fff' : 'var(--text-primary)',
            }}
          >
            {m.label}
          </button>
        ))}
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
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Target
          <select
            aria-label="Target class"
            value={classIdx}
            onChange={(e) => setClassIdx(Number(e.target.value))}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {Array.from({ length: classCount }, (_, i) => (
              <option key={i} value={i}>
                class {i}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showAxes}
            onChange={(e) => setShowAxes(e.target.checked)}
          />
          Eigen-Axes
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showTethers}
            onChange={(e) => setShowTethers(e.target.checked)}
          />
          Tethers
        </label>
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
          60×60 grid · tilted ellipsoids
        </span>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      {ells.length > 0 && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          {ells.map((e) => `class ${e.class_idx}: θ=${(e.rotation_angle_rad * 180 / Math.PI).toFixed(1)}°`).join(' · ')}
        </p>
      )}

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} float32 vertices ready | Z Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      {surface && <CovarianceEllipsoids3D liveFeatures={liveFeatures} />}

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

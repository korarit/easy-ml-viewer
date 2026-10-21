import { useEffect, useMemo } from 'react';
import { useSvmSliceStore } from '../store/useSvmSliceStore';
import { KernelSaddleSurface3D } from '../../stage3d/components/surfaces/KernelSaddleSurface3D';

/** Phase 23 — axis selectors, class switcher, pillar/tether/contour toggles and mesh telemetry. */
export function SvmKernelSurfaceSandbox() {
  const {
    featureNames,
    axisX,
    axisY,
    classIdx,
    classCount,
    isClassifier,
    isOneClass,
    surface,
    isSurfacing,
    showPillars,
    showTethers,
    showContours,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setClassIdx,
    setShowPillars,
    setShowTethers,
    setShowContours,
  } = useSvmSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useSvmSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useSvmSliceStore((s) => s.rowInput);

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
  const kernel = typeof overlays?.kernel === 'string' ? overlays.kernel : null;
  const unscaled = telemetry !== undefined && telemetry !== null && Math.max(Math.abs(telemetry.z_min), Math.abs(telemetry.z_max)) > 1e4;

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D kernel saddle + SV pillars
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
        {isClassifier && !isOneClass && classCount > 2 && (
          <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            Class
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
        )}
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showPillars}
            onChange={(e) => setShowPillars(e.target.checked)}
          />
          SV Pillars
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showTethers}
            onChange={(e) => setShowTethers(e.target.checked)}
          />
          Top-5 Tethers
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showContours}
            onChange={(e) => setShowContours(e.target.checked)}
          />
          Margin Contours
        </label>
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
          60×60 grid · top-1000 SVs
        </span>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      {kernel && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Kernel: {kernel}
          {typeof overlays?.surface_value === 'string' ? ` · Surface: ${overlays.surface_value}` : ''}
        </p>
      )}

      {unscaled && (
        <p
          className="mt-2 rounded px-2 py-1.5 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="note"
        >
          Unscaled Features Detected: Large Gradient Magnitude (|z| &gt; 1e4) — train on standardized
          features (or a Scaler → SVM pipeline) for a readable saddle.
        </p>
      )}

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} float32 vertices ready | Z Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      {surface && <KernelSaddleSurface3D liveFeatures={liveFeatures} />}

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

import { useEffect, useMemo } from 'react';
import { useAdaBoostSliceStore } from '../store/useAdaBoostSliceStore';
import { AdaBoostCascadeLadder3D } from '../../stage3d/components/surfaces/AdaBoostCascadeLadder3D';

/** Phase 30 — axis selectors, class switcher, stage scrubber and mesh telemetry. */
export function AdaBoostSurfaceSandbox() {
  const {
    featureNames,
    axisX,
    axisY,
    classIdx,
    classCount,
    surface,
    isSurfacing,
    isRegressor,
    selectedStage,
    showLadder,
    showCurtains,
    showWireframe,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setClassIdx,
    setSelectedStage,
    setShowLadder,
    setShowCurtains,
    setShowWireframe,
    traces,
  } = useAdaBoostSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useAdaBoostSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useAdaBoostSliceStore((s) => s.rowInput);
  const total = traces[0]?.extra_mechanics.n_estimators_actual ?? surface?.extra_overlays?.n_estimators_actual ?? 0;

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

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D cascade
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
        {!isRegressor && (
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
        )}
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Stage ≤ {Math.min(selectedStage + 1, Math.max(total, 1))}/{Math.max(total, 1)}
          <input
            aria-label="Stage scrubber"
            type="range"
            min={0}
            max={Math.max(0, total - 1)}
            value={Math.min(selectedStage, Math.max(0, total - 1))}
            onChange={(e) => setSelectedStage(Number(e.target.value))}
            className="w-28"
          />
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showLadder} onChange={(e) => setShowLadder(e.target.checked)} />
          Ladder
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showCurtains} onChange={(e) => setShowCurtains(e.target.checked)} />
          Curtains
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showWireframe}
            onChange={(e) => setShowWireframe(e.target.checked)}
          />
          Wireframe
        </label>
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
          60×60 plaid voxels · rungs dim past scrubber
          {overlays?.n_estimators_actual ? ` · ${overlays.n_estimators_actual} learners` : ''}
        </span>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} voxels ready | Z Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      {surface && <AdaBoostCascadeLadder3D liveFeatures={liveFeatures} />}

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
              n_estimators_actual: surface.extra_overlays?.n_estimators_actual,
              stump_splits: surface.extra_overlays?.stump_splits?.slice(0, 8),
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

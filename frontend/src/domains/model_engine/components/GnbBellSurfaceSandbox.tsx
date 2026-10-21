import { useEffect, useMemo } from 'react';
import { useGnbSliceStore, type GnbSurfaceMode } from '../store/useGnbSliceStore';
import { GaussianBellHills3D } from '../../stage3d/components/surfaces/GaussianBellHills3D';

/** Phase 24 — axis selectors, class switcher, hill visibility, mode switch and mesh telemetry. */
export function GnbBellSurfaceSandbox() {
  const {
    featureNames,
    axisX,
    axisY,
    classIdx,
    classCount,
    surface,
    isSurfacing,
    surfaceMode,
    visibleHills,
    showPins,
    showRings,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setClassIdx,
    setSurfaceMode,
    toggleHill,
    setShowPins,
    setShowRings,
  } = useGnbSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useGnbSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useGnbSliceStore((s) => s.rowInput);

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
  const hillOrder = overlays?.hill_class_order ?? [];

  const modes: Array<{ key: GnbSurfaceMode; label: string }> = [
    { key: 'generative', label: 'Generative Bell Hills' },
    { key: 'posterior', label: 'Posterior P(C|x)' },
  ];

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D bell hills
        </span>
        {modes.map((m) => (
          <button
            key={m.key}
            onClick={() => setSurfaceMode(m.key)}
            className="font-mono-tabular rounded px-2 py-1 text-[11px]"
            style={{
              border: '1px solid var(--border-strong)',
              background: surfaceMode === m.key ? 'var(--ml-primary)' : 'var(--surface-elevated)',
              color: surfaceMode === m.key ? '#fff' : 'var(--text-primary)',
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
            checked={showPins}
            onChange={(e) => setShowPins(e.target.checked)}
          />
          Drop-Pins
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showRings}
            onChange={(e) => setShowRings(e.target.checked)}
          />
          1σ/2σ Rings
        </label>
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
          60×60 grid · analytic hills
        </span>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      {hillOrder.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
            Hills:
          </span>
          {hillOrder.map((k) => (
            <button
              key={k}
              onClick={() => toggleHill(k)}
              className="font-mono-tabular rounded px-2 py-0.5 text-[11px]"
              style={{
                border: '1px solid var(--border-strong)',
                background: visibleHills.includes(k) ? 'var(--ml-primary)' : 'var(--surface-elevated)',
                color: visibleHills.includes(k) ? '#fff' : 'var(--text-primary)',
              }}
            >
              class {k}
            </button>
          ))}
          {overlays?.hills_truncated_to_top5 === true && (
            <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
              (top-5 priors of {classCount})
            </span>
          )}
        </div>
      )}

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} float32 vertices ready | Z Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      {surface && <GaussianBellHills3D liveFeatures={liveFeatures} />}

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
              extra_overlays: {
                ...surface.extra_overlays,
                bell_hills_per_class: surface.extra_overlays?.bell_hills_per_class
                  ? Object.fromEntries(
                      Object.entries(surface.extra_overlays.bell_hills_per_class).map(([k, v]) => [
                        k,
                        Array.isArray(v) ? (v as number[]).slice(0, 8) : v,
                      ]),
                    )
                  : undefined,
              },
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

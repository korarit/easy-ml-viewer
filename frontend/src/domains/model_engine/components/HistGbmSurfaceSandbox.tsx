import { useEffect, useMemo } from 'react';
import { useHistGbmSliceStore, parseHistGbmRow } from '../store/useHistGbmSliceStore';
import { HistGbmBinTerraces3D } from '../../stage3d/components/surfaces/HistGbmBinTerraces3D';
import { findHistBin, histBinRange } from '../../stage3d/components/surfaces/histGbmSurfaceMath';

/** Phase 32 — axis selectors, class switcher, stage scrubber, NaN injector and mesh telemetry. */
export function HistGbmSurfaceSandbox() {
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
    showSpiral,
    showBinGrid,
    showWireframe,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setClassIdx,
    setSelectedStage,
    setShowSpiral,
    setShowBinGrid,
    setShowWireframe,
    injectNan,
    traces,
  } = useHistGbmSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useHistGbmSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useHistGbmSliceStore((s) => s.rowInput);
  const total =
    traces[0]?.extra_mechanics.n_iter_total ??
    surface?.extra_overlays?.n_iter_total ??
    surface?.extra_overlays?.n_estimators_actual ??
    0;

  const liveFeatures = useMemo(() => {
    const parsed = parseHistGbmRow(rowInput);
    if (!parsed || parsed.length !== featureNames.length || parsed.some((v) => v === null)) return null;
    return (parsed as number[]).map((v) => (Number.isFinite(v) ? v : NaN));
  }, [rowInput, featureNames.length]);

  const binBadge = useMemo(() => {
    const thresholds = surface?.extra_overlays?.axis_bin_thresholds as
      | { x: number[]; y: number[] }
      | undefined;
    if (!thresholds || !liveFeatures) return null;
    const bx = findHistBin(thresholds.x, liveFeatures[axisX]);
    const by = findHistBin(thresholds.y, liveFeatures[axisY]);
    const [xLo, xHi] = histBinRange(thresholds.x, bx);
    const [yLo, yHi] = histBinRange(thresholds.y, by);
    return { bx, by, xLo, xHi, yLo, yHi, nx: thresholds.x.length, ny: thresholds.y.length };
  }, [surface, liveFeatures, axisX, axisY]);

  if (featureNames.length === 0) return null;
  const telemetry = surface?.telemetry;
  const overlays = surface?.extra_overlays;
  const mech = traces[0]?.extra_mechanics ?? null;

  const fmtBound = (v: number | null) => (v == null ? '−∞/​+∞' : v.toFixed(2));

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D bin terraces
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
          Iter ≤ {Math.min(selectedStage + 1, Math.max(total, 1))}/{Math.max(total, 1)}
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
        <button
          onClick={() => injectNan(axisX)}
          className="rounded-md px-2 py-0.5 text-[11px] font-medium"
          style={{ border: '1px solid var(--ml-class-1)', color: 'var(--ml-class-1)' }}
          title="Set Axis X feature to NaN (native missing routing)"
        >
          Inject NaN → X
        </button>
        <button
          onClick={() => injectNan(axisY)}
          className="rounded-md px-2 py-0.5 text-[11px] font-medium"
          style={{ border: '1px solid var(--ml-class-1)', color: 'var(--ml-class-1)' }}
          title="Set Axis Y feature to NaN (native missing routing)"
        >
          Inject NaN → Y
        </button>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showBinGrid} onChange={(e) => setShowBinGrid(e.target.checked)} />
          Bin grid
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showSpiral} onChange={(e) => setShowSpiral(e.target.checked)} />
          Spiral
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
          {overlays?.n_iter_total ? `${overlays.n_iter_total} iters` : ''} · rungs dim past scrubber
        </span>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing surface…</span>}
      </div>

      {binBadge && liveFeatures && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          X ({featureNames[axisX]}={Number.isFinite(liveFeatures[axisX]) ? liveFeatures[axisX].toFixed(2) : 'NaN!'})
          {' ➔ '}Bin #{binBadge.bx >= 0 ? binBadge.bx : 'NaN'}/{binBadge.nx}
          {' '}({fmtBound(binBadge.xLo)} .. {fmtBound(binBadge.xHi)}) · Y ({featureNames[axisY]}=
          {Number.isFinite(liveFeatures[axisY]) ? liveFeatures[axisY].toFixed(2) : 'NaN!'})
          {' ➔ '}Bin #{binBadge.by >= 0 ? binBadge.by : 'NaN'}/{binBadge.ny} ({fmtBound(binBadge.yLo)} ..{' '}
          {fmtBound(binBadge.yHi)})
          {mech?.has_missing_values ? ' · ◆ NaN routed via missing_go_to_left' : ''}
        </p>
      )}

      {telemetry && (
        <p className="font-mono-tabular mt-1" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} voxels ready | Z Range: [
          {telemetry.z_min.toFixed(3)} .. {telemetry.z_max.toFixed(3)}] | Compute Time:{' '}
          {telemetry.compute_ms.toFixed(1)} ms
        </p>
      )}

      {surface && <HistGbmBinTerraces3D liveFeatures={liveFeatures} />}

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
              n_iter_total: surface.extra_overlays?.n_iter_total,
              learning_rate: surface.extra_overlays?.learning_rate,
              monotonic_cst: surface.extra_overlays?.monotonic_cst,
              categorical_feature_idx: surface.extra_overlays?.categorical_feature_idx,
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

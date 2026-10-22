import { useEffect, useMemo } from 'react';
import { useCatBoostSliceStore, type CatboostImportanceMetric } from '../store/useCatBoostSliceStore';
import { CatBoostObliviousGrid3D } from '../../stage3d/components/surfaces/CatBoostObliviousGrid3D';
import { catboostBitmask } from '../../stage3d/components/surfaces/catboostSurfaceMath';

/** Phase 35 — axis selectors, class switcher, iteration scrubber, tree explorer and telemetry. */
export function CatBoostSurfaceSandbox() {
  const {
    featureNames,
    categoricalIdx,
    axisX,
    axisY,
    classIdx,
    classCount,
    surface,
    isSurfacing,
    isRegressor,
    selectedRound,
    selectedTree,
    importanceMetric,
    showForceVectors,
    showSplitLines,
    showCategoricalTicks,
    showWireframe,
    showTelemetry,
    setShowTelemetry,
    setAxisX,
    setAxisY,
    setClassIdx,
    setSelectedRound,
    setSelectedTree,
    setImportanceMetric,
    setShowForceVectors,
    setShowSplitLines,
    setShowCategoricalTicks,
    setShowWireframe,
    traces,
  } = useCatBoostSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useCatBoostSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useCatBoostSliceStore((s) => s.rowInput);
  const total =
    traces[0]?.extra_mechanics.n_iterations ??
    surface?.extra_overlays?.n_iterations ??
    surface?.extra_overlays?.n_boost_rounds ??
    0;

  const liveFeatures = useMemo(() => {
    const vals = rowInput
      .split(/[\s,;]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => {
        const n = Number(s);
        return s !== '' && Number.isFinite(n) ? n : s;
      });
    if (vals.length !== featureNames.length) return null;
    return vals;
  }, [rowInput, featureNames.length]);

  const topology = surface?.extra_overlays?.oblivious_tree_topology ?? [];
  const activeLeaves = (traces[0]?.extra_mechanics.leaf_indices_sample ?? []) as number[];
  const badges = (traces[0]?.extra_mechanics.leaf_bitmask_badges ?? []) as Array<{
    tree: number;
    leaf: number;
    bitmask: string;
  }>;
  const treeCount = topology.length;

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
          3D symmetric oblivious grid
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
                {n}{categoricalIdx.includes(i) ? ' [CAT]' : ''}
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
                {n}{categoricalIdx.includes(i) ? ' [CAT]' : ''}
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
          Iter ≤ {Math.min(selectedRound + 1, Math.max(total, 1))}/{Math.max(total, 1)}
          <input
            aria-label="Iteration scrubber"
            type="range"
            min={0}
            max={Math.max(0, total - 1)}
            value={Math.min(selectedRound, Math.max(0, total - 1))}
            onChange={(e) => setSelectedRound(Number(e.target.value))}
            className="w-28"
          />
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Metric
          <select
            aria-label="Importance metric"
            value={importanceMetric}
            onChange={(e) => setImportanceMetric(e.target.value as CatboostImportanceMetric)}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {(['prediction_values_change', 'gain', 'loss_function_change', 'split'] as const).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showForceVectors} onChange={(e) => setShowForceVectors(e.target.checked)} />
          SHAP vectors
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showSplitLines} onChange={(e) => setShowSplitLines(e.target.checked)} />
          Split planes
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={showCategoricalTicks}
            onChange={(e) => setShowCategoricalTicks(e.target.checked)}
          />
          CAT ticks
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
          {overlays?.n_iterations ? `${overlays.n_iterations} trees` : ''}
          {overlays?.loss_function ? ` · ${overlays.loss_function}` : ''}
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

      {surface && <CatBoostObliviousGrid3D liveFeatures={liveFeatures} />}

      {badges.length > 0 && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Bitwise leaves:{' '}
          {badges.map((b) => `#${b.tree}: leaf ${b.leaf} [${b.bitmask}₂]`).join(' · ')}
        </p>
      )}

      {surface && treeCount > 0 && (
        <div
          className="mt-2 rounded-lg px-4 py-3"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
        >
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
              Oblivious tree explorer
            </span>
            <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
              Tree #
              <select
                aria-label="Oblivious tree selector"
                value={Math.min(selectedTree, treeCount - 1)}
                onChange={(e) => setSelectedTree(Number(e.target.value))}
                className="rounded px-1.5 py-0.5"
                style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-panel)' }}
              >
                {topology.map((t) => (
                  <option key={t.round} value={topology.indexOf(t)}>
                    #{t.round} (depth {t.depth} · {t.num_leaves} leaves)
                  </option>
                ))}
              </select>
            </label>
            <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
              first {treeCount} of {total} trees · active leaf #{activeLeaves[selectedTree] ?? '—'}
              {topology[selectedTree]
                ? ` [${catboostBitmask(activeLeaves[selectedTree] ?? 0, topology[selectedTree].depth)}₂]`
                : ''}
            </span>
          </div>
          {topology[Math.min(selectedTree, treeCount - 1)] ? (
            <ol className="font-mono-tabular flex flex-col gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {topology[Math.min(selectedTree, treeCount - 1)].levels.map((lv) => (
                <li key={lv.depth} className="flex flex-wrap gap-x-3">
                  <span>
                    Depth {lv.depth}: {lv.feature_name} [{lv.feature_idx}]{' '}
                    {lv.kind === 'categorical' ? `value = ${lv.category ?? '—'}` : `≤ ${lv.threshold}`} · gain{' '}
                    {lv.gain}
                  </span>
                </li>
              ))}
            </ol>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>
              Run a trace to highlight the symmetric split levels of this tree.
            </p>
          )}
        </div>
      )}

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
              n_iterations: surface.extra_overlays?.n_iterations,
              loss_function: surface.extra_overlays?.loss_function,
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

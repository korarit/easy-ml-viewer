import { useEffect, useMemo } from 'react';
import { useLightGbmSliceStore, type LgbmImportanceMetric } from '../store/useLightGbmSliceStore';
import { LightgbmLeafWise3D } from '../../stage3d/components/surfaces/LightgbmLeafWise3D';
import { lgbmLeafPath, type LgbmFlatNode } from '../../stage3d/components/surfaces/lightgbmSurfaceMath';

/** Phase 34 — axis selectors, class switcher, iteration scrubber, tree explorer and telemetry. */
export function LightgbmSurfaceSandbox() {
  const {
    featureNames,
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
    setShowWireframe,
    traces,
  } = useLightGbmSliceStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        setShowTelemetry(!useLightGbmSliceStore.getState().showTelemetry);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setShowTelemetry]);

  const rowInput = useLightGbmSliceStore((s) => s.rowInput);
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
      .map(Number);
    if (vals.length !== featureNames.length || vals.some((v) => !Number.isFinite(v))) return null;
    return vals;
  }, [rowInput, featureNames.length]);

  const topology = (surface?.extra_overlays?.tree_topology_sample ?? []) as Array<{
    round: number;
    nodes: LgbmFlatNode[];
    leaf_depths?: Record<number, number>;
  }>;
  const activeLeaves = (traces[0]?.extra_mechanics.leaf_indices_sample ?? []) as number[];
  const treeCount = topology.length;

  const activePath = useMemo(() => {
    const tree = topology[selectedTree];
    if (!tree) return [];
    const leafIdx = activeLeaves[selectedTree];
    if (leafIdx == null) return [];
    const leafNode = tree.nodes.find((n) => n.leaf_index === leafIdx);
    if (!leafNode) return [];
    const path = lgbmLeafPath(leafNode.id, tree.nodes);
    const byId = new Map(tree.nodes.map((n) => [n.id, n]));
    return path.map((id) => byId.get(id)).filter(Boolean) as LgbmFlatNode[];
  }, [topology, selectedTree, activeLeaves]);

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
          3D leaf-wise terrain
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
            onChange={(e) => setImportanceMetric(e.target.value as LgbmImportanceMetric)}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {(['gain', 'split'] as const).map((m) => (
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
          Split lines
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
          {overlays?.n_iterations ? `${overlays.n_iterations} iters` : ''}
          {overlays?.objective ? ` · ${overlays.objective}` : ''}
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

      {surface && <LightgbmLeafWise3D liveFeatures={liveFeatures} />}

      {surface && treeCount > 0 && (
        <div
          className="mt-2 rounded-lg px-4 py-3"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
        >
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
              Leaf-wise tree explorer
            </span>
            <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
              Tree #
              <select
                aria-label="Leaf-wise tree selector"
                value={Math.min(selectedTree, treeCount - 1)}
                onChange={(e) => setSelectedTree(Number(e.target.value))}
                className="rounded px-1.5 py-0.5"
                style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-panel)' }}
              >
                {topology.map((t) => (
                  <option key={t.round} value={topology.indexOf(t)}>
                    #{t.round} ({t.nodes.length} nodes)
                  </option>
                ))}
              </select>
            </label>
            <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>
              first {treeCount} of {total} iters · active leaf #{activeLeaves[selectedTree] ?? '—'}
            </span>
          </div>
          {activePath.length > 0 ? (
            <ol className="font-mono-tabular flex flex-col gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {[...activePath].reverse().map((n) => (
                <li key={n.id} className="flex flex-wrap gap-x-3">
                  {n.leaf_index !== undefined ? (
                    <span style={{ color: 'var(--ml-primary)' }}>
                      ● Leaf {n.leaf_index}: value {n.leaf_value != null && n.leaf_value >= 0 ? '+' : ''}{n.leaf_value?.toFixed(4)} · count{' '}
                      {n.leaf_count} · depth {n.depth}
                    </span>
                  ) : (
                    <span>
                      ○ Node {n.id}: {featureNames[n.split_feature_idx ?? -1] ?? `f${n.split_feature_idx}`} [
                      {n.split_feature_idx}]{' '}
                      {n.categories && n.categories.length > 0
                        ? `∈ {${n.categories.join(', ')}}`
                        : `≤ ${n.threshold}`}
                      {' → '}yes {n.yes} / no {n.no} · gain {n.gain} · depth {n.depth}
                    </span>
                  )}
                </li>
              ))}
            </ol>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>
              Run a trace to highlight the root → leaf path of this row.
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
              objective: surface.extra_overlays?.objective,
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

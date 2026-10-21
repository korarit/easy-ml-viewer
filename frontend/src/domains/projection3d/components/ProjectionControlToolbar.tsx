import { useEffect } from 'react';
import { useProjectionStore } from '../store/useProjectionStore';
import type { AvailableFeature, LodLevel } from '../types/projection.types';
import { LOD_RESOLUTIONS } from '../types/projection.types';
import { COLORMAP_LABELS, type ColormapName } from './colormaps';

/**
 * Phase 09 — projection control bar: mode toggle, axis selectors + swap,
 * Z-mode, pinning, LOD badges, appearance popover. Shortcuts: W wireframe,
 * P mode toggle. (G keeps its Phase-08 meaning: drop-lines & grid.)
 */
export function ProjectionControlToolbar({ availableFeatures }: { availableFeatures: AvailableFeature[] }) {
  const mode = useProjectionStore((s) => s.mode);
  const xFeature = useProjectionStore((s) => s.xFeature);
  const yFeature = useProjectionStore((s) => s.yFeature);
  const zMode = useProjectionStore((s) => s.zMode);
  const pinningMode = useProjectionStore((s) => s.pinningMode);
  const lodLevel = useProjectionStore((s) => s.lodLevel);
  const useRobustRange = useProjectionStore((s) => s.useRobustRange);
  const colormap = useProjectionStore((s) => s.colormap);
  const opacity = useProjectionStore((s) => s.opacity);
  const wireframe = useProjectionStore((s) => s.wireframe);
  const surfaceData = useProjectionStore((s) => s.surfaceData);
  const isLoading = useProjectionStore((s) => s.isLoading);

  const s = useProjectionStore.getState();
  const pcaDisabled = availableFeatures.length < 3;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target && (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA');
      if (typing) return;
      if (e.key === 'w' || e.key === 'W') {
        const st = useProjectionStore.getState();
        st.setWireframe(!st.wireframe);
      } else if (e.key === 'p' || e.key === 'P') {
        const st = useProjectionStore.getState();
        if (st.mode === 'axis_explorer' && availableFeatures.length < 3) return;
        st.setMode(st.mode === 'axis_explorer' ? 'pca_3d' : 'axis_explorer');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [availableFeatures.length]);

  const ranked = [...availableFeatures].sort(
    (a, b) => Math.abs(b.importance ?? 0) - Math.abs(a.importance ?? 0),
  );
  const rankOf = (name: string) => ranked.findIndex((f) => f.name === name) + 1;

  const selectStyle: React.CSSProperties = {
    background: 'var(--surface-elevated)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-subtle)',
  };
  const segOn: React.CSSProperties = { background: 'var(--ml-primary)', color: '#fff', border: '1px solid var(--border-subtle)' };
  const segOff: React.CSSProperties = { background: 'transparent', color: 'var(--text-secondary)', border: '1px solid var(--border-subtle)' };

  const lods: Array<{ key: LodLevel; label: string }> = [
    { key: 'low', label: `Low ${LOD_RESOLUTIONS.low}×${LOD_RESOLUTIONS.low}` },
    { key: 'standard', label: `Std ${LOD_RESOLUTIONS.standard}×${LOD_RESOLUTIONS.standard}` },
    { key: 'high', label: `High ${LOD_RESOLUTIONS.high}×${LOD_RESOLUTIONS.high}` },
  ];
  const tris = surfaceData ? 2 * (surfaceData.gridResolution - 1) * (surfaceData.gridResolution - 1) : null;

  return (
    <div
      className="flex flex-wrap items-center gap-2 rounded-lg px-3 py-2 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex overflow-hidden rounded" role="tablist" aria-label="Projection mode">
        <button
          role="tab"
          aria-selected={mode === 'axis_explorer'}
          onClick={() => s.setMode('axis_explorer')}
          className="px-2 py-1"
          style={mode === 'axis_explorer' ? segOn : segOff}
        >
          Axis Explorer
        </button>
        <button
          role="tab"
          aria-selected={mode === 'pca_3d'}
          disabled={pcaDisabled}
          title={pcaDisabled ? 'Needs at least 3 features for a 3-D PCA manifold' : '3D PCA manifold (key P)'}
          onClick={() => s.setMode('pca_3d')}
          className="px-2 py-1 disabled:opacity-40"
          style={mode === 'pca_3d' ? segOn : segOff}
        >
          3D PCA
        </button>
      </div>

      {mode === 'axis_explorer' ? (
        <>
          <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
            X:
            <select
              aria-label="X-Axis Feature"
              className="max-w-40 rounded px-1 py-1"
              style={selectStyle}
              value={xFeature}
              disabled={isLoading || availableFeatures.length === 0}
              onChange={(e) => useProjectionStore.getState().setXFeature(e.target.value)}
            >
              {availableFeatures.map((f) => (
                <option key={f.name} value={f.name}>
                  [#{(rankOf(f.name) || 0)}] {f.name}
                </option>
              ))}
            </select>
          </label>
          <button
            title="Swap X/Y axes"
            onClick={() => useProjectionStore.getState().swapAxes()}
            className="rounded px-1 py-1"
            style={selectStyle}
          >
            ⇄
          </button>
          <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
            Y:
            <select
              aria-label="Y-Axis Feature"
              className="max-w-40 rounded px-1 py-1"
              style={selectStyle}
              value={yFeature}
              disabled={isLoading || availableFeatures.length === 0}
              onChange={(e) => useProjectionStore.getState().setYFeature(e.target.value)}
            >
              {availableFeatures.map((f) => (
                <option key={f.name} value={f.name}>
                  [#{(rankOf(f.name) || 0)}] {f.name}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
            Pin others:
            <select
              aria-label="Background feature pinning"
              title="How non-axis features are frozen while sweeping X×Y"
              className="rounded px-1 py-1"
              style={selectStyle}
              value={pinningMode}
              onChange={(e) => useProjectionStore.getState().setPinningMode(e.target.value as typeof pinningMode)}
            >
              <option value="median">Dataset Median (Recommended)</option>
              <option value="mean">Dataset Mean</option>
              <option value="zero">Zero Baseline</option>
              <option value="current_row">Dynamic Current Row Follow</option>
            </select>
          </label>
          <label
            className="flex items-center gap-1"
            style={{ color: 'var(--text-secondary)' }}
            title="Clamp the X/Y sweep to the 1st–99th percentile so one outlier cannot flatten the surface"
          >
            <input
              type="checkbox"
              checked={useRobustRange}
              onChange={(e) => useProjectionStore.getState().setUseRobustRange(e.target.checked)}
            />
            P01–P99
          </label>
        </>
      ) : (
        <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          Standardized PCA manifold — axes are variance-ranked component scores.
        </span>
      )}

      <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
        Z:
        <select
          aria-label="Z-Axis Target Mode"
          className="rounded px-1 py-1"
          style={selectStyle}
          value={zMode}
          onChange={(e) => useProjectionStore.getState().setZMode(e.target.value as typeof zMode)}
        >
          <option value="prediction">Predicted Value</option>
          <option value="probability">Class Probability</option>
          <option value="residual_error">Prediction Error |y − ŷ|</option>
          <option value="feature_3">Feature 3</option>
        </select>
      </label>

      <div className="flex overflow-hidden rounded" aria-label="LOD resolution">
        {lods.map((l) => (
          <button
            key={l.key}
            onClick={() => useProjectionStore.getState().setLodLevel(l.key)}
            className="font-mono-tabular px-2 py-1 text-[11px]"
            style={lodLevel === l.key ? segOn : segOff}
            title={`${(2 * (LOD_RESOLUTIONS[l.key] - 1) * (LOD_RESOLUTIONS[l.key] - 1)).toLocaleString()} triangles`}
          >
            {l.label}
          </button>
        ))}
      </div>

      <details className="relative">
        <summary className="cursor-pointer rounded px-2 py-1" style={selectStyle}>
          🎨 Appearance
        </summary>
        <div
          className="absolute z-30 mt-1 w-56 rounded-lg p-3"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-strong)' }}
        >
          <label className="mb-2 block">
            <span style={{ color: 'var(--text-secondary)' }}>Colormap</span>
            <select
              className="mt-1 w-full rounded px-1 py-1"
              style={selectStyle}
              value={colormap}
              onChange={(e) => useProjectionStore.getState().setColormap(e.target.value as ColormapName)}
            >
              {(Object.keys(COLORMAP_LABELS) as ColormapName[]).map((c) => (
                <option key={c} value={c}>
                  {COLORMAP_LABELS[c]}
                </option>
              ))}
            </select>
          </label>
          <label className="mb-2 block">
            <span style={{ color: 'var(--text-secondary)' }}>Opacity {(opacity * 100).toFixed(0)}%</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.01}
              value={opacity}
              onChange={(e) => useProjectionStore.getState().setOpacity(Number(e.target.value))}
              className="w-full"
            />
          </label>
          <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
            <input
              type="checkbox"
              checked={wireframe}
              onChange={(e) => useProjectionStore.getState().setWireframe(e.target.checked)}
            />
            Wireframe overlay (W)
          </label>
        </div>
      </details>

      <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
        {isLoading
          ? 'Baking surface…'
          : surfaceData
            ? `${(tris ?? 0).toLocaleString()} tris • ${surfaceData.computationTimeMs.toFixed(1)} ms • Z ${surfaceData.zRange[0].toPrecision(3)}…${surfaceData.zRange[1].toPrecision(3)}`
            : 'No surface yet'}
      </span>
      {surfaceData?.hasConstantAxisWarning && (
        <span className="rounded px-2 py-1 text-[11px]" style={{ background: 'rgba(251,191,36,0.15)', color: '#fbbf24' }}>
          Constant axis — coordinates pinned to 0
        </span>
      )}
      {surfaceData && surfaceData.resolvedZMode !== zMode && (
        <span className="rounded px-2 py-1 text-[11px]" style={{ background: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>
          Z fell back to {surfaceData.resolvedZMode}
        </span>
      )}
    </div>
  );
}

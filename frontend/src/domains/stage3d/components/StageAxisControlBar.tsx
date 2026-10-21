import { useStage3DStore } from '../store/useStage3DStore';

/** Axis assignment dropdowns + layer visibility pills + surface LOD selector. */
export function StageAxisControlBar() {
  const featureNames = useStage3DStore((s) => s.featureNames);
  const axisX = useStage3DStore((s) => s.axisXFeatureIndex);
  const axisY = useStage3DStore((s) => s.axisYFeatureIndex);
  const setAxes = useStage3DStore((s) => s.setAxes);
  const surfaceResolution = useStage3DStore((s) => s.surfaceResolution);
  const setSurfaceResolution = useStage3DStore((s) => s.setSurfaceResolution);
  const isLoading = useStage3DStore((s) => s.isLoading);
  const layers = useStage3DStore((s) => ({
    showSurface: s.showSurface,
    showDropLines: s.showDropLines,
    showResidualBars: s.showResidualBars,
    showBackgroundCloud: s.showBackgroundCloud,
    showTickLabels: s.showTickLabels,
  }));
  const toggleLayer = useStage3DStore((s) => s.toggleLayer);

  const selectStyle: React.CSSProperties = {
    background: 'var(--surface-elevated)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-subtle)',
  };

  const pills: Array<{ key: Parameters<typeof toggleLayer>[0]; label: string }> = [
    { key: 'showSurface', label: '3D Model Surface' },
    { key: 'showDropLines', label: 'Drop-Lines & Shadows' },
    { key: 'showResidualBars', label: 'Residual Error Bars' },
    { key: 'showBackgroundCloud', label: 'Background Point Cloud' },
    { key: 'showTickLabels', label: 'Axis Tick Labels' },
  ];

  return (
    <div
      className="flex flex-wrap items-center gap-2 rounded-lg px-3 py-2 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
        <span className="font-mono-tabular inline-block h-2 w-2 rounded-full" style={{ background: '#f43f5e' }} />
        X:
        <select
          aria-label="X-Axis Feature"
          className="rounded px-1 py-1"
          style={selectStyle}
          value={axisX}
          disabled={isLoading || featureNames.length === 0}
          onChange={(e) => void setAxes(Number(e.target.value), axisY)}
        >
          {featureNames.map((n, i) => (
            <option key={i} value={i}>
              {n}
            </option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
        <span className="font-mono-tabular inline-block h-2 w-2 rounded-full" style={{ background: '#22d3ee' }} />
        Y:
        <select
          aria-label="Y-Axis Feature"
          className="rounded px-1 py-1"
          style={selectStyle}
          value={axisY}
          disabled={isLoading || featureNames.length === 0}
          onChange={(e) => void setAxes(axisX, Number(e.target.value))}
        >
          {featureNames.map((n, i) => (
            <option key={i} value={i}>
              {n}
            </option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
        Z: <span style={{ color: 'var(--text-primary)' }}>Predicted value</span>
      </label>
      <label className="flex items-center gap-1" style={{ color: 'var(--text-secondary)' }}>
        LOD:
        <select
          aria-label="Surface resolution"
          className="rounded px-1 py-1"
          style={selectStyle}
          value={surfaceResolution}
          disabled={isLoading}
          onChange={(e) => void setSurfaceResolution(Number(e.target.value))}
        >
          {[30, 50, 80].map((r) => (
            <option key={r} value={r}>
              {r}×{r}
            </option>
          ))}
        </select>
      </label>
      <div className="flex flex-wrap gap-1">
        {pills.map((p) => {
          const on = layers[p.key];
          return (
            <button
              key={p.key}
              onClick={() => toggleLayer(p.key)}
              className="font-mono-tabular rounded px-2 py-1 text-[11px]"
              style={{
                background: on ? 'var(--surface-elevated)' : 'transparent',
                color: on ? 'var(--text-primary)' : 'var(--text-muted)',
                border: '1px solid var(--border-subtle)',
              }}
              aria-pressed={on}
            >
              [{on ? 'x' : ' '}] {p.label}
            </button>
          );
        })}
      </div>
      {isLoading && (
        <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Rebuilding scene…
        </span>
      )}
    </div>
  );
}

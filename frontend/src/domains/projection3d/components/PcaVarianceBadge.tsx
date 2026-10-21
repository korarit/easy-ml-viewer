import { useProjectionStore } from '../store/useProjectionStore';

/** Phase 09 — % variance explained HUD + PCA loadings matrix popover. */
export function PcaVarianceBadge({ featureNames }: { featureNames: string[] }) {
  const mode = useProjectionStore((s) => s.mode);
  const surfaceData = useProjectionStore((s) => s.surfaceData);

  if (mode !== 'pca_3d' || !surfaceData?.pcaVarianceRatio) return null;
  const [v1, v2, v3] = surfaceData.pcaVarianceRatio;
  const total = v1 + v2 + v3;
  const bars: Array<{ label: string; value: number; color: string }> = [
    { label: 'PC1', value: v1, color: '#4f8ff7' },
    { label: 'PC2', value: v2, color: '#34d399' },
    { label: 'PC3', value: v3, color: '#fbbf24' },
  ];
  const loadings = surfaceData.pcaLoadings;

  return (
    <div
      className="pointer-events-auto absolute top-2 left-2 w-56 rounded-lg p-2 text-[11px]"
      style={{ background: 'rgba(10,15,22,0.82)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
    >
      <div className="font-mono-tabular mb-1 font-semibold">
        3D PCA Manifold · Total {(total * 100).toFixed(1)}%
      </div>
      {bars.map((b) => (
        <div key={b.label} className="mb-1 flex items-center gap-1">
          <span className="font-mono-tabular w-7" style={{ color: b.color }}>
            {b.label}
          </span>
          <div className="h-1.5 flex-1 rounded" style={{ background: 'var(--surface-elevated)' }}>
            <div className="h-1.5 rounded" style={{ width: `${(b.value * 100).toFixed(1)}%`, background: b.color }} />
          </div>
          <span className="font-mono-tabular w-12 text-right" style={{ color: 'var(--text-secondary)' }}>
            {(b.value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
      {loadings && (
        <details className="mt-1">
          <summary className="cursor-pointer" style={{ color: 'var(--ml-primary)' }}>
            View PCA Loadings Matrix
          </summary>
          <div className="mt-1 max-h-40 overflow-auto">
            <table className="font-mono-tabular w-full text-[10px]">
              <thead>
                <tr style={{ color: 'var(--text-muted)' }}>
                  <th className="text-left">feature</th>
                  <th className="text-right">PC1</th>
                  <th className="text-right">PC2</th>
                  <th className="text-right">PC3</th>
                </tr>
              </thead>
              <tbody>
                {featureNames.map((name, i) => (
                  <tr key={name} style={{ color: 'var(--text-secondary)' }}>
                    <td className="max-w-20 truncate">{name}</td>
                    {[0, 1, 2].map((pc) => (
                      <td key={pc} className="text-right">
                        {loadings[pc]?.[i] !== undefined ? loadings[pc][i].toFixed(3) : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}

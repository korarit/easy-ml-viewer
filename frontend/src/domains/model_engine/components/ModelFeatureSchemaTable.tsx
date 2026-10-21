import { useModelStore } from '../store/useModelStore';

function fmtWeight(v: number | null): string {
  if (v === null || v === undefined) return '—';
  const abs = Math.abs(v);
  if (abs !== 0 && (abs >= 1e7 || abs < 1e-4)) return v.toExponential(3);
  return String(Math.round(v * 1000) / 1000);
}

export function ModelFeatureSchemaTable() {
  const { currentModel } = useModelStore();
  if (!currentModel) return null;

  return (
    <div className="rounded-lg" style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}>
      <p className="border-b px-3 py-2 text-xs font-semibold tracking-wide" style={{ borderColor: 'var(--border-subtle)' }}>
        EXPECTED INPUT FEATURE SCHEMA (strict matrix order 0 → {currentModel.n_features_in - 1})
      </p>
      <div className="max-h-80 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0" style={{ background: 'var(--surface-panel)' }}>
            <tr className="text-left" style={{ color: 'var(--text-muted)' }}>
              <th className="px-3 py-2 font-medium tracking-wider uppercase">Index</th>
              <th className="px-3 py-2 font-medium tracking-wider uppercase">Expected Feature Name</th>
              <th className="px-3 py-2 font-medium tracking-wider uppercase">Name Source</th>
              <th className="px-3 py-2 font-medium tracking-wider uppercase">Required DType</th>
              <th className="px-3 py-2 text-right font-medium tracking-wider uppercase">Learned Coef / Importance</th>
            </tr>
          </thead>
          <tbody>
            {currentModel.features.map((f) => (
              <tr key={f.index} className="border-t" style={{ borderColor: 'var(--border-subtle)' }}>
                <td className="font-mono-tabular px-3 py-1.5" style={{ color: 'var(--text-muted)' }}>
                  {f.index}
                </td>
                <td className="font-mono-tabular px-3 py-1.5 font-medium">{f.name}</td>
                <td className="px-3 py-1.5">
                  {f.is_auto_generated_name ? (
                    <span
                      className="rounded px-1.5 py-0.5 text-[11px] font-medium"
                      style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--ml-anomaly)' }}
                    >
                      Auto-Generated Placeholder
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-secondary)' }}>Embedded Name</span>
                  )}
                </td>
                <td className="font-mono-tabular px-3 py-1.5">
                  <span className="rounded px-1.5 py-0.5 text-[11px]" style={{ background: 'var(--surface-elevated)', color: 'var(--ml-primary)' }}>
                    {f.expected_dtype}
                  </span>
                </td>
                <td className="font-mono-tabular px-3 py-1.5 text-right">{fmtWeight(f.learned_weight_or_importance)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

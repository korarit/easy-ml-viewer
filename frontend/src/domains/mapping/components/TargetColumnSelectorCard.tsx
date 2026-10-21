import type { ColumnProfile } from '../../dataset/store/useDatasetStore';
import { useMappingStore } from '../store/useMappingStore';

export function TargetColumnSelectorCard({ columns }: { columns: ColumnProfile[] }) {
  const { targetColumnName, targetLabelMap, setTargetColumn, setTargetLabelMap } = useMappingStore();
  const selected = columns.find((c) => c.name === targetColumnName);

  return (
    <section
      className="flex flex-col gap-2 rounded-lg p-3"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      aria-label="Optional ground-truth target"
    >
      <h2 className="text-xs font-semibold tracking-wide" style={{ color: 'var(--text-primary)' }}>
        OPTIONAL ACTUAL TARGET (GROUND TRUTH FOR ERROR EVALUATION)
      </h2>
      <label className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
        Target Column
        <select
          value={targetColumnName ?? ''}
          onChange={(e) => void setTargetColumn(e.target.value === '' ? null : e.target.value)}
          className="font-mono-tabular rounded px-2 py-1.5 text-xs"
          style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
        >
          <option value="">None (Unlabeled Inference Only)</option>
          {columns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} ({c.logical_type}
              {c.min_value !== null && c.max_value !== null ? ` | Min: ${c.min_value} .. Max: ${c.max_value}` : ''})
            </option>
          ))}
        </select>
      </label>
      {selected && selected.logical_type !== 'numeric' && selected.logical_type !== 'boolean' && (
        <div className="flex flex-col gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <span>
            Class Label Mapper — map text labels in <span className="font-mono-tabular">{selected.name}</span> to model classes:
          </span>
          {selected.sample_values.slice(0, 8).map((v) => {
            const key = String(v);
            return (
              <label key={key} className="font-mono-tabular flex items-center gap-2">
                <span className="w-32 truncate">“{key}” →</span>
                <input
                  type="number"
                  step="any"
                  value={targetLabelMap?.[key] ?? ''}
                  placeholder="0"
                  onChange={(e) => {
                    const next = { ...(targetLabelMap ?? {}) };
                    if (e.target.value === '') delete next[key];
                    else next[key] = Number(e.target.value);
                    setTargetLabelMap(next);
                  }}
                  className="w-24 rounded px-1.5 py-0.5"
                  style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
                />
              </label>
            );
          })}
        </div>
      )}
    </section>
  );
}

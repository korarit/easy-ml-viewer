import { Plus, X } from 'lucide-react';
import {
  createFilterRule,
  useTrajectoryQueueStore,
  type QueryOperator,
} from '../store/useTrajectoryQueueStore';

const OPERATORS: { id: QueryOperator; label: string }[] = [
  { id: 'eq', label: '==' },
  { id: 'neq', label: '!=' },
  { id: 'gt', label: '>' },
  { id: 'gte', label: '>=' },
  { id: 'lt', label: '<' },
  { id: 'lte', label: '<=' },
  { id: 'between', label: 'BETWEEN' },
];

interface Props {
  availableColumns: string[];
}

/** Visual rule-chip filter builder — structured rules only, no raw expressions (Phase 06 §2.1). */
export function FilterConditionBar({ availableColumns }: Props) {
  const { filters, setFilters } = useTrajectoryQueueStore();

  const addRule = () => {
    setFilters([...filters, createFilterRule(availableColumns[0] ?? '', 'gte', 0)]);
  };

  const patchRule = (id: string, patch: Partial<(typeof filters)[number]>) => {
    setFilters(filters.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  };

  const removeRule = (id: string) => {
    setFilters(filters.filter((r) => r.id !== id));
  };

  return (
    <div
      className="flex flex-col gap-2 rounded-md px-3 py-2"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          Filters {filters.length > 0 && `(${filters.length} AND)`}
        </span>
        <div className="flex gap-2">
          {filters.length > 0 && (
            <button
              onClick={() => setFilters([])}
              className="rounded px-2 py-1 text-xs underline"
              style={{ color: 'var(--text-secondary)' }}
            >
              Reset
            </button>
          )}
          <button
            onClick={addRule}
            className="flex items-center gap-1 rounded-md px-2.5 py-1.5 text-xs font-medium"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            <Plus size={13} /> Add Rule (Enter)
          </button>
        </div>
      </div>

      {filters.length === 0 && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          No filters — showing all rows. Add a rule like [area_sqm] [&gt;=] [60].
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {filters.map((rule) => (
          <div
            key={rule.id}
            className="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-xs"
            style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
          >
            <select
              value={rule.column}
              onChange={(e) => patchRule(rule.id, { column: e.target.value })}
              aria-label="Filter column"
              className="max-w-36 rounded px-1 py-0.5"
              style={{ background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)' }}
            >
              {availableColumns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <select
              value={rule.operator}
              onChange={(e) => patchRule(rule.id, { operator: e.target.value as QueryOperator })}
              aria-label="Filter operator"
              className="rounded px-1 py-0.5"
              style={{ background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)' }}
            >
              {OPERATORS.map((op) => (
                <option key={op.id} value={op.id}>
                  {op.label}
                </option>
              ))}
            </select>
            <input
              value={String(rule.value ?? '')}
              onChange={(e) => {
                const raw = e.target.value;
                const num = Number(raw);
                patchRule(rule.id, { value: raw === '' ? null : Number.isNaN(num) ? raw : num });
              }}
              inputMode="decimal"
              aria-label="Filter value"
              className="w-20 rounded px-1.5 py-0.5 font-mono-tabular"
              style={{ background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)' }}
            />
            {rule.operator === 'between' && (
              <>
                <span style={{ color: 'var(--text-muted)' }}>and</span>
                <input
                  value={rule.value_to == null ? '' : String(rule.value_to)}
                  onChange={(e) => {
                    const num = Number(e.target.value);
                    patchRule(rule.id, { value_to: e.target.value === '' ? null : num });
                  }}
                  inputMode="decimal"
                  aria-label="Filter upper bound"
                  className="w-20 rounded px-1.5 py-0.5 font-mono-tabular"
                  style={{ background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)' }}
                />
              </>
            )}
            <button
              onClick={() => removeRule(rule.id)}
              aria-label="Remove filter rule"
              className="rounded p-0.5"
              style={{ color: 'var(--text-muted)' }}
            >
              <X size={13} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

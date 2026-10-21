import { useMemo } from 'react';
import { useDatasetStore, type ColumnProfile, type TypeFilter } from '../store/useDatasetStore';

const FILTERS: Array<{ id: TypeFilter; label: string }> = [
  { id: 'all', label: 'All' },
  { id: 'numeric', label: 'Numeric' },
  { id: 'categorical', label: 'Categorical' },
  { id: 'boolean', label: 'Boolean' },
  { id: 'datetime', label: 'Datetime' },
];

function Sparkline({ counts }: { counts: number[] }) {
  if (!counts.length) return <span style={{ color: 'var(--text-muted)' }}>[Text Col]</span>;
  const peak = Math.max(...counts, 1);
  return (
    <svg width="90" height="18" role="img" aria-label="distribution histogram">
      {counts.map((c, i) => {
        const h = Math.max(2, Math.round((c / peak) * 16));
        const w = 90 / counts.length;
        return (
          <rect
            key={i}
            x={i * w + 0.5}
            y={18 - h}
            width={Math.max(1, w - 1)}
            height={h}
            fill="var(--ml-primary)"
            opacity={0.55 + 0.45 * (c / peak)}
          />
        );
      })}
    </svg>
  );
}

function fmtStat(v: number | null): string {
  if (v === null || v === undefined) return '—';
  const abs = Math.abs(v);
  if (abs !== 0 && (abs >= 1e7 || abs < 1e-3)) return v.toExponential(2);
  return String(Math.round(v * 100) / 100);
}

function ColumnRow({ col }: { col: ColumnProfile }) {
  return (
    <tr className="border-t" style={{ borderColor: 'var(--border-subtle)' }}>
      <td className="font-mono-tabular px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>
        {col.index}
      </td>
      <td className="font-mono-tabular px-2 py-1.5 font-medium">{col.name}</td>
      <td className="px-2 py-1.5">
        <span
          className="font-mono-tabular rounded px-1.5 py-0.5 text-[11px]"
          style={{ background: 'var(--surface-elevated)', color: 'var(--ml-primary)' }}
        >
          {col.storage_dtype}
        </span>
      </td>
      <td className="px-2 py-1.5">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-1.5 w-14 overflow-hidden rounded-full"
            style={{ background: 'var(--surface-elevated)' }}
            title={`${col.null_count} nulls`}
          >
            <span
              className="block h-full rounded-full"
              style={{
                width: `${Math.min(100, col.null_percentage)}%`,
                background: col.null_percentage > 5 ? 'var(--ml-anomaly)' : 'var(--ml-class-0)',
              }}
            />
          </span>
          <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            {col.null_percentage.toFixed(2)}%
          </span>
        </span>
      </td>
      <td className="font-mono-tabular px-2 py-1.5 text-right">{col.unique_count.toLocaleString()}</td>
      <td className="font-mono-tabular px-2 py-1.5 text-right" style={{ color: 'var(--text-secondary)' }}>
        {fmtStat(col.min_value)} … {fmtStat(col.max_value)}
      </td>
      <td className="font-mono-tabular px-2 py-1.5 text-right" style={{ color: 'var(--text-secondary)' }}>
        {col.logical_type === 'numeric' ? `${fmtStat(col.mean_value)} ± ${fmtStat(col.std_value)}` : '—'}
      </td>
      <td className="px-2 py-1.5">
        <Sparkline counts={col.histogram_counts} />
      </td>
      <td className="font-mono-tabular max-w-40 truncate px-2 py-1.5 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {col.sample_values.map((v) => String(v)).join(', ')}
      </td>
    </tr>
  );
}

export function ColumnProfileTable() {
  const { currentDataset, searchQuery, setSearchQuery, typeFilter, setTypeFilter } = useDatasetStore();

  const filtered = useMemo(() => {
    if (!currentDataset) return [];
    const q = searchQuery.trim().toLowerCase();
    return currentDataset.columns.filter(
      (c) =>
        (typeFilter === 'all' || c.logical_type === typeFilter) &&
        (q === '' || c.name.toLowerCase().includes(q)),
    );
  }, [currentDataset, searchQuery, typeFilter]);

  if (!currentDataset) return null;
  const counts = (t: TypeFilter) =>
    t === 'all' ? currentDataset.columns.length : currentDataset.columns.filter((c) => c.logical_type === t).length;

  return (
    <div className="rounded-lg" style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}>
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2" style={{ borderColor: 'var(--border-subtle)' }}>
        <input
          id="column-search"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search column name… (Ctrl+F)"
          className="rounded-md px-2.5 py-1.5 text-xs"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
        />
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by column type">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setTypeFilter(f.id)}
              className="rounded-md px-2 py-1 text-[11px] font-medium"
              style={{
                border: '1px solid var(--border-subtle)',
                background: typeFilter === f.id ? 'var(--ml-primary)' : 'transparent',
                color: typeFilter === f.id ? '#fff' : 'var(--text-secondary)',
              }}
            >
              {f.label} ({counts(f.id)})
            </button>
          ))}
        </div>
      </div>
      <div className="max-h-80 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0" style={{ background: 'var(--surface-panel)' }}>
            <tr className="text-left" style={{ color: 'var(--text-muted)' }}>
              <th className="px-2 py-2 font-medium tracking-wider uppercase">#</th>
              <th className="px-2 py-2 font-medium tracking-wider uppercase">Column</th>
              <th className="px-2 py-2 font-medium tracking-wider uppercase">DType</th>
              <th className="px-2 py-2 font-medium tracking-wider uppercase">Null %</th>
              <th className="px-2 py-2 text-right font-medium tracking-wider uppercase">Unique</th>
              <th className="px-2 py-2 text-right font-medium tracking-wider uppercase">Min … Max</th>
              <th className="px-2 py-2 text-right font-medium tracking-wider uppercase">Mean ± SD</th>
              <th className="px-2 py-2 font-medium tracking-wider uppercase">Distribution</th>
              <th className="px-2 py-2 font-medium tracking-wider uppercase">Samples</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c) => (
              <ColumnRow key={c.index} col={c} />
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <p className="px-4 py-6 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
            No columns match the current filter.
          </p>
        )}
      </div>
    </div>
  );
}

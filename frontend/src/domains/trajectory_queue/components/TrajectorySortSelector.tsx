import { useTrajectoryQueueStore } from '../store/useTrajectoryQueueStore';

interface Props {
  availableColumns: string[];
}

/** Playback-order sorter (Phase 06 §2.1): original order or sort by any column. */
export function TrajectorySortSelector({ availableColumns }: Props) {
  const { sortBy, sortAscending, setSort } = useTrajectoryQueueStore();

  return (
    <label
      className="flex items-center gap-2 rounded-md px-3 py-2 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
    >
      <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
        Playback order
      </span>
      <select
        value={sortBy ?? '__original__'}
        onChange={(e) => {
          const v = e.target.value;
          setSort(v === '__original__' ? null : v, sortAscending);
        }}
        aria-label="Trajectory playback order"
        className="rounded px-1.5 py-1 text-xs"
        style={{ background: 'var(--bg-canvas)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
      >
        <option value="__original__">Original CSV Row Order</option>
        {availableColumns.map((c) => (
          <option key={c} value={c}>
            {c} (Low → High)
          </option>
        ))}
      </select>
      <button
        onClick={() => setSort(sortBy, !sortAscending)}
        disabled={!sortBy}
        title={sortAscending ? 'Ascending — click for descending' : 'Descending — click for ascending'}
        className="rounded px-2 py-1 font-mono-tabular text-xs disabled:opacity-40"
        style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
      >
        {sortAscending ? '↑ Asc' : '↓ Desc'}
      </button>
    </label>
  );
}

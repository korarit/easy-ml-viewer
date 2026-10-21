import { useState } from 'react';
import { useTrajectoryQueueStore, type QueuePreset } from '../store/useTrajectoryQueueStore';

const PRESETS: { id: QueuePreset; label: string; title: string }[] = [
  { id: 'first_100', label: 'First 100', title: 'Select first 100 matching rows' },
  { id: 'random_100', label: 'Random 100', title: 'Select seeded-random 100 rows (seed 42)' },
  { id: 'filtered_all', label: 'Select Filtered', title: 'Select all matching rows (capped at 10,000 animated frames)' },
  { id: 'clear', label: 'Clear (Esc)', title: 'Clear the trajectory queue' },
];

/** Preset + range selection toolbar (Phase 06 §2.1). */
export function DataSelectorToolbar() {
  const { selectPreset, selectRange, isResolving } = useTrajectoryQueueStore();
  const [rangeFrom, setRangeFrom] = useState('0');
  const [rangeTo, setRangeTo] = useState('250');
  const [rangeStep, setRangeStep] = useState('1');

  const applyRange = () => {
    const from = Number.parseInt(rangeFrom, 10);
    const to = Number.parseInt(rangeTo, 10);
    const step = Number.parseInt(rangeStep, 10);
    if (Number.isNaN(from) || Number.isNaN(to) || Number.isNaN(step) || step < 1) return;
    selectRange(from, to, step);
  };

  return (
    <div
      className="flex flex-wrap items-center gap-2 rounded-md px-3 py-2"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
        Presets
      </span>
      {PRESETS.map((p) => (
        <button
          key={p.id}
          disabled={isResolving}
          onClick={() => void selectPreset(p.id)}
          title={p.title}
          className="rounded-md px-2.5 py-1.5 text-xs font-medium disabled:opacity-40"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          {p.label}
        </button>
      ))}
      <span className="mx-1 text-xs" style={{ color: 'var(--text-muted)' }}>|</span>
      <label className="flex items-center gap-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
        Range
        <input
          value={rangeFrom}
          onChange={(e) => setRangeFrom(e.target.value)}
          inputMode="numeric"
          aria-label="Range from row"
          className="w-16 rounded px-1.5 py-1 font-mono-tabular text-xs"
          style={{ border: '1px solid var(--border-strong)', background: 'var(--bg-canvas)' }}
        />
        <span>..</span>
        <input
          value={rangeTo}
          onChange={(e) => setRangeTo(e.target.value)}
          inputMode="numeric"
          aria-label="Range to row"
          className="w-16 rounded px-1.5 py-1 font-mono-tabular text-xs"
          style={{ border: '1px solid var(--border-strong)', background: 'var(--bg-canvas)' }}
        />
        <span>Step</span>
        <input
          value={rangeStep}
          onChange={(e) => setRangeStep(e.target.value)}
          inputMode="numeric"
          aria-label="Range step stride"
          className="w-12 rounded px-1.5 py-1 font-mono-tabular text-xs"
          style={{ border: '1px solid var(--border-strong)', background: 'var(--bg-canvas)' }}
        />
        <button
          onClick={applyRange}
          className="rounded-md px-2.5 py-1.5 text-xs font-medium"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          Apply
        </button>
      </label>
      {isResolving && (
        <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--ml-primary)' }}>
          Resolving selection…
        </span>
      )}
    </div>
  );
}

import { ArrowDownWideNarrow, ListOrdered, Sparkles } from 'lucide-react';
import { useMappingStore } from '../store/useMappingStore';
import type { ImputationStrategy } from '../api/mappingApi';

const NAN_OPTIONS: Array<{ value: ImputationStrategy; label: string }> = [
  { value: 'median', label: 'Fill Median' },
  { value: 'mean', label: 'Fill Mean' },
  { value: 'zero', label: 'Fill Zero (0.0)' },
  { value: 'custom', label: 'Custom Value' },
  { value: 'pass_nan', label: 'Keep NaN' },
];

export function MappingHeaderBar({
  onAutoMatch,
  onMapByOrder,
}: {
  onAutoMatch: () => void;
  onMapByOrder: () => void;
}) {
  const {
    items,
    isValid,
    isLoading,
    featureErrors,
    duplicateColumns,
    setGlobalNaNStrategy,
    supportsNan,
  } = useMappingStore();

  const mapped = items.filter((i) => i.dataset_column_name).length;
  const exact = items.filter((i) => i.match_tier === 'exact').length;
  const fuzzy = items.filter((i) => i.match_tier === 'fuzzy' || i.match_tier === 'normalized').length;
  const errors = Object.keys(featureErrors).length;

  return (
    <div
      className="flex flex-col gap-2 rounded-lg p-3"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-sm font-semibold tracking-wide" style={{ color: 'var(--text-primary)' }}>
          STEP 3: COLUMN MAPPING &amp; TYPE CASTING
        </h1>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={onAutoMatch}
            disabled={isLoading}
            title="Re-run smart auto-match (Ctrl+M)"
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            <Sparkles size={13} /> Re-Run Smart Auto-Match
          </button>
          <button
            onClick={onMapByOrder}
            disabled={isLoading}
            title="Map by column order 0..N"
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            <ListOrdered size={13} /> Map by Order 0..N
          </button>
          <label
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title={supportsNan ? 'Global NaN policy' : 'Target model rejects NaN — Keep NaN disabled downstream'}
          >
            <ArrowDownWideNarrow size={13} />
            NaN:
            <select
              onChange={(e) => void setGlobalNaNStrategy(e.target.value as ImputationStrategy)}
              defaultValue="median"
              className="rounded px-1 py-0.5 text-xs"
              style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
            >
              {NAN_OPTIONS.map((o) => (
                <option key={o.value} value={o.value} disabled={o.value === 'pass_nan' && !supportsNan}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div className="flex flex-wrap gap-2 text-[11px]" aria-live="polite">
        <span
          className="font-mono-tabular rounded px-2 py-1"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
        >
          Mapped: {mapped} / {items.length} Features
        </span>
        <span
          className="font-mono-tabular rounded px-2 py-1"
          style={{ background: 'rgba(16,185,129,0.12)', color: 'var(--ml-success, #10b981)' }}
        >
          Exact/High Confidence: {exact}
        </span>
        <span
          className="font-mono-tabular rounded px-2 py-1"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
        >
          Fuzzy Review Needed: {fuzzy}
        </span>
        <span
          className="font-mono-tabular rounded px-2 py-1"
          style={{
            background: errors ? 'rgba(244,63,94,0.12)' : 'var(--surface-elevated)',
            color: errors ? 'var(--ml-class-1)' : 'var(--text-secondary)',
          }}
        >
          Unmapped/Error: {errors}
        </span>
        {!isValid && items.length > 0 && (
          <span className="font-mono-tabular rounded px-2 py-1" style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}>
            {duplicateColumns.length > 0 ? `Duplicate: ${duplicateColumns.join(', ')}` : 'Resolve red rows to continue'}
          </span>
        )}
      </div>
    </div>
  );
}

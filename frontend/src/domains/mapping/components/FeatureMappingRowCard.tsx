import type { ColumnProfile } from '../../dataset/store/useDatasetStore';
import type { FeatureMappingItem, ImputationStrategy } from '../api/mappingApi';
import { useMappingStore } from '../store/useMappingStore';

const TIER_STYLE: Record<string, { label: (c: number) => string; bg: string; fg: string }> = {
  exact: { label: (c) => `100% Exact Match`, bg: 'rgba(16,185,129,0.12)', fg: 'var(--ml-success, #10b981)' },
  normalized: { label: (c) => `${Math.round(c * 100)}% Normalized Match`, bg: 'rgba(6,182,212,0.12)', fg: 'var(--ml-primary)' },
  fuzzy: { label: (c) => `${Math.round(c * 100)}% Fuzzy — Verify`, bg: 'rgba(245,158,11,0.14)', fg: 'var(--ml-anomaly)' },
  positional: { label: () => 'Positional Fallback', bg: 'rgba(129,140,248,0.14)', fg: '#818cf8' },
  manual: { label: (c) => `Manual (${Math.round(c * 100)}%)`, bg: 'var(--surface-elevated)', fg: 'var(--text-secondary)' },
  unmapped: { label: () => 'Unmapped', bg: 'rgba(244,63,94,0.12)', fg: 'var(--ml-class-1)' },
};

export function FeatureMappingRowCard({
  item,
  columns,
  weight,
  error,
}: {
  item: FeatureMappingItem;
  columns: ColumnProfile[];
  weight: number | null;
  error?: string;
}) {
  const { updateFeatureMapping, supportsNan, targetColumnName } = useMappingStore();
  const tier = TIER_STYLE[item.match_tier] ?? TIER_STYLE.unmapped;
  const selected = columns.find((c) => c.name === item.dataset_column_name);
  const isLeakage = !!targetColumnName && item.dataset_column_name === targetColumnName;

  return (
    <div
      className="grid gap-2 rounded-lg p-3 md:grid-cols-[1fr_auto_1.2fr]"
      style={{
        background: 'var(--surface-panel)',
        border: error ? '1px solid var(--ml-class-1)' : '1px solid var(--border-subtle)',
      }}
    >
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span
            className="font-mono-tabular rounded px-1.5 py-0.5 text-[11px] font-semibold"
            style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
          >
            #{item.feature_index}
          </span>
          <span className="font-mono-tabular text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
            {item.model_feature_name}
          </span>
          <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
            float32
          </span>
        </div>
        <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          Weight: {weight === null || weight === undefined ? '—' : Number(weight).toLocaleString('en-US', { maximumFractionDigits: 2 })}
        </span>
      </div>

      <div className="flex flex-col justify-center gap-1">
        <span
          className="font-mono-tabular rounded px-2 py-1 text-[11px] font-medium"
          style={{ background: error ? 'rgba(244,63,94,0.12)' : tier.bg, color: error ? 'var(--ml-class-1)' : tier.fg }}
        >
          {error ? 'Type Incompatible Error' : tier.label(item.match_confidence)}
        </span>
        <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {error ?? item.cast_warning_or_error ?? (selected ? `Direct ${selected.storage_dtype} → f32` : 'Pick a source column')}
        </span>
      </div>

      <div className="flex flex-col gap-1.5">
        <select
          value={item.dataset_column_name ?? ''}
          onChange={(e) =>
            void updateFeatureMapping(item.feature_index, {
              dataset_column_name: e.target.value === '' ? null : e.target.value,
            })
          }
          className="font-mono-tabular rounded px-2 py-1.5 text-xs"
          style={{
            border: '1px solid var(--border-strong)',
            background: 'var(--surface-elevated)',
            color: 'var(--text-primary)',
          }}
          aria-label={`Dataset column for ${item.model_feature_name}`}
        >
          <option value="">— unmapped —</option>
          {columns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} ({c.logical_type}, {c.null_percentage.toFixed(1)}% null)
            </option>
          ))}
        </select>
        {selected && (
          <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
            Sample: {selected.sample_values.slice(0, 3).map((v) => String(v)).join(', ') || '—'}
            {' | '}NaN:
            <select
              value={item.imputation_strategy}
              onChange={(e) =>
                void updateFeatureMapping(item.feature_index, {
                  imputation_strategy: e.target.value as ImputationStrategy,
                })
              }
              className="ml-1 rounded px-1 py-0.5 text-[11px]"
              style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
              aria-label={`NaN strategy for ${item.model_feature_name}`}
            >
              <option value="median">Median{selected.median_value !== null ? ` (${selected.median_value})` : ''}</option>
              <option value="mean">Mean{selected.mean_value !== null ? ` (${selected.mean_value})` : ''}</option>
              <option value="zero">Zero (0.0)</option>
              <option value="custom">Custom Value</option>
              <option value="pass_nan" disabled={!supportsNan} title={supportsNan ? '' : 'Model rejects NaN input'}>
                Keep NaN
              </option>
            </select>
          </span>
        )}
        {item.imputation_strategy === 'custom' && (
          <label className="font-mono-tabular flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            Fill =
            <input
              type="number"
              step="any"
              value={Number.isFinite(item.custom_fill_value) ? item.custom_fill_value : 0}
              onChange={(e) =>
                void updateFeatureMapping(item.feature_index, {
                  custom_fill_value: Number(e.target.value),
                  resolved_fill_value: Number(e.target.value),
                })
              }
              className="font-mono-tabular w-24 rounded px-1.5 py-0.5"
              style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
            />
          </label>
        )}
        {isLeakage && (
          <span className="rounded px-2 py-1 text-[11px]" style={{ background: 'rgba(245,158,11,0.14)', color: 'var(--ml-anomaly)' }} role="alert">
            Target Leakage Warning: this column is also selected as Actual Target!
          </span>
        )}
      </div>
    </div>
  );
}

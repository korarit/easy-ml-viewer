import { useCallback } from 'react';
import { formatDelta, formatWhatIfValue, type FeatureChangeDelta, type FeatureRange } from '../types/whatif.types';

interface Props {
  range: FeatureRange;
  baseValue: number;
  mutatedValue: number | undefined;
  locked: boolean;
  change: FeatureChangeDelta | undefined;
  disabled: boolean;
  onChange: (featureIndex: number, value: number) => void;
  onToggleLock: (featureIndex: number) => void;
}

/**
 * Phase 13 — live mutation slider row (spec §2.1).
 *
 * Slider + numeric spinbox stay in sync; the original-value tick stays
 * pinned while the thumb moves. Integer/binary columns snap to whole
 * steps (edge #2) and out-of-envelope values raise the OOD badge.
 */
export function WhatIfFeatureSliderRow({
  range,
  baseValue,
  mutatedValue,
  locked,
  change,
  disabled,
  onChange,
  onToggleLock,
}: Props) {
  const shown = mutatedValue ?? baseValue;
  const span = range.max_value - range.min_value;
  const tickPct = span > 1e-12 ? ((baseValue - range.min_value) / span) * 100 : 50;
  const isActive = change !== undefined;
  const ood = change?.is_out_of_distribution ?? false;

  const clampAndSnap = useCallback(
    (raw: number): number => {
      if (!Number.isFinite(raw)) return baseValue;
      let v = Math.min(range.max_value * 4, Math.max(range.min_value - Math.abs(range.min_value) - span, raw));
      if (range.kind === 'integer' || range.kind === 'binary') v = Math.round(v);
      if (range.kind === 'binary') v = Math.min(1, Math.max(0, v));
      return v;
    },
    [baseValue, range, span],
  );

  return (
    <div
      data-testid={`whatif-row-${range.feature_name}`}
      className="flex flex-col gap-1 rounded-md px-2 py-1.5"
      style={{
        background: isActive ? 'rgba(56,189,248,0.08)' : 'transparent',
        border: '1px solid var(--border-subtle)',
        opacity: locked ? 0.75 : 1,
      }}
    >
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => onToggleLock(range.feature_index)}
          disabled={disabled}
          aria-pressed={locked}
          aria-label={locked ? `Unlock ${range.feature_name}` : `Lock ${range.feature_name} as immutable`}
          title={locked ? 'Immutable — counterfactual will not touch this' : 'Lock as immutable'}
          className="rounded px-1 text-[12px] leading-none disabled:opacity-40"
        >
          {locked ? '🔒' : '🔓'}
        </button>
        <span className="min-w-0 flex-1 truncate text-[12px] font-medium" title={range.feature_name}>
          {range.feature_name}
          {locked && (
            <span className="ml-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
              (Immutable)
            </span>
          )}
        </span>
        {ood && (
          <span
            data-testid={`whatif-ood-${range.feature_name}`}
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ background: 'rgba(245,158,11,0.15)', color: '#f59e0b' }}
          >
            ⚠️ Out-of-Distribution
          </span>
        )}
        {isActive && change && (
          <span className="font-mono-tabular text-[10px]" style={{ color: '#38bdf8' }}>
            {formatDelta(change.delta_value, change.delta_percent)}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <input
            type="range"
            min={range.min_value}
            max={range.max_value}
            step={range.step}
            value={Math.min(range.max_value, Math.max(range.min_value, shown))}
            disabled={disabled || locked}
            onChange={(e) => onChange(range.feature_index, clampAndSnap(Number(e.target.value)))}
            aria-label={`${range.feature_name} what-if value`}
            className="w-full"
          />
          {/* Original-value tick mark (spec §2.1). */}
          <div
            aria-hidden
            className="pointer-events-none absolute top-1/2 h-3 w-[2px] -translate-y-1/2"
            style={{ left: `${Math.min(100, Math.max(0, tickPct))}%`, background: 'var(--text-muted)' }}
          />
        </div>
        <input
          type="number"
          value={Number.isFinite(shown) ? shown : ''}
          min={range.kind === 'binary' ? 0 : undefined}
          max={range.kind === 'binary' ? 1 : undefined}
          step={range.step}
          disabled={disabled || locked}
          onChange={(e) => {
            const raw = Number(e.target.value);
            if (e.target.value !== '' && Number.isFinite(raw)) onChange(range.feature_index, clampAndSnap(raw));
          }}
          aria-label={`${range.feature_name} exact value`}
          className="font-mono-tabular w-24 rounded border px-1.5 py-0.5 text-[11px] disabled:opacity-40"
          style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-canvas)' }}
        />
      </div>

      <div className="font-mono-tabular flex justify-between text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <span>{formatWhatIfValue(range.min_value)}</span>
        <span>
          orig <span style={{ color: 'var(--text-secondary)' }}>{formatWhatIfValue(baseValue)}</span>
          {' · '}
          {range.kind}
        </span>
        <span>{formatWhatIfValue(range.max_value)}</span>
      </div>
    </div>
  );
}

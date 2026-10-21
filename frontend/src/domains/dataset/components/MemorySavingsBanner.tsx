import { useDatasetStore } from '../store/useDatasetStore';

function fmtMb(v: number): string {
  return `${v.toFixed(2)} MB`;
}

export function MemorySavingsBanner() {
  const { currentDataset } = useDatasetStore();
  if (!currentDataset) return null;
  const { original_ram_mb, optimized_ram_mb, inf_sanitized_count, detected_encoding, detected_delimiter } =
    currentDataset;
  const savedPct = original_ram_mb > 0 ? ((1 - optimized_ram_mb / original_ram_mb) * 100).toFixed(1) : '0.0';

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-mono-tabular font-medium" style={{ color: 'var(--text-primary)' }}>
          FILE: {currentDataset.filename} ({currentDataset.row_count.toLocaleString()} rows)
        </span>
        <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)' }}>
          RAM: {fmtMb(original_ram_mb)} → {fmtMb(optimized_ram_mb)} (float32 saved {savedPct}%)
        </span>
      </div>
      <div className="font-mono-tabular mt-1.5 flex flex-wrap gap-1.5" style={{ color: 'var(--text-secondary)' }}>
        {detected_encoding && (
          <span className="rounded px-1.5 py-0.5" style={{ background: 'var(--surface-elevated)' }}>
            ENC: {detected_encoding.toUpperCase()}
          </span>
        )}
        {detected_delimiter && (
          <span className="rounded px-1.5 py-0.5" style={{ background: 'var(--surface-elevated)' }}>
            DELIM: {detected_delimiter === '\t' ? 'Tab' : detected_delimiter === ',' ? 'Comma (,)' : detected_delimiter}
          </span>
        )}
        <span className="rounded px-1.5 py-0.5" style={{ background: 'var(--surface-elevated)' }}>
          STORAGE: Arrow Parquet Cache
        </span>
        {inf_sanitized_count > 0 && (
          <span className="rounded px-1.5 py-0.5" style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--ml-anomaly)' }}>
            Sanitized {inf_sanitized_count} Infinity/#DIV/0! → NaN
          </span>
        )}
      </div>
    </div>
  );
}

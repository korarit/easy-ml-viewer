import { useState } from 'react';
import { useSystemStore } from '../store/useSystemStore';
import { HardwareStatusPill } from '../../domains/compute/components/HardwareStatusPill';

function ramColor(pct: number): string {
  if (pct < 60) return 'var(--ml-success, #10b981)';
  if (pct < 85) return 'var(--ml-anomaly)';
  return 'var(--ml-class-1)';
}

export function SystemStatusBar() {
  const { health, lastError, refreshHealth, clearDiskCache } = useSystemStore();
  const [notice, setNotice] = useState<string | null>(null);
  const mem = health?.memory;
  const pct = mem?.process_usage_percent ?? 0;

  const purgeCache = async () => {
    setNotice(null);
    try {
      const freed = await clearDiskCache();
      setNotice(`Temp cache cleared — freed ${freed.toFixed(1)} MB.`);
    } catch {
      setNotice('Cache purge failed — is the backend online?');
    }
    window.setTimeout(() => setNotice(null), 4000);
  };

  return (
    <footer
      className="font-mono-tabular flex items-center gap-4 overflow-x-auto border-t px-4 py-1.5 text-[11px] whitespace-nowrap"
      style={{ background: 'var(--surface-panel)', borderColor: 'var(--border-subtle)', color: 'var(--text-secondary)' }}
    >
      {health && health.status !== 'offline' ? (
        <>
          {/* System dot uses neutral blue — data classes keep red/green. */}
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full" style={{ background: 'var(--ml-primary)' }} />
            Backend · {health.latency_ms} ms
          </span>
          <span className="flex items-center gap-2" title="App memory usage vs budget">
            RAM:
            <span className="inline-block h-2 w-28 overflow-hidden rounded-full" style={{ background: 'var(--surface-elevated)' }}>
              <span className="block h-full rounded-full" style={{ width: `${Math.min(100, pct)}%`, background: ramColor(pct) }} />
            </span>
            {mem?.process_rss_mb.toFixed(0)}/{mem?.process_budget_mb.toFixed(0)} MB
            {mem && mem.is_low_memory && <span style={{ color: 'var(--ml-class-1)' }}>· LOW</span>}
          </span>
          <span title={health.webgl2_renderer ?? 'Checking graphics…'}>
            GPU: {health.webgl2_renderer ? 'Ready' : 'Checking…'}
          </span>
          <button
            onClick={() => void purgeCache()}
            className="min-h-[28px] rounded px-2 py-0.5"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Delete temporary preview files"
          >
            Clear temp files
          </button>
          <HardwareStatusPill />
          {notice && (
            <span role="status" style={{ color: 'var(--text-primary)' }}>
              {notice}
            </span>
          )}
        </>
      ) : (
        <>
          <span className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: 'var(--text-muted)' }} />
            Backend offline{lastError ? ` — ${lastError}` : ''}
          </span>
          <button
            onClick={() => void refreshHealth()}
            className="min-h-[28px] rounded px-2 py-0.5"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            Retry now
          </button>
          {notice && (
            <span role="status" style={{ color: 'var(--text-primary)' }}>
              {notice}
            </span>
          )}
        </>
      )}
    </footer>
  );
}

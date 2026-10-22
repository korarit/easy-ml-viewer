import { useSystemStore } from '../store/useSystemStore';
import { HardwareStatusPill } from '../../domains/compute/components/HardwareStatusPill';

function ramColor(pct: number): string {
  if (pct < 60) return 'var(--ml-class-0)';
  if (pct < 85) return 'var(--ml-anomaly)';
  return 'var(--ml-class-1)';
}

export function SystemStatusBar() {
  const { health, lastError, refreshHealth, clearDiskCache } = useSystemStore();
  const mem = health?.memory;
  const pct = mem?.process_usage_percent ?? 0;

  const purgeCache = async () => {
    try {
      const freed = await clearDiskCache();
      window.alert(`Temp cache cleared — freed ${freed.toFixed(1)} MB.`);
    } catch {
      window.alert('Cache purge failed — is the backend online?');
    }
  };

  return (
    <footer
      className="font-mono-tabular flex items-center gap-4 overflow-x-auto border-t px-4 py-1.5 text-[11px] whitespace-nowrap"
      style={{ background: 'var(--surface-panel)', borderColor: 'var(--border-subtle)', color: 'var(--text-secondary)' }}
    >
      {health && health.status !== 'offline' ? (
        <>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full" style={{ background: 'var(--ml-class-0)' }} />
            Backend: Online ({health.latency_ms} ms)
          </span>
          <span className="flex items-center gap-2" title={`Process RAM budget: ${mem?.process_budget_mb} MB`}>
            RAM:
            <span className="inline-block h-2 w-28 overflow-hidden rounded-full" style={{ background: 'var(--surface-elevated)' }}>
              <span className="block h-full rounded-full" style={{ width: `${Math.min(100, pct)}%`, background: ramColor(pct) }} />
            </span>
            {mem?.process_rss_mb.toFixed(0)}/{mem?.process_budget_mb.toFixed(0)} MB ({pct.toFixed(1)}%)
            {mem && mem.is_low_memory && <span style={{ color: 'var(--ml-class-1)' }}>· LOW — close other apps</span>}
          </span>
          <span>
            CPU: {health.cpu_logical_cores} Cores (OMP={health.omp_threads})
          </span>
          <span title={health.webgl2_renderer ?? ''}>
            GPU: {health.webgl2_renderer ? `WebGL2 Ready (${health.webgl2_renderer.slice(0, 32)})` : 'WebGL2: probing…'}
          </span>
          <span title="100% air-gapped — no requests leave 127.0.0.1">
            Offline Mode: 100% Local
          </span>
          <span>Cache: {health.cache_size_mb.toFixed(1)} MB</span>
          <button
            onClick={() => void purgeCache()}
            className="rounded px-2 py-0.5"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Purge .cache temp parquet files (DELETE /api/v1/system/cache)"
          >
            Clear Temp Cache
          </button>
          <HardwareStatusPill />
        </>
      ) : (
        <>
          <span className="flex items-center gap-1.5" style={{ color: 'var(--ml-class-1)' }}>
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: 'var(--ml-class-1)' }} />
            Backend: Offline {lastError ? `— ${lastError}` : ''}
          </span>
          <button
            onClick={() => void refreshHealth()}
            className="rounded px-2 py-0.5"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            Retry Now
          </button>
        </>
      )}
    </footer>
  );
}

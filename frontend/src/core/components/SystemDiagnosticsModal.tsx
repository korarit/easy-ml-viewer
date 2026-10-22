import { useState } from 'react';
import { Copy, Trash2, X } from 'lucide-react';
import { useSystemStore } from '../store/useSystemStore';

/** Hardware readiness modal — GC + cache purge + copyable diagnostic JSON. */
export function SystemDiagnosticsModal() {
  const { isDiagnosticsOpen, setDiagnosticsOpen, health, triggerGarbageCollection, clearDiskCache } =
    useSystemStore();
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  if (!isDiagnosticsOpen) return null;

  const diagnosticJson = JSON.stringify(
    {
      health,
      apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:30211',
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString(),
    },
    null,
    2,
  );

  const runGc = async () => {
    setBusy('gc');
    setNotice(null);
    try {
      await triggerGarbageCollection();
      setNotice('Garbage collection completed.');
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const purge = async () => {
    setBusy('purge');
    setNotice(null);
    try {
      const freed = await clearDiskCache();
      setNotice(`Purged .cache/ — reclaimed ${freed.toFixed(2)} MB.`);
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.55)' }}
      onClick={() => setDiagnosticsOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="System diagnostics"
    >
      <div
        className="w-full max-w-2xl rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold">System Diagnostics & Hardware Readiness</h2>
          <button
            onClick={() => setDiagnosticsOpen(false)}
            className="rounded-md p-1.5"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
            aria-label="Close (Esc)"
          >
            <X size={16} />
          </button>
        </div>

        <dl
          className="font-mono-tabular grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2"
          style={{ color: 'var(--text-secondary)' }}
        >
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>OS Platform: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health?.os_platform ?? '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>Python: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health?.python_version ?? '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>CPU Cores: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health?.cpu_logical_cores ?? '—'} (OMP={health?.omp_threads ?? '—'})</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>WebGL2 GPU: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health?.webgl2_renderer ?? '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>Process RAM: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health ? `${health.memory.process_rss_mb.toFixed(0)} / ${health.memory.process_budget_mb.toFixed(0)} MB (${health.memory.process_usage_percent.toFixed(1)}%)` : '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>System Free: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health ? `${health.memory.system_available_mb.toFixed(0)} / ${health.memory.system_total_mb.toFixed(0)} MB` : '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>Disk Cache: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health ? `${health.cache_size_mb.toFixed(2)} MB (.cache/)` : '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>Backend Latency: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{health ? `${health.latency_ms} ms` : '—'}</dd></div>
        </dl>

        {notice && (
          <p className="mt-3 rounded-md px-3 py-2 text-xs" style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}>
            {notice}
          </p>
        )}

        <div className="mt-4 flex flex-wrap gap-2">
          <button
            onClick={() => void runGc()}
            disabled={busy !== null}
            className="rounded-md px-3 py-2 text-xs font-medium disabled:opacity-50"
            style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            {busy === 'gc' ? 'Running GC…' : 'Force Python GC'}
          </button>
          <button
            onClick={() => void purge()}
            disabled={busy !== null}
            className="flex items-center gap-1.5 rounded-md px-3 py-2 text-xs font-medium disabled:opacity-50"
            style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            <Trash2 size={13} /> {busy === 'purge' ? 'Purging…' : 'Purge .cache/ Directory'}
          </button>
          <button
            onClick={() => void navigator.clipboard.writeText(diagnosticJson).then(() => setNotice('Diagnostic JSON copied.')).catch(() => setNotice('Copy failed.'))}
            className="flex items-center gap-1.5 rounded-md px-3 py-2 text-xs font-medium"
            style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            <Copy size={13} /> Copy Diagnostic JSON
          </button>
        </div>

        <p className="mt-4 text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
          Keyboard shortcuts
        </p>
        <dl className="mt-1 grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2" style={{ color: 'var(--text-secondary)' }}>
          <div><dt className="font-mono-tabular inline">Ctrl/⌘ K: </dt><dd className="inline">command search</dd></div>
          <div><dt className="font-mono-tabular inline">Alt + 1…5: </dt><dd className="inline">jump to step</dd></div>
          <div><dt className="font-mono-tabular inline">Tab: </dt><dd className="inline">cinema mode on/off</dd></div>
          <div><dt className="font-mono-tabular inline">Space / ← →: </dt><dd className="inline">play / step frame</dd></div>
          <div><dt className="font-mono-tabular inline">Ctrl + Enter: </dt><dd className="inline">confirm mapping</dd></div>
          <div><dt className="font-mono-tabular inline">Shift + ?: </dt><dd className="inline">open this panel</dd></div>
          <div><dt className="font-mono-tabular inline">Ctrl + O / Ctrl + M: </dt><dd className="inline">browse file / model</dd></div>
          <div><dt className="font-mono-tabular inline">Ctrl + Shift + D: </dt><dd className="inline">also opens this panel</dd></div>
          <div><dt className="font-mono-tabular inline">Esc: </dt><dd className="inline">close dialog</dd></div>
        </dl>
        <p className="mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Tip: Tab doubles as fullscreen — it never hijacks focus inside text fields or open dialogs.
        </p>
      </div>
    </div>
  );
}

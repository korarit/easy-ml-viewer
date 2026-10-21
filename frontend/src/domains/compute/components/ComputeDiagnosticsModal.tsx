import { useState } from 'react';
import { Trash2, X } from 'lucide-react';
import { useComputeStatusStore } from '../store/useComputeStatusStore';
import {
  COMPUTE_CHUNK_SIZE_LABELS,
  COMPUTE_DEVICE_MODE_LABELS,
  type ComputeChunkSize,
  type ComputeDeviceMode,
} from '../types/compute.types';

const DEVICE_MODES: ComputeDeviceMode[] = ['auto_smart_router', 'force_cpu_only', 'prefer_gpu'];
const CHUNK_SIZES: ComputeChunkSize[] = [2500, 5000, 10000];

/** Phase 17 — compute engine diagnostics: device router, threads, chunks, purge. */
export function ComputeDiagnosticsModal() {
  const {
    isDiagnosticsOpen,
    setDiagnosticsOpen,
    telemetry,
    deviceMode,
    chunkSize,
    workerThreads,
    activeJob,
    lastError,
    setDeviceMode,
    setChunkSize,
    setWorkerThreads,
    purgeCaches,
    cancelActiveJob,
  } = useComputeStatusStore();
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  if (!isDiagnosticsOpen) return null;

  const maxThreads = Math.max(1, telemetry?.physical_cores ?? 8);
  const threads = workerThreads || telemetry?.optimal_workers || maxThreads - 1;

  const run = async (key: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(key);
    setNotice(null);
    try {
      const res = await fn();
      setNotice(typeof res === 'number' ? `Purged — reclaimed ${res.toFixed(2)} MB.` : ok);
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
      aria-label="Compute engine and hardware diagnostics"
    >
      <div
        className="max-h-[90vh] w-full max-w-3xl overflow-auto rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold">⚙️ Compute Engine & Hardware Diagnostics</h2>
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
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>System CPU: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{telemetry ? `${telemetry.physical_cores} Physical / ${telemetry.logical_cores} Logical` : '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>GPU Providers: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{telemetry ? telemetry.available_onnx_providers.join(', ') : '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>Process Memory: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{telemetry ? `${telemetry.process_rss_mb.toFixed(0)} MB RSS (Available: ${telemetry.system_available_mb.toFixed(0)} MB)` : '—'}</dd></div>
          <div><dt className="inline" style={{ color: 'var(--text-muted)' }}>Guard Level: </dt><dd className="inline" style={{ color: 'var(--text-primary)' }}>{telemetry?.memory_guard_level ?? '—'}</dd></div>
        </dl>

        <div className="mt-4">
          <p className="mb-1.5 text-xs font-medium">Execution Mode</p>
          <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Execution device preference">
            {DEVICE_MODES.map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={deviceMode === m}
                onClick={() => void run('mode', () => setDeviceMode(m), `Device mode → ${COMPUTE_DEVICE_MODE_LABELS[m]}`)}
                disabled={busy !== null}
                className="rounded-md px-3 py-1.5 text-xs"
                style={{
                  border: `1px solid ${deviceMode === m ? 'var(--ml-primary)' : 'var(--border-subtle)'}`,
                  color: 'var(--text-primary)',
                  opacity: busy !== null ? 0.6 : 1,
                }}
              >
                {deviceMode === m ? '● ' : '○ '}{COMPUTE_DEVICE_MODE_LABELS[m]}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="compute-threads" className="mb-1.5 block text-xs font-medium">
              Worker Threads: {threads} (reserved 1 core for UI & OS)
            </label>
            <input
              id="compute-threads"
              type="range"
              min={1}
              max={maxThreads}
              value={Math.min(threads, maxThreads)}
              onChange={(e) => setWorkerThreads(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label htmlFor="compute-chunk" className="mb-1.5 block text-xs font-medium">
              Batch Chunk Size
            </label>
            <select
              id="compute-chunk"
              value={chunkSize}
              onChange={(e) => void run('chunk', () => setChunkSize(Number(e.target.value) as ComputeChunkSize), 'Chunk size updated.')}
              disabled={busy !== null}
              className="w-full rounded-md px-2 py-1.5 text-xs"
              style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
            >
              {CHUNK_SIZES.map((c) => (
                <option key={c} value={c}>{COMPUTE_CHUNK_SIZE_LABELS[c]}</option>
              ))}
            </select>
          </div>
        </div>

        {activeJob && (
          <div className="mt-4 rounded-md px-3 py-2 text-xs" style={{ background: 'var(--surface-elevated)' }}>
            Active Job: {activeJob.percent.toFixed(0)}% (Chunk {Math.min(activeJob.chunksDone + 1, activeJob.chunksTotal)}/{activeJob.chunksTotal} · {activeJob.processedRows.toLocaleString()}/{activeJob.totalRows.toLocaleString()} rows)
            {activeJob.status === 'running' && (
              <button
                onClick={() => void run('cancel', () => cancelActiveJob(), 'Job cancelled — memory released.')}
                disabled={busy !== null}
                className="ml-3 rounded px-2 py-1"
                style={{ border: '1px solid var(--border-strong)' }}
              >
                {busy === 'cancel' ? 'Cancelling…' : '⏹ Cancel'}
              </button>
            )}
          </div>
        )}

        {notice && (
          <p className="mt-3 rounded-md px-3 py-2 text-xs" style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}>
            {notice}
          </p>
        )}
        {lastError && (
          <p className="mt-2 text-xs" style={{ color: 'var(--ml-class-1)' }}>{lastError}</p>
        )}

        <div className="mt-4 flex flex-wrap gap-2">
          <button
            onClick={() => void run('purge', () => purgeCaches(), 'Caches purged.')}
            disabled={busy !== null}
            className="flex items-center gap-1.5 rounded-md px-3 py-2 text-xs font-medium disabled:opacity-50"
            style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
          >
            <Trash2 size={13} /> {busy === 'purge' ? 'Purging…' : '🧹 Purge Inactive Caches'}
          </button>
        </div>
      </div>
    </div>
  );
}

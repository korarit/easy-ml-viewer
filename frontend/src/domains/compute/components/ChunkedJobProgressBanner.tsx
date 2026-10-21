import { useEffect } from 'react';
import { OctagonX, X } from 'lucide-react';
import { useComputeStatusStore } from '../store/useComputeStatusStore';

/** Phase 17 — chunked job progress + cancel banner (shows for jobs > 5,000 rows). */
export function ChunkedJobProgressBanner() {
  const { activeJob, cancelActiveJob, dismissActiveJob } = useComputeStatusStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && activeJob?.status === 'running') void cancelActiveJob();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [activeJob?.status, cancelActiveJob]);

  if (!activeJob) return null;

  const chunkFrom = Math.min(activeJob.processedRows + 1, activeJob.totalRows);
  const chunkTo = Math.min(
    activeJob.processedRows + activeJob.chunkSize,
    activeJob.totalRows,
  );
  const done = activeJob.status !== 'running';

  return (
    <div
      className="flex flex-wrap items-center gap-3 border-t px-4 py-2 text-xs"
      style={{
        background: 'var(--surface-elevated)',
        borderColor: 'var(--border-subtle)',
        color: 'var(--text-primary)',
      }}
      role="status"
      aria-live="polite"
    >
      <span className="font-medium">
        {done
          ? activeJob.status === 'cancelled'
            ? 'Job cancelled — memory released.'
            : `Job complete — ${activeJob.totalRows.toLocaleString()} rows in ${activeJob.chunksTotal} chunks.`
          : `Processing Chunk ${Math.min(activeJob.chunksDone + 1, activeJob.chunksTotal)} / ${activeJob.chunksTotal} (Rows ${chunkFrom.toLocaleString()} .. ${chunkTo.toLocaleString()} of ${activeJob.totalRows.toLocaleString()}) — ${activeJob.percent.toFixed(1)}%`}
      </span>
      {!done && (
        <>
          <span className="inline-block h-2 w-48 overflow-hidden rounded-full" style={{ background: 'var(--surface-panel)' }}>
            <span
              className="block h-full rounded-full"
              style={{ width: `${Math.min(100, activeJob.percent)}%`, background: 'var(--ml-primary)' }}
            />
          </span>
          <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)' }}>
            Speed: {activeJob.rowsPerSecond.toLocaleString()} rows/s · ETA: {activeJob.etaSeconds.toFixed(1)}s
          </span>
          <button
            onClick={() => void cancelActiveJob()}
            className="flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-medium"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
            title="Cancel job (Esc)"
          >
            <OctagonX size={13} /> ⏹ Cancel Job (Esc)
          </button>
        </>
      )}
      {done && (
        <button
          onClick={() => dismissActiveJob()}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs"
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
          aria-label="Dismiss job banner"
        >
          <X size={13} /> Dismiss
        </button>
      )}
    </div>
  );
}

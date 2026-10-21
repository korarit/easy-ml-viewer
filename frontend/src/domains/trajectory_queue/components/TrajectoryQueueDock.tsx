import { ArrowLeft, ArrowRight, Play, X } from 'lucide-react';

interface Props {
  selectedRowIndices: number[];
  truncatedNotice: string | null;
  isLaunchDisabled: boolean;
  onRemove: (origRowIndex: number) => void;
  onMove: (fromIdx: number, toIdx: number) => void;
  onClear: () => void;
  onBack: () => void;
  onLaunch: () => void;
}

/** Bottom filmstrip dock previewing Frame 0..T-1 playback order (Phase 06 §2.3). */
export function TrajectoryQueueDock({
  selectedRowIndices,
  truncatedNotice,
  isLaunchDisabled,
  onRemove,
  onMove,
  onClear,
  onBack,
  onLaunch,
}: Props) {
  const seconds = (selectedRowIndices.length / 10).toFixed(1);
  const preview = selectedRowIndices.slice(0, 24);

  return (
    <div
      className="flex flex-col gap-2 rounded-md px-3 py-2"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium">
          Active Trajectory Queue:{' '}
          <span className="font-mono-tabular" style={{ color: 'var(--ml-primary)' }}>
            {selectedRowIndices.length.toLocaleString()} frames
          </span>{' '}
          <span style={{ color: 'var(--text-muted)' }}>(est. playback @ 1x: {seconds}s)</span>
        </p>
        {selectedRowIndices.length > 0 && (
          <button onClick={onClear} className="text-xs underline" style={{ color: 'var(--text-secondary)' }}>
            Clear queue
          </button>
        )}
      </div>

      {truncatedNotice && (
        <p className="text-[11px]" style={{ color: 'var(--ml-anomaly)' }} role="status">
          {truncatedNotice}
        </p>
      )}

      {selectedRowIndices.length === 0 ? (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Queue is empty — tick rows in the table or use a preset above.
        </p>
      ) : (
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Trajectory queue filmstrip">
          {preview.map((rowIdx, frameIdx) => (
            <span
              key={`${frameIdx}-${rowIdx}`}
              className="font-mono-tabular flex items-center gap-1 rounded px-1.5 py-1 text-[11px]"
              style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
              title={`Frame ${frameIdx} = CSV Row #${rowIdx}`}
            >
              <span style={{ color: 'var(--text-muted)' }}>F{frameIdx}:</span> #{rowIdx}
              <button
                onClick={() => onRemove(rowIdx)}
                aria-label={`Remove frame ${frameIdx} (row ${rowIdx})`}
                className="rounded px-0.5"
                style={{ color: 'var(--text-muted)' }}
              >
                <X size={11} />
              </button>
              {frameIdx > 0 && (
                <button
                  onClick={() => onMove(frameIdx, frameIdx - 1)}
                  aria-label={`Move frame ${frameIdx} earlier`}
                  className="rounded px-0.5"
                  style={{ color: 'var(--text-muted)' }}
                >
                  ‹
                </button>
              )}
              {frameIdx < selectedRowIndices.length - 1 && (
                <button
                  onClick={() => onMove(frameIdx, frameIdx + 1)}
                  aria-label={`Move frame ${frameIdx} later`}
                  className="rounded px-0.5"
                  style={{ color: 'var(--text-muted)' }}
                >
                  ›
                </button>
              )}
            </span>
          ))}
          {selectedRowIndices.length > preview.length && (
            <span className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
              +{(selectedRowIndices.length - preview.length).toLocaleString()} more…
            </span>
          )}
        </div>
      )}

      <div className="flex justify-between gap-2">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          <ArrowLeft size={14} /> Back to Mapping
        </button>
        <button
          onClick={onLaunch}
          disabled={isLaunchDisabled}
          title={isLaunchDisabled ? 'Select at least 1 row first' : `Launch 3D simulation with ${selectedRowIndices.length} frames`}
          className="flex items-center gap-1.5 rounded-md px-4 py-2 text-xs font-medium disabled:opacity-40"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          <Play size={14} /> Launch 3D Simulation Studio ({selectedRowIndices.length}) <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}

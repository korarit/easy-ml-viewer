interface Props {
  outlierCount: number;
  queuedCount: number;
  onSliceOod: () => void;
  disabled: boolean;
}

/** OOD / statistical outlier slicer — works even without a target column (§2.4). */
export function OutlierSlicerCard({ outlierCount, queuedCount, onSliceOod, disabled }: Props) {
  return (
    <div
      className="flex flex-col gap-2 rounded-md p-3"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <p className="text-xs font-medium">Out-of-Distribution (OOD) & Outlier Slicer</p>
      <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {outlierCount.toLocaleString()} rows with extreme feature values (|z-score| &gt; 3.0)
        {queuedCount > 0 && ` · ${queuedCount} queued`}
      </p>
      <button
        onClick={onSliceOod}
        disabled={disabled || outlierCount === 0}
        title="Queue rows whose features fall far outside the normal range"
        className="rounded-md px-3 py-2 text-left text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40"
        style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
      >
        ⚠️ Queue Extreme Feature Outliers (|z| &gt; 3.0)
      </button>
    </div>
  );
}

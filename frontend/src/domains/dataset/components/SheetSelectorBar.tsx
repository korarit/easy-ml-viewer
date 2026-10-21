import { useDatasetStore } from '../store/useDatasetStore';

/** Appears automatically when an .xlsx workbook exposes more than one sheet. */
export function SheetSelectorBar() {
  const { currentDataset, uploadDataset, isUploading, lastFile: pendingFile } = useDatasetStore();
  const sheets = currentDataset?.available_sheets ?? [];
  if (sheets.length <= 1) return null;

  return (
    <div
      className="mt-3 flex flex-wrap items-center gap-2 rounded-md px-3 py-2 text-xs"
      style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
    >
      <span style={{ color: 'var(--text-secondary)' }}>
        Workbook has {sheets.length} sheets · active: <strong>{currentDataset?.selected_sheet}</strong>
      </span>
      <select
        aria-label="Select Excel sheet"
        disabled={isUploading || !pendingFile}
        defaultValue={currentDataset?.selected_sheet ?? ''}
        onChange={(e) => {
          if (pendingFile && e.target.value) void uploadDataset(pendingFile, e.target.value).catch(() => undefined);
        }}
        className="rounded-md px-2 py-1 text-xs"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        title={pendingFile ? 'Re-read workbook with this sheet' : 'Re-upload the .xlsx to switch sheets'}
      >
        {sheets.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      {!pendingFile && (
        <span style={{ color: 'var(--text-muted)' }}>Re-drop the .xlsx file to switch sheets.</span>
      )}
    </div>
  );
}

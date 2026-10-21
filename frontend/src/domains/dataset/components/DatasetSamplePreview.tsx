import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useDatasetStore } from '../store/useDatasetStore';

/** Collapsible head-10 raw preview — Thai text / decimals sanity check. */
export function DatasetSamplePreview() {
  const { currentDataset } = useDatasetStore();
  const [open, setOpen] = useState(false);
  if (!currentDataset || currentDataset.sample_rows.length === 0) return null;
  const headers = Object.keys(currentDataset.sample_rows[0] ?? {});

  return (
    <div className="rounded-lg" style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-3 py-2 text-xs font-medium"
        aria-expanded={open}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Show raw sample rows (first {currentDataset.sample_rows.length})
      </button>
      {open && (
        <div className="overflow-auto border-t px-3 py-2" style={{ borderColor: 'var(--border-subtle)' }}>
          <table className="font-mono-tabular w-full text-[11px]">
            <thead>
              <tr style={{ color: 'var(--text-muted)' }}>
                {headers.map((h) => (
                  <th key={h} className="px-2 py-1 text-left font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody style={{ color: 'var(--text-secondary)' }}>
              {currentDataset.sample_rows.map((row, i) => (
                <tr key={i} className="border-t" style={{ borderColor: 'var(--border-subtle)' }}>
                  {headers.map((h) => (
                    <td key={h} className="max-w-48 truncate px-2 py-1">
                      {row[h] === null || row[h] === undefined ? <span style={{ color: 'var(--text-muted)' }}>∅</span> : String(row[h])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

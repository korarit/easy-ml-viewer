import { useMemo, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import type { QueueRow } from '../store/useTrajectoryQueueStore';

interface Props {
  rows: QueueRow[];
  columns: string[];
  columnBadges: Record<string, string>;
  selectedLookupSet: Set<number>;
  frameIndexByRow: Map<number, number>;
  onToggleRow: (origRowIndex: number, shiftKey: boolean) => void;
  isLoading: boolean;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    return Number.isInteger(value) ? String(value) : value.toPrecision(6).replace(/\.?0+$/, (m) => (m.startsWith('.') ? '' : m));
  }
  return String(value);
}

/**
 * Virtualized data grid — renders only the visible window (overscan 8),
 * so `<tr>` count in the DOM never exceeds ~40 rows (Phase 06 DoD).
 */
export function VirtualizedDatasetTable({
  rows,
  columns,
  columnBadges,
  selectedLookupSet,
  frameIndexByRow,
  onToggleRow,
  isLoading,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 33,
    overscan: 8,
  });

  const virtualItems = virtualizer.getVirtualItems();
  const totalHeight = virtualizer.getTotalSize();

  const colWidth = useMemo(() => (columns.length > 6 ? 130 : 150), [columns.length]);

  return (
    <div
      className="rounded-md"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div
        ref={scrollRef}
        data-testid="trajectory-scroll"
        style={{ height: 420, overflow: 'auto', position: 'relative' }}
      >
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 2 }}>
            <tr style={{ background: 'var(--surface-elevated)' }}>
              <th style={{ width: 92, padding: '6px 8px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600 }}>
                Select
              </th>
              <th style={{ width: 110, padding: '6px 8px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600 }}>
                Orig Row #
              </th>
              {columns.map((c) => (
                <th
                  key={c}
                  style={{ minWidth: colWidth, padding: '6px 8px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600 }}
                >
                  {columnBadges[c] && (
                    <span
                      className="font-mono-tabular mr-1 rounded px-1"
                      style={{ background: 'rgba(34,211,238,0.15)', color: 'var(--ml-primary)', fontSize: 10 }}
                    >
                      {columnBadges[c]}
                    </span>
                  )}
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody style={{ position: 'relative', height: totalHeight }}>
            {virtualItems.map((vi) => {
              const row = rows[vi.index];
              if (!row) return null;
              const orig = row._original_row_index;
              const checked = selectedLookupSet.has(orig);
              const frame = frameIndexByRow.get(orig);
              return (
                <tr
                  key={orig}
                  data-index={vi.index}
                  ref={virtualizer.measureElement}
                  onClick={(e) => onToggleRow(orig, e.shiftKey)}
                  title={checked ? `Queued as Frame #${frame} — click to remove` : 'Click to queue · Shift+Click for range'}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: vi.size,
                    transform: `translateY(${vi.start}px)`,
                    background: checked ? 'rgba(34,211,238,0.10)' : vi.index % 2 ? 'transparent' : 'rgba(255,255,255,0.015)',
                    borderBottom: '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                  }}
                >
                  <td style={{ padding: '4px 8px', whiteSpace: 'nowrap' }}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        e.stopPropagation();
                        onToggleRow(orig, e.nativeEvent instanceof MouseEvent && (e.nativeEvent as MouseEvent).shiftKey);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      aria-label={`Select row ${orig}`}
                    />{' '}
                    {frame !== undefined && (
                      <span
                        className="font-mono-tabular rounded px-1"
                        style={{ background: 'var(--ml-primary)', color: '#fff', fontSize: 10 }}
                      >
                        F{frame}
                      </span>
                    )}
                  </td>
                  <td className="font-mono-tabular" style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>
                    #{orig} <span style={{ color: 'var(--text-muted)' }}>· L{orig + 2}</span>
                  </td>
                  {columns.map((c) => (
                    <td key={c} className="font-mono-tabular" style={{ padding: '4px 8px' }}>
                      {formatCell(row[c])}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length === 0 && !isLoading && (
          <p className="p-4 text-xs" style={{ color: 'var(--text-muted)' }}>
            No rows match the current filters.
          </p>
        )}
        {isLoading && (
          <p className="font-mono-tabular p-2 text-[11px]" style={{ color: 'var(--ml-primary)' }}>
            Loading window…
          </p>
        )}
      </div>
    </div>
  );
}

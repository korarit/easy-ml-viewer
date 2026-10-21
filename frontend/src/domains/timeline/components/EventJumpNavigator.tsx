import { ChevronsLeft, ChevronsRight } from 'lucide-react';
import { useTimelineStore } from '../store/useTimelineStore';
import { JUMP_FILTER_LABEL, type JumpFilter } from '../types/timeline.types';

/**
 * Phase 11 — event filter checkboxes with live pin counts.
 * `[🔴 Errors (12)] [⚡ Flips (6)] [🟡 Boundary (19)] [🟣 Outliers (4)] [🔖 Notes (2)]`
 */
export function TimelineFilterBar({ rowIndices }: { rowIndices: number[] }) {
  const data = useTimelineStore((s) => s.data);
  const visible = useTimelineStore((s) => s.visible);
  const showBookmarks = useTimelineStore((s) => s.showBookmarks);

  const counts = {
    misclassified: 0,
    decision_flip: 0,
    low_confidence: 0,
    input_outlier: 0,
  };
  for (const p of data?.autoEvents ?? []) counts[p.event_type] += 1;
  const noteCount = (data?.bookmarks ?? []).filter(
    (b) => rowIndices.indexOf(b.dataset_row_index) !== -1 || b.frame_index >= 0,
  ).length;

  const items = [
    { key: 'misclassified' as const, glyph: '🔴', label: 'Errors', count: counts.misclassified, on: visible.misclassified },
    { key: 'decision_flip' as const, glyph: '⚡', label: 'Flips', count: counts.decision_flip, on: visible.decision_flip },
    { key: 'low_confidence' as const, glyph: '🟡', label: 'Boundary', count: counts.low_confidence, on: visible.low_confidence },
    { key: 'input_outlier' as const, glyph: '🟣', label: 'Outliers', count: counts.input_outlier, on: visible.input_outlier },
  ];

  return (
    <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Timeline event filters">
      {items.map((it) => (
        <label
          key={it.key}
          className="flex cursor-pointer items-center gap-1 rounded px-1.5 py-0.5 text-[11px]"
          style={{
            border: '1px solid var(--border-subtle)',
            color: it.on ? 'var(--text-primary)' : 'var(--text-muted)',
            background: it.on ? 'rgba(125,211,252,0.08)' : 'transparent',
            opacity: it.on ? 1 : 0.6,
          }}
        >
          <input
            type="checkbox"
            className="sr-only"
            checked={it.on}
            onChange={() => useTimelineStore.getState().toggleEventType(it.key)}
            aria-label={`Show ${it.label} pins`}
          />
          <span aria-hidden>{it.on ? '✓' : '○'}</span>
          <span aria-hidden>{it.glyph}</span> {it.label} ({it.count})
        </label>
      ))}
      <label
        className="flex cursor-pointer items-center gap-1 rounded px-1.5 py-0.5 text-[11px]"
        style={{
          border: '1px solid var(--border-subtle)',
          color: showBookmarks ? 'var(--text-primary)' : 'var(--text-muted)',
          background: showBookmarks ? 'rgba(245,158,11,0.10)' : 'transparent',
          opacity: showBookmarks ? 1 : 0.6,
        }}
      >
        <input
          type="checkbox"
          className="sr-only"
          checked={showBookmarks}
          onChange={() => useTimelineStore.getState().toggleBookmarks()}
          aria-label="Show bookmark pins"
        />
        <span aria-hidden>{showBookmarks ? '✓' : '○'}</span>
        <span aria-hidden>🔖</span> Notes ({noteCount})
      </label>
    </div>
  );
}

/**
 * Phase 11 — jump-to-event navigation bar.
 * `[|< Prev Event (,)] [Next Event (.) >|]` + type dropdown + wrap-around toast.
 */
export function EventJumpNavigator({ rowIndices }: { rowIndices: number[] }) {
  const jumpFilter = useTimelineStore((s) => s.jumpFilter);
  const wrapNoticeAt = useTimelineStore((s) => s.wrapNoticeAt);
  const data = useTimelineStore((s) => s.data);

  const showToast = wrapNoticeAt > 0 && Date.now() - wrapNoticeAt < 2500;

  return (
    <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Jump to event">
      <button
        type="button"
        aria-label="Previous event (,)"
        title="Previous event (,)"
        disabled={!data || data.totalFrames <= 0}
        onClick={() => useTimelineStore.getState().jumpToPrev(rowIndices)}
        className="flex items-center gap-1 rounded-md px-2 py-1 text-xs disabled:opacity-40"
        style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
      >
        <ChevronsLeft size={14} /> Prev Event
      </button>
      <button
        type="button"
        aria-label="Next event (.)"
        title="Next event (.)"
        disabled={!data || data.totalFrames <= 0}
        onClick={() => useTimelineStore.getState().jumpToNext(rowIndices)}
        className="flex items-center gap-1 rounded-md px-2 py-1 text-xs disabled:opacity-40"
        style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
      >
        Next Event <ChevronsRight size={14} />
      </button>
      <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
        Jump
        <select
          aria-label="Jump filter"
          value={jumpFilter}
          onChange={(e) => useTimelineStore.getState().setJumpFilter(e.target.value as JumpFilter)}
          className="font-mono-tabular rounded px-1.5 py-1"
          style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)' }}
        >
          {(Object.keys(JUMP_FILTER_LABEL) as JumpFilter[]).map((f) => (
            <option key={f} value={f}>
              {JUMP_FILTER_LABEL[f]}
            </option>
          ))}
        </select>
      </label>
      {showToast && (
        <span
          className="rounded px-2 py-0.5 text-[11px]"
          style={{ background: 'rgba(125,211,252,0.14)', color: '#7dd3fc' }}
          role="status"
        >
          วนกลับไปยังหมุดแรกของคิว
        </span>
      )}
    </div>
  );
}

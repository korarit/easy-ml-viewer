import { useMemo, useState } from 'react';
import { Bookmark, ChevronDown, Pencil, Trash2 } from 'lucide-react';
import { useTimelineStore, resolveBookmarkFrame } from '../store/useTimelineStore';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { BOOKMARK_COLOR_HEX } from '../types/timeline.types';

/**
 * Phase 11 — bookmark + auto-event drawer list (spec §2.4).
 * Searchable cards; click jumps to the frame (edge #6: orphaned rows
 * show "not in current queue" instead of a stale frame).
 */
export function BookmarkListSidebar({ rowIndices }: { rowIndices: number[] }) {
  const data = useTimelineStore((s) => s.data);
  const selectedBookmarkId = useTimelineStore((s) => s.selectedBookmarkId);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');

  const cards = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = (data?.bookmarks ?? []).map((b) => ({
      key: b.bookmark_id,
      frame: resolveBookmarkFrame(b, rowIndices),
      title: b.note_text || `Bookmark @ row ${b.dataset_row_index}`,
      detail: `Row #${b.dataset_row_index} · ${b.color} · ${b.created_at_iso.slice(0, 10)}`,
      color: BOOKMARK_COLOR_HEX[b.color],
      bookmarkId: b.bookmark_id as string,
      kind: 'bookmark' as const,
      raw: b,
    }));
    const auto = (data?.autoEvents ?? [])
      .filter((p) => !q || `${p.title} ${p.detail}`.toLowerCase().includes(q))
      .slice(0, 200)
      .map((p, i) => ({
        key: `auto-${p.frame_index}-${p.event_type}-${i}`,
        frame: p.frame_index,
        title: p.title,
        detail: p.detail,
        color: '#f43e5e',
        bookmarkId: null as string | null,
        kind: 'auto' as const,
        raw: null,
      }));
    const notes = q
      ? list.filter(
          (c) => c.title.toLowerCase().includes(q) || c.detail.toLowerCase().includes(q),
        )
      : list;
    return [...notes, ...auto];
  }, [data, query, rowIndices]);

  const noteCount = data?.bookmarks.length ?? 0;

  return (
    <div className="rounded-md" style={{ border: '1px solid var(--border-subtle)' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 px-2 py-1.5 text-xs"
        style={{ color: 'var(--text-primary)' }}
      >
        <Bookmark size={13} style={{ color: '#f59e0b' }} />
        Notes & Events ({noteCount} notes)
        <ChevronDown
          size={13}
          style={{ transform: open ? 'rotate(180deg)' : undefined, color: 'var(--text-muted)' }}
        />
      </button>
      {open && (
        <div className="flex max-h-56 flex-col gap-1 overflow-y-auto px-2 pb-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.stopPropagation()}
            placeholder="Search notes & events…"
            aria-label="Search notes and events"
            className="rounded px-2 py-1 text-xs"
            style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
          />
          {cards.length === 0 && (
            <p className="py-2 text-center text-[11px]" style={{ color: 'var(--text-muted)' }}>
              No notes yet — press B on any frame.
            </p>
          )}
          {cards.map((c) => (
            <div
              key={c.key}
              className="flex items-center gap-2 rounded px-2 py-1 text-left text-[11px]"
              style={{
                background: c.bookmarkId === selectedBookmarkId ? 'rgba(125,211,252,0.10)' : 'rgba(148,163,184,0.06)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: c.color }} />
              <button
                type="button"
                className="min-w-0 flex-1"
                title={c.frame !== null ? `Jump to frame ${c.frame}` : 'Row not in current queue'}
                onClick={() => {
                  if (c.frame === null) return;
                  if (c.bookmarkId) useTimelineStore.getState().setSelectedBookmark(c.bookmarkId);
                  usePlaybackStore.getState().seekToFrame(c.frame, true);
                }}
              >
                <span className="block truncate" style={{ color: 'var(--text-primary)' }}>
                  {c.frame !== null ? `#${c.frame} · ` : '(not in queue) · '}{c.title}
                </span>
                <span className="block truncate" style={{ color: 'var(--text-muted)' }}>{c.detail}</span>
              </button>
              {c.kind === 'bookmark' && c.raw && (
                <>
                  <button
                    type="button"
                    aria-label={`Edit note ${c.raw.bookmark_id}`}
                    title="Edit note"
                    onClick={() => {
                      const f = resolveBookmarkFrame(c.raw!, rowIndices);
                      useTimelineStore.getState().openEditor(f ?? c.raw!.frame_index, c.raw!);
                    }}
                    className="shrink-0 rounded p-1"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    type="button"
                    aria-label={`Delete note ${c.raw.bookmark_id}`}
                    title="Delete note"
                    onClick={() => void useTimelineStore.getState().deleteBookmark(c.raw!.bookmark_id)}
                    className="shrink-0 rounded p-1"
                    style={{ color: '#f43e5e' }}
                  >
                    <Trash2 size={12} />
                  </button>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

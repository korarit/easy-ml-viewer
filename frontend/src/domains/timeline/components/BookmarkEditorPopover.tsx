import { useEffect, useState } from 'react';
import { Bookmark, Trash2, X } from 'lucide-react';
import { useTimelineStore } from '../store/useTimelineStore';
import { BOOKMARK_COLOR_HEX, type BookmarkColor } from '../types/timeline.types';

const COLORS: BookmarkColor[] = ['amber', 'cyan', 'emerald', 'rose', 'purple'];

/**
 * Phase 11 — bookmark + note editor popover (`B`).
 * Note capped at 280 chars; color swatches; Save / Delete / Cancel.
 */
export function BookmarkEditorPopover() {
  const editorOpen = useTimelineStore((s) => s.editorOpen);
  const editorFrame = useTimelineStore((s) => s.editorFrame);
  const editingBookmarkId = useTimelineStore((s) => s.editingBookmarkId);
  const data = useTimelineStore((s) => s.data);

  const existing = editingBookmarkId
    ? (data?.bookmarks.find((b) => b.bookmark_id === editingBookmarkId) ?? null)
    : null;

  const [note, setNote] = useState('');
  const [color, setColor] = useState<BookmarkColor>('amber');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (editorOpen) {
      setNote(existing?.note_text ?? '');
      setColor(existing?.color ?? 'amber');
      setSaving(false);
    }
  }, [editorOpen, editingBookmarkId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!editorOpen || editorFrame === null) return null;

  const close = () => useTimelineStore.getState().closeEditor();
  const save = async () => {
    if (saving) return;
    setSaving(true);
    try {
      await useTimelineStore.getState().saveBookmark(note.trim(), color);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      role="dialog"
      aria-modal="true"
      aria-label={existing ? 'Edit bookmark' : 'Add bookmark'}
      style={{ background: 'rgba(4,8,14,0.6)' }}
      onClick={close}
    >
      <div
        className="w-full max-w-sm rounded-lg p-4"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            <Bookmark size={15} style={{ color: BOOKMARK_COLOR_HEX[color] }} />
            {existing ? 'Edit Bookmark' : 'Bookmark Frame'} #{editorFrame}
          </h3>
          <button
            type="button"
            onClick={close}
            aria-label="Close bookmark editor"
            className="rounded p-1"
            style={{ color: 'var(--text-muted)' }}
          >
            <X size={15} />
          </button>
        </div>
        <label className="mt-3 block text-xs" style={{ color: 'var(--text-secondary)' }}>
          Note (max 280 chars)
          <textarea
            // eslint-disable-next-line jsx-a11y/no-autofocus
            autoFocus
            value={note}
            maxLength={280}
            rows={3}
            onChange={(e) => setNote(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void save();
              e.stopPropagation();
            }}
            placeholder="เช่น ข้อมูลรายได้เป็น 0 แต่ยอดหนี้สูง โมเดลจึงปัดตก"
            className="mt-1 w-full rounded-md px-2 py-1.5 text-xs"
            style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)' }}
          />
        </label>
        <div className="mt-1 text-right font-mono-tabular text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {note.length}/280
        </div>
        <div className="mt-2 flex items-center gap-1.5" role="radiogroup" aria-label="Bookmark color">
          {COLORS.map((c) => (
            <button
              key={c}
              type="button"
              role="radio"
              aria-checked={color === c}
              aria-label={`${c} pin`}
              title={c}
              onClick={() => setColor(c)}
              className="h-6 w-6 rounded-full"
              style={{
                background: BOOKMARK_COLOR_HEX[c],
                outline: color === c ? '2px solid #fff' : '1px solid rgba(255,255,255,0.25)',
                outlineOffset: 1,
              }}
            />
          ))}
        </div>
        <div className="mt-3 flex items-center justify-between gap-2">
          {existing ? (
            <button
              type="button"
              onClick={() => void useTimelineStore.getState().deleteBookmark(existing.bookmark_id)}
              className="flex items-center gap-1 rounded-md px-2.5 py-1.5 text-xs"
              style={{ border: '1px solid #f43e5e', color: '#f43e5e' }}
            >
              <Trash2 size={13} /> Delete
            </button>
          ) : (
            <span />
          )}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={close}
              className="rounded-md px-2.5 py-1.5 text-xs"
              style={{ border: '1px solid var(--border-strong)', color: 'var(--text-secondary)' }}
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void save()}
              disabled={saving}
              className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50"
              style={{ background: 'var(--ml-primary)', color: '#fff' }}
            >
              {saving ? 'Saving…' : existing ? 'Save Changes' : 'Save Bookmark'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

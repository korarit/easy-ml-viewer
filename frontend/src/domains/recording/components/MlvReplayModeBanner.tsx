import { useRecordingStore } from '../store/useRecordingStore';
import { downloadMlvBlob, fetchRebakeBookmarks } from '../api/recordingApi';

/**
 * Phase 15 — replay-mode status banner (spec §2.3).
 * Read-only replay: What-If sliders stay disabled upstream; bookmarks can
 * still be added and re-baked without the original model.
 */
export function MlvReplayModeBanner() {
  const session = useRecordingStore((s) => s.replaySession);
  const isReplay = useRecordingStore((s) => s.isReplayMode);
  const exitReplay = useRecordingStore((s) => s.exitReplay);

  if (!isReplay || !session) return null;
  const h = session.header;

  const onUpdateAndRebake = async () => {
    try {
      const blob = await fetchRebakeBookmarks(session.session_id, h.bookmarks);
      downloadMlvBlob(blob, `${session.session_id}_updated.mlv`);
    } catch (err) {
      useRecordingStore
        .getState()
        .setLoadError(err instanceof Error ? err.message : 'Re-bake failed');
    }
  };

  return (
    <div
      className="flex flex-wrap items-center gap-3 rounded-md px-3 py-2 text-xs"
      style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.45)' }}
      role="status"
      aria-label="Replay mode banner"
    >
      <span className="font-semibold" style={{ color: '#a5b4fc' }}>
        📼 REPLAY MODE: {h.title} (v{h.schema_version} · {h.total_frames.toLocaleString()} Frames)
      </span>
      <span className="hidden sm:inline" style={{ color: 'var(--text-secondary)' }}>
        {h.model_type} · {h.bookmarks.length} bookmarks · {h.auto_events.length} auto-events
      </span>
      <span className="ml-auto flex items-center gap-2">
        <button
          title={`Model: ${h.model_type} · Baked: ${h.created_at_iso}`}
          className="rounded px-2 py-1"
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
          onClick={() =>
            window.alert(
              `Recording Info\nTitle: ${h.title}\nModel: ${h.model_type} (${h.model_family})\nBaked: ${h.created_at_iso}\nFrames: ${h.total_frames}\nFeatures: ${h.feature_names.join(', ')}`,
            )
          }
        >
          ℹ️ Recording Info
        </button>
        <button
          onClick={() => void onUpdateAndRebake()}
          className="rounded px-2 py-1 font-medium"
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}
          title="Update bookmarks and re-bake .mlv without the model"
        >
          ↻ Update & Re-bake .mlv
        </button>
        <button
          onClick={exitReplay}
          className="rounded px-2 py-1"
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
        >
          Exit Replay
        </button>
      </span>
    </div>
  );
}

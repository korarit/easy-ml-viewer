import { useStudioLayoutStore } from '../store/useStudioLayoutStore';

/**
 * Phase 14 — cinema presentation floating control pill.
 * Rendered only in cinema mode; exits on `Tab` / `Esc` (see shortcuts hook).
 */
export function CinemaModeOverlay() {
  const cinema = useStudioLayoutStore((s) => s.cinemaMode);
  const setCinema = useStudioLayoutStore((s) => s.setCinema);
  const setPalette = useStudioLayoutStore((s) => s.setCommandPalette);
  if (!cinema) return null;
  return (
    <div className="pointer-events-none fixed bottom-5 left-1/2 z-[65] -translate-x-1/2">
      <div
        className="pointer-events-auto flex items-center gap-2 rounded-full px-4 py-2 text-xs"
        style={{
          background: 'rgba(10,15,22,0.55)',
          border: '1px solid rgba(148,163,184,0.3)',
          backdropFilter: 'blur(14px)',
          WebkitBackdropFilter: 'blur(14px)',
          color: '#e2e8f0',
          boxShadow: '0 12px 32px rgba(0,0,0,0.4)',
        }}
        role="toolbar"
        aria-label="Cinema mode controls"
      >
        <span aria-hidden>🎬</span>
        <span className="font-medium">Cinema Mode</span>
        <button
          onClick={() => setPalette(true)}
          className="rounded-full px-2.5 py-1"
          style={{ border: '1px solid rgba(148,163,184,0.35)' }}
          title="Open command palette (Ctrl+K)"
        >
          ⌘K Commands
        </button>
        <button
          onClick={() => setCinema(false)}
          className="rounded-full px-2.5 py-1 font-medium"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
          title="Exit cinema mode (Tab / Esc)"
        >
          Exit Cinema (Tab / Esc)
        </button>
      </div>
    </div>
  );
}


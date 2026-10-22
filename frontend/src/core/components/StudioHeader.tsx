import { useEffect, useRef, useState } from 'react';
import { Check, Clapperboard, EllipsisVertical, FileDown, GitCompareArrows, LayoutGrid, Lock, MonitorCog, Moon, Package, Search, Sun, Tags } from 'lucide-react';
import { useSystemStore, type WorkflowStep } from '../store/useSystemStore';
import { useStudioLayoutStore } from '../../domains/studio-ux/store/useStudioLayoutStore';
import { useRecordingStore } from '../../domains/recording/store/useRecordingStore';
import { useDualCompareStore } from '../../domains/recording/store/useDualCompareStore';
import { DemoGalleryModal } from '../../domains/recording/components/DemoGalleryModal';
import { useExportStore } from '../../domains/export-studio/store/useExportStore';

const STEPS: { id: WorkflowStep; label: string }[] = [
  { id: 1, label: 'Dataset' },
  { id: 2, label: 'Model' },
  { id: 3, label: 'Mapping' },
  { id: 4, label: 'Queue' },
  { id: 5, label: '3D Studio' },
];

export function StudioHeader() {
  const { activeStep, unlockedMaxStep, health, setActiveStep, setDiagnosticsOpen } = useSystemStore();
  const setPalette = useStudioLayoutStore((s) => s.setCommandPalette);
  const cinemaMode = useStudioLayoutStore((s) => s.cinemaMode);
  const toggleCinema = useStudioLayoutStore((s) => s.toggleCinema);
  const setAliasModal = useStudioLayoutStore((s) => s.setAliasModal);
  const setBakeModal = useRecordingStore((s) => s.setBakeModal);
  const setCompareOpen = useDualCompareStore((s) => s.setCompareOpen);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const [theme, setTheme] = useState(() =>
    typeof document !== 'undefined' ? (document.documentElement.dataset.theme ?? 'dark') : 'dark',
  );
  const moreRef = useRef<HTMLDivElement | null>(null);
  const setExportModal = useExportStore((s) => s.setModal);
  const isReplayMode = useRecordingStore((s) => s.isReplayMode);
  const online = health?.status === 'healthy' || health?.status === 'degraded';

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('emv-theme', next);
    } catch {
      /* private mode — ignore */
    }
    setTheme(next);
  };

  // Close the overflow menu on outside click / Esc.
  useEffect(() => {
    if (!moreOpen) return;
    const onDown = (e: PointerEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) setMoreOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMoreOpen(false);
    };
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [moreOpen]);

  const iconBtn =
    'flex min-h-[36px] min-w-[36px] items-center justify-center rounded-md p-2 transition-colors';
  const primaryBtn =
    'flex min-h-[36px] items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors';

  return (
    <header
      className="flex items-center justify-between gap-4 border-b px-4 py-2"
      style={{ background: 'var(--surface-panel)', borderColor: 'var(--border-subtle)' }}
    >
      <div className="flex items-center gap-3">
        <div
          className="flex h-8 w-8 items-center justify-center rounded-md text-sm font-bold"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
          aria-hidden
        >
          EM
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">Easy ML Viewer</div>
          <div
            className="flex items-center gap-1.5 text-[11px]"
            title={
              online
                ? `Connected: http://127.0.0.1:30211 · ${health?.latency_ms ?? 0} ms · ${health?.os_platform ?? ''}`
                : 'Backend offline — start it on port 30211'
            }
          >
            {/* System health uses neutral blue/gray — red/green are reserved for data classes. */}
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: online ? 'var(--ml-primary)' : 'var(--text-muted)' }}
            />
            <span style={{ color: 'var(--text-secondary)' }}>
              {online ? '100% OFFLINE STUDIO' : 'BACKEND OFFLINE'}
            </span>
          </div>
        </div>
      </div>

      <nav className="hidden items-center gap-1 lg:flex" aria-label="Workflow steps">
        <span className="mr-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Step {activeStep}/5
        </span>
        {STEPS.map((s, i) => {
          const locked = s.id > unlockedMaxStep;
          const active = s.id === activeStep;
          const done = s.id < activeStep;
          return (
            <div key={s.id} className="flex items-center gap-1">
              {i > 0 && (
                <span className="mx-0.5 text-xs" style={{ color: 'var(--text-muted)' }} aria-hidden>
                  ›
                </span>
              )}
              <button
                disabled={locked}
                onClick={() => setActiveStep(s.id)}
                title={locked ? 'Locked — complete previous steps first' : `Go to ${s.label}`}
                aria-current={active ? 'step' : undefined}
                className="flex min-h-[36px] items-center gap-1 rounded-md px-2.5 py-1.5 text-xs font-medium"
                style={{
                  background: active ? 'var(--ml-primary)' : 'transparent',
                  border: active ? '1px solid var(--ml-primary)' : '1px solid transparent',
                  color: active ? '#fff' : locked ? 'var(--text-muted)' : 'var(--text-secondary)',
                  opacity: locked ? 0.6 : 1,
                  cursor: locked ? 'not-allowed' : 'pointer',
                }}
              >
                {locked ? (
                  <Lock size={12} aria-hidden />
                ) : done ? (
                  <Check size={12} aria-hidden />
                ) : null}
                {s.id}. {s.label}
              </button>
            </div>
          );
        })}
      </nav>

      <div className="flex items-center gap-2">
        {/* Primary: the 3 actions that move work forward. Everything else lives in ⋯. */}
        <button
          title="Try a ready-made demo in one click (offline)"
          onClick={() => setGalleryOpen(true)}
          className={primaryBtn}
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          <LayoutGrid size={14} />
          Try Demo
        </button>
        <button
          title="Search commands, jump to row, change axes (Ctrl+K)"
          onClick={() => setPalette(true)}
          className={`${primaryBtn} hidden md:flex`}
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', minWidth: 200 }}
        >
          <Search size={14} />
          <span className="flex-1 text-left">Search…</span>
          <kbd className="font-mono-tabular text-[10px]" style={{ color: 'var(--text-muted)' }}>⌘K</kbd>
        </button>
        <button
          title="Export table, 3D video or report (Ctrl+E)"
          onClick={() => setExportModal(true)}
          className={`${primaryBtn} hidden md:flex`}
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
        >
          <FileDown size={14} />
          Export
        </button>
        <button
          title={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
          onClick={toggleTheme}
          aria-label={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
          className={iconBtn}
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
        >
          {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
        </button>
        <button
          title="Toggle cinema fullscreen mode (Tab)"
          onClick={() => toggleCinema()}
          aria-pressed={cinemaMode}
          className={iconBtn}
          style={{
            border: '1px solid var(--border-subtle)',
            color: cinemaMode ? '#fff' : 'var(--text-secondary)',
            background: cinemaMode ? 'var(--ml-primary)' : 'transparent',
          }}
        >
          <Clapperboard size={16} />
        </button>
        <button
          title="System diagnostics & shortcuts (Ctrl+Shift+D, Shift+?)"
          onClick={() => setDiagnosticsOpen(true)}
          className={iconBtn}
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
        >
          <MonitorCog size={16} />
        </button>
        <div className="relative" ref={moreRef}>
          <button
            title="More studio actions"
            onClick={() => setMoreOpen((v) => !v)}
            aria-expanded={moreOpen}
            aria-haspopup="menu"
            aria-label="More studio actions"
            className={iconBtn}
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
          >
            <EllipsisVertical size={16} />
          </button>
          {moreOpen && (
            <div
              role="menu"
              className="absolute right-0 z-50 mt-2 w-60 rounded-lg p-1.5"
              style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)' }}
            >
              {[
                {
                  icon: <GitCompareArrows size={14} />,
                  label: 'Compare 2 models (.mlv)',
                  hint: 'Alt+D finds conflicts',
                  fn: () => setCompareOpen(true),
                },
                {
                  icon: <Package size={14} />,
                  label: isReplayMode ? 'Replay mode: baking locked' : 'Bake .mlv recording',
                  hint: 'MsgPack + Zstd',
                  fn: () => setBakeModal(true),
                },
                {
                  icon: <Tags size={14} />,
                  label: 'Edit class & feature aliases',
                  hint: '',
                  fn: () => setAliasModal(true),
                },
              ].map((a) => (
                <button
                  key={a.label}
                  role="menuitem"
                  onClick={() => {
                    a.fn();
                    setMoreOpen(false);
                  }}
                  className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-xs"
                  style={{ color: 'var(--text-primary)' }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'var(--surface-elevated)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <span style={{ color: 'var(--text-secondary)' }}>{a.icon}</span>
                  <span className="flex-1">{a.label}</span>
                  {a.hint && (
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {a.hint}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <DemoGalleryModal open={galleryOpen} onClose={() => setGalleryOpen(false)} />
    </header>
  );
}

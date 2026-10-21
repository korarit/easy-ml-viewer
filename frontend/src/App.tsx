import { useCallback, useEffect, useState } from 'react';
import { StudioHeader } from './core/components/StudioHeader';
import { SystemStatusBar } from './core/components/SystemStatusBar';
import { SystemDiagnosticsModal } from './core/components/SystemDiagnosticsModal';
import { GlobalErrorBoundary } from './core/components/GlobalErrorBoundary';
import { useSystemStore } from './core/store/useSystemStore';
import { API_BASE_URL } from './core/api/client';
import { DatasetStepView } from './domains/dataset/DatasetStepView';
import { ModelStepView } from './domains/model_engine/ModelStepView';

const STEP_COPY: Record<number, { title: string; body: string }> = {
  1: { title: 'Step 1 · Dataset', body: 'CSV / XLSX / Parquet ingestion lands here in Phase 02.' },
  2: { title: 'Step 2 · Model', body: 'Model Registry + schema introspection lands here in Phase 03.' },
  3: { title: 'Step 3 · Column Mapping', body: 'Fuzzy auto-matcher UI lands here in Phase 05.' },
  4: { title: 'Step 4 · Trajectory Queue', body: 'Virtualized table + queue builder lands here in Phase 06.' },
  5: { title: 'Step 5 · 3D Stage & VCR', body: 'WebGL stage + 60 FPS VCR lands here in Phases 08–11.' },
};

function useHealthPoller() {
  const refreshHealth = useSystemStore((s) => s.refreshHealth);
  useEffect(() => {
    void refreshHealth();
    const id = window.setInterval(() => {
      if (document.visibilityState !== 'hidden') void refreshHealth();
    }, 10_000);
    return () => window.clearInterval(id);
  }, [refreshHealth]);
}

function useShortcuts() {
  const { setDiagnosticsOpen, setActiveStep, unlockedMaxStep } = useSystemStore();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDiagnosticsOpen(false);
      if (e.shiftKey && (e.key === '?' || e.key === '/')) setDiagnosticsOpen(true);
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        setDiagnosticsOpen(true);
      }
      if (e.altKey && ['1', '2', '3', '4', '5'].includes(e.key)) {
        const step = Number(e.key) as 1 | 2 | 3 | 4 | 5;
        if (step <= unlockedMaxStep) setActiveStep(step);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setDiagnosticsOpen, setActiveStep, unlockedMaxStep]);
}

function OfflineBanner() {
  const { health, refreshHealth } = useSystemStore();
  const [retrying, setRetrying] = useState(false);
  if (health && health.status !== 'offline') return null;
  const retry = async () => {
    setRetrying(true);
    try {
      await refreshHealth();
    } finally {
      setRetrying(false);
    }
  };
  return (
    <div
      className="flex flex-wrap items-center gap-3 border-b px-4 py-2 text-xs"
      style={{ background: 'var(--surface-elevated)', borderColor: 'var(--border-subtle)', color: 'var(--text-primary)' }}
      role="alert"
    >
      <span style={{ color: 'var(--ml-class-1)' }}>● Backend offline.</span>
      <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)' }}>
        Start it with: uvicorn app.main:app --port 30211 (in backend/)
      </span>
      <button
        onClick={() => void retry()}
        disabled={retrying}
        className="font-mono-tabular rounded px-2 py-1 disabled:opacity-50"
        style={{ border: '1px solid var(--border-strong)' }}
      >
        {retrying ? 'Retrying…' : 'Copy command'.replace('Copy command', 'Retry now')}
      </button>
      <button
        onClick={() => void navigator.clipboard.writeText('uvicorn app.main:app --port 30211')}
        className="font-mono-tabular rounded px-2 py-1"
        style={{ border: '1px solid var(--border-strong)' }}
      >
        Copy command
      </button>
    </div>
  );
}

export default function App() {
  useHealthPoller();
  useShortcuts();
  const { activeStep, health, unlockStep } = useSystemStore();

  const toggleTheme = useCallback(() => {
    const el = document.documentElement;
    el.dataset.theme = el.dataset.theme === 'light' ? 'dark' : 'light';
  }, []);

  const copy = STEP_COPY[activeStep];
  const lowMem = health?.memory.is_low_memory ?? false;

  return (
    <GlobalErrorBoundary>
      <div className="flex h-full flex-col" style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)' }}>
        <StudioHeader />
        <OfflineBanner />
        {lowMem && (
          <div className="px-4 py-1.5 text-xs" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }} role="alert">
            Low system memory — close other apps before loading large datasets. Open diagnostics (Ctrl+Shift+D) for Force GC.
          </div>
        )}

        <main className="flex min-h-0 flex-1 items-stretch justify-center overflow-auto">
          {activeStep === 1 ? (
            <DatasetStepView />
          ) : activeStep === 2 ? (
            <ModelStepView />
          ) : (
          <section
            className="m-6 w-full max-w-xl self-center rounded-lg p-6"
            style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
          >
            <p className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
              Phase 01 · App shell ready
            </p>
            <h1 className="mt-1 text-lg font-semibold">{copy.title}</h1>
            <p className="mt-1 text-sm" style={{ color: 'var(--text-secondary)' }}>
              {copy.body}
            </p>
            <p className="font-mono-tabular mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
              API: {API_BASE_URL} · Theme: {typeof document !== 'undefined' ? document.documentElement.dataset.theme : 'dark'}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={() => unlockStep(1)}
                className="rounded-md px-3 py-2 text-xs font-medium"
                style={{ background: 'var(--ml-primary)', color: '#fff' }}
              >
                Start New Session
              </button>
              <button
                onClick={toggleTheme}
                className="rounded-md px-3 py-2 text-xs font-medium"
                style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
              >
                Toggle Dark / Light
              </button>
            </div>
            {health && (
              <dl className="font-mono-tabular mt-4 grid grid-cols-2 gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                <div>OS: {health.os_platform}</div>
                <div>Python: {health.python_version}</div>
                <div>RAM: {health.memory.process_rss_mb.toFixed(0)}/{health.memory.process_budget_mb.toFixed(0)} MB</div>
                <div>Cache: {health.cache_size_mb.toFixed(2)} MB</div>
              </dl>
            )}
          </section>
          )}
        </main>

        <SystemStatusBar />
        <SystemDiagnosticsModal />
      </div>
    </GlobalErrorBoundary>
  );
}

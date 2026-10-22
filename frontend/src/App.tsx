import { useEffect, useState } from 'react';
import { StudioHeader } from './core/components/StudioHeader';
import { SystemStatusBar } from './core/components/SystemStatusBar';
import { SystemDiagnosticsModal } from './core/components/SystemDiagnosticsModal';
import { ComputeDiagnosticsModal } from './domains/compute/components/ComputeDiagnosticsModal';
import { ChunkedJobProgressBanner } from './domains/compute/components/ChunkedJobProgressBanner';
import { GlobalErrorBoundary } from './core/components/GlobalErrorBoundary';
import { useSystemStore } from './core/store/useSystemStore';
import { API_BASE_URL } from './core/api/client';
import { DatasetStepView } from './domains/dataset/DatasetStepView';
import { ModelStepView } from './domains/model_engine/ModelStepView';
import { MappingStepView } from './domains/mapping/MappingStepView';
import { TrajectoryQueueStepView } from './domains/trajectory_queue/TrajectoryQueueStepView';
import { Stage3DStepView } from './domains/stage3d/Stage3DStepView';
import { useStudioKeyboardShortcuts } from './domains/studio-ux/hooks/useStudioKeyboardShortcuts';

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

const BACKEND_START_COMMAND = 'cd backend; python -m uvicorn app.main:app --host 127.0.0.1 --port 30211';

function OfflineBanner() {
  const { health, refreshHealth } = useSystemStore();
  const [retrying, setRetrying] = useState(false);
  const [copied, setCopied] = useState(false);
  if (health && health.status !== 'offline') return null;
  const retry = async () => {
    setRetrying(true);
    try {
      await refreshHealth();
    } finally {
      setRetrying(false);
    }
  };
  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(BACKEND_START_COMMAND);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
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
        Backend is offline. Start it with:
      </span>
      <code
        className="font-mono-tabular cursor-pointer rounded px-2 py-0.5"
        style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
        onClick={() => void copyCommand()}
        title="Click to copy"
      >
        {BACKEND_START_COMMAND}
      </code>
      <button
        onClick={() => void retry()}
        disabled={retrying}
        className="font-mono-tabular min-h-[32px] rounded px-2 py-1 disabled:opacity-50"
        style={{ border: '1px solid var(--border-strong)' }}
      >
        {retrying ? 'Retrying…' : 'Retry now'}
      </button>
      <button
        onClick={() => void copyCommand()}
        className="font-mono-tabular min-h-[32px] rounded px-2 py-1"
        style={{ border: '1px solid var(--border-strong)' }}
      >
        {copied ? 'Copied ✓' : 'Copy command'}
      </button>
    </div>
  );
}

export default function App() {
  useHealthPoller();
  useShortcuts();
  useStudioKeyboardShortcuts(true);
  const { activeStep, health, unlockStep } = useSystemStore();

  const lowMem = health?.memory.is_low_memory ?? false;

  return (
    <GlobalErrorBoundary>
      <div className="flex h-full flex-col" style={{ background: 'var(--bg-canvas)', color: 'var(--text-primary)' }}>
        <a
          href="#studio-main"
          className="sr-only focus:not-sr-only focus:absolute focus:z-[60] focus:rounded focus:px-3 focus:py-2 focus:text-xs"
          style={{ background: 'var(--ml-primary)', color: '#fff' }}
        >
          Skip to studio content
        </a>
        <StudioHeader />
        <OfflineBanner />
        {lowMem && (
          <div className="px-4 py-1.5 text-xs" style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }} role="alert">
            Low system memory — close other apps before loading large datasets. Open diagnostics (Ctrl+Shift+D) for Force GC.
          </div>
        )}

        {/* Step 5 owns its viewport (no page scroll): canvas + dock + transport share 100% height. */}
        <main
          id="studio-main"
          className={`flex min-h-0 flex-1 items-stretch justify-center ${activeStep === 5 ? 'overflow-hidden' : 'overflow-auto'}`}
        >
          {activeStep === 1 ? (
            <DatasetStepView />
          ) : activeStep === 2 ? (
            <ModelStepView />
          ) : activeStep === 3 ? (
            <MappingStepView />
          ) : activeStep === 4 ? (
            <TrajectoryQueueStepView />
          ) : (
            <Stage3DStepView />
          )}
        </main>

        <ChunkedJobProgressBanner />
        <SystemStatusBar />
        <SystemDiagnosticsModal />
        <ComputeDiagnosticsModal />
      </div>
    </GlobalErrorBoundary>
  );
}

import { Activity, Keyboard, MonitorCog, Trash2 } from 'lucide-react';
import { useSystemStore, type WorkflowStep } from '../store/useSystemStore';

const STEPS: { id: WorkflowStep; label: string }[] = [
  { id: 1, label: 'Dataset' },
  { id: 2, label: 'Model' },
  { id: 3, label: 'Mapping' },
  { id: 4, label: 'Queue' },
  { id: 5, label: '3D Studio' },
];

export function StudioHeader() {
  const { activeStep, unlockedMaxStep, health, setActiveStep, setDiagnosticsOpen } = useSystemStore();
  const online = health?.status === 'healthy' || health?.status === 'degraded';

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
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: online ? 'var(--ml-class-0)' : 'var(--ml-class-1)' }}
            />
            <span style={{ color: 'var(--text-secondary)' }}>
              {online ? '100% OFFLINE STUDIO' : 'BACKEND OFFLINE'}
            </span>
          </div>
        </div>
      </div>

      <nav className="hidden items-center gap-1 lg:flex" aria-label="Workflow steps">
        {STEPS.map((s, i) => {
          const locked = s.id > unlockedMaxStep;
          const active = s.id === activeStep;
          const done = s.id < activeStep;
          return (
            <div key={s.id} className="flex items-center gap-1">
              {i > 0 && (
                <span className="mx-1 text-xs" style={{ color: 'var(--text-muted)' }}>
                  →
                </span>
              )}
              <button
                disabled={locked}
                onClick={() => setActiveStep(s.id)}
                title={locked ? 'Locked — complete previous steps first' : `Go to ${s.label}`}
                className="rounded-md px-2.5 py-1.5 text-xs font-medium"
                style={{
                  background: active ? 'transparent' : 'transparent',
                  border: active ? '1px solid var(--ml-primary)' : '1px solid transparent',
                  color: locked ? 'var(--text-muted)' : active ? 'var(--text-primary)' : 'var(--text-secondary)',
                  opacity: locked ? 0.5 : 1,
                  cursor: locked ? 'not-allowed' : 'pointer',
                }}
              >
                {done ? '✓ ' : ''}
                {s.id}. {s.label}
              </button>
            </div>
          );
        })}
      </nav>

      <div className="flex items-center gap-2">
        <button
          title="System diagnostics (Ctrl+Shift+D)"
          onClick={() => setDiagnosticsOpen(true)}
          className="rounded-md p-2"
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
        >
          <MonitorCog size={16} />
        </button>
        <button
          title="Keyboard shortcuts (Shift+?)"
          onClick={() => setDiagnosticsOpen(true)}
          className="rounded-md p-2"
          style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
        >
          <Keyboard size={16} />
        </button>
        <span className="hidden text-[11px] xl:inline" style={{ color: 'var(--text-muted)' }}>
          <Activity size={12} className="mr-1 inline" />
          Phase 01 · App Shell
        </span>
        <span className="hidden" aria-hidden>
          <Trash2 size={12} />
        </span>
      </div>
    </header>
  );
}

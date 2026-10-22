import { useEffect } from 'react';
import { Cpu } from 'lucide-react';
import { useComputeStatusStore } from '../store/useComputeStatusStore';

const GUARD_COLORS: Record<string, string> = {
  normal: 'var(--ml-class-0)',
  elevated: 'var(--ml-anomaly)',
  guard_active: 'var(--ml-class-1)',
};

const GUARD_LABELS: Record<string, string> = {
  normal: 'Normal',
  elevated: 'Elevated',
  guard_active: 'Memory Guard Active',
};

/** Phase 17 — compact compute status pill for the bottom status bar. */
export function HardwareStatusPill() {
  const { telemetry, lastError, refreshTelemetry, setDiagnosticsOpen } = useComputeStatusStore();

  useEffect(() => {
    void refreshTelemetry();
    const id = window.setInterval(() => {
      if (document.visibilityState !== 'hidden') void refreshTelemetry();
    }, 10_000);
    return () => window.clearInterval(id);
  }, [refreshTelemetry]);

  if (!telemetry) {
    return (
      <span className="flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }} title={lastError ?? 'Probing compute engine…'}>
        <Cpu size={12} />
        Compute: probing…
      </span>
    );
  }

  const gpu = telemetry.available_onnx_providers.find((p) => p !== 'CPUExecutionProvider');
  const gpuLabel = gpu ? gpu.replace('ExecutionProvider', '') : 'CPU-only';
  const dot = GUARD_COLORS[telemetry.memory_guard_level] ?? GUARD_COLORS.normal;

  return (
    <button
      onClick={() => setDiagnosticsOpen(true)}
      title="Open compute details — cores, memory guard, GPU provider"
      className="flex min-h-[28px] items-center gap-1.5 rounded-full px-2 py-0.5"
      style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
    >
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: dot }} />
      <Cpu size={12} />
      <span className="font-mono-tabular">
        Compute: {GUARD_LABELS[telemetry.memory_guard_level] ?? 'Normal'} · {gpuLabel}
        {telemetry.active_jobs_count > 0 ? ` · ${telemetry.active_jobs_count} jobs` : ''}
      </span>
    </button>
  );
}

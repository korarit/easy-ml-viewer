import { TriangleAlert } from 'lucide-react';
import { TASK_TYPE_META, VISUAL_3D_LABELS, useModelStore } from '../store/useModelStore';

export function ModelArchitectureCard() {
  const { currentModel } = useModelStore();
  if (!currentModel) return null;
  const task = TASK_TYPE_META[currentModel.task_type] ?? {
    label: currentModel.task_type,
    color: 'var(--text-secondary)',
  };

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          {currentModel.display_name}
        </span>
        <span
          className="rounded px-1.5 py-0.5 font-medium"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
        >
          {currentModel.family_name}
        </span>
        <span className="rounded px-1.5 py-0.5 font-medium" style={{ background: `${task.color}22`, color: task.color }}>
          {task.label}
        </span>
        <span className="font-mono-tabular" style={{ color: 'var(--text-secondary)' }}>
          3D: {VISUAL_3D_LABELS[currentModel.visual_3d_type] ?? currentModel.visual_3d_type}
        </span>
        <span className="flex items-center gap-1 font-medium" style={{ color: 'var(--ml-class-0)' }}>
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: 'var(--ml-class-0)' }} />
          Fitted & Ready ({currentModel.n_features_in} input features)
        </span>
      </div>

      {currentModel.is_pipeline && currentModel.pipeline_steps.length > 0 && (
        <div className="font-mono-tabular mt-2 flex flex-wrap items-center gap-1" aria-label="Pipeline steps">
          {currentModel.pipeline_steps.map((step, i) => (
            <span key={`${step}-${i}`} className="flex items-center gap-1">
              {i > 0 && <span style={{ color: 'var(--text-muted)' }}>→</span>}
              <span
                className="rounded px-1.5 py-0.5"
                style={{
                  background: 'var(--surface-elevated)',
                  border:
                    i === currentModel.pipeline_steps.length - 1
                      ? '1px solid var(--ml-primary)'
                      : '1px solid transparent',
                }}
              >
                {i === 0 ? '' : i === currentModel.pipeline_steps.length - 1 ? 'Final: ' : ''}
                {step}
              </span>
            </span>
          ))}
        </div>
      )}

      {currentModel.warnings.length > 0 && (
        <div
          className="mt-2 flex items-start gap-1.5 rounded-md px-3 py-2"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="alert"
        >
          <TriangleAlert size={14} className="mt-0.5 shrink-0" />
          <div>
            {currentModel.warnings.map((w, i) => (
              <p key={i} className="font-mono-tabular text-[11px]">
                {w}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

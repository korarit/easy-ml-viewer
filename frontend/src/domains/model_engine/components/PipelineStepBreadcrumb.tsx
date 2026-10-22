import { useState } from 'react';
import type { PipelineStepSummary } from '../api/pipelineApi';

interface Props {
  steps: PipelineStepSummary[];
  terminalEstimator: string;
}

/**
 * Phase 37 — Interactive Pipeline Step Breadcrumb Bar.
 * `[Raw CSV (4)] ➔ [preprocessor (4→8)] ➔ [RandomForest]`, click a block to
 * open its Before vs After column table.
 */
export function PipelineStepBreadcrumb({ steps, terminalEstimator }: Props) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  return (
    <div className="flex flex-col gap-1.5" aria-label="Pipeline step breadcrumb">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
        {steps.map((s, i) => (
          <span key={`${s.step_name}-${i}`} className="flex items-center gap-1.5">
            {i > 0 && <span style={{ color: 'var(--text-muted)' }}>➔</span>}
            <button
              onClick={() => setOpenIdx(openIdx === i ? null : i)}
              title={`${s.kind}: ${s.n_in} → ${s.n_out} columns. Click for Before/After columns.`}
              className="font-mono-tabular rounded-md px-2 py-1"
              style={{
                border: '1px solid var(--border-strong)',
                background: openIdx === i ? 'var(--surface-elevated)' : 'transparent',
                color: 'var(--text-primary)',
              }}
            >
              [{s.step_name} ({s.n_in}→{s.n_out})]
            </button>
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          {steps.length > 0 && <span style={{ color: 'var(--text-muted)' }}>➔</span>}
          <span
            className="font-mono-tabular rounded-md px-2 py-1"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
            title={`Delegated terminal estimator: ${terminalEstimator}`}
          >
            [{terminalEstimator}]
          </span>
        </span>
      </div>
      {openIdx !== null && steps[openIdx] && (
        <div
          className="font-mono-tabular grid gap-1 rounded-md p-2 text-[11px] md:grid-cols-2"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
        >
          <div>
            <p style={{ color: 'var(--text-muted)' }}>
              Before [{steps[openIdx].step_name}] ({steps[openIdx].n_in})
            </p>
            <p style={{ color: 'var(--text-secondary)' }}>{steps[openIdx].input_columns.join(', ') || '—'}</p>
          </div>
          <div>
            <p style={{ color: 'var(--text-muted)' }}>
              After [{steps[openIdx].step_name}] ({steps[openIdx].n_out})
            </p>
            <p style={{ color: 'var(--text-secondary)' }}>{steps[openIdx].output_columns.join(', ') || '—'}</p>
          </div>
        </div>
      )}
    </div>
  );
}

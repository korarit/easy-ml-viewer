import { useMemo } from 'react';
import type { PipelineTraceMechanics } from '../api/pipelineApi';
import { topNonZeroEntries } from '../../stage3d/components/surfaces/pipelineSurfaceMath';

interface Props {
  mech: PipelineTraceMechanics;
  waterfallMode: 'grouped' | 'expanded';
}

/**
 * Phase 37 — Step-by-step transformation tracer for the current row.
 * Raw Value ➔ Imputed ➔ Scaled/Encoded with [Imputed] / [OOV] warning badges.
 */
export function PipelineTransformationCard({ mech, waterfallMode }: Props) {
  const top16 = useMemo(
    () => topNonZeroEntries(mech.transformed_row_top16 ?? {}, 16),
    [mech.transformed_row_top16],
  );

  return (
    <div
      className="flex flex-col gap-2 rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      aria-label="Pipeline transformation tracer"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          Pipeline tunnel — row tracer
        </span>
        {mech.has_imputed_values && (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--ml-class-2)', border: '1px solid rgba(245,158,11,0.5)' }}
            title={`Imputed columns: ${mech.imputed_columns.join(', ')}`}
          >
            ◆ Imputed Value{mech.imputed_columns.length > 1 ? 's' : ''}: {mech.imputed_columns.join(', ')}
          </span>
        )}
        {mech.has_oov_category && (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)', border: '1px solid rgba(244,63,94,0.5)' }}
            title={`Unseen categories collapsed to all-zeros: ${mech.oov_columns.join(', ')}`}
          >
            ◆ OOV Category: {mech.oov_columns.join(', ')} → zeros
          </span>
        )}
      </div>

      <ol className="font-mono-tabular flex flex-col gap-0.5 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {(Object.entries(mech.raw_row_values ?? {}) as Array<[string, string | number | null]>).map(([k, v]) => (
          <li key={k}>
            {k}: {v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v)) ? 'NaN [Missing!]' : String(v)}
            {mech.imputed_columns.includes(k) ? ' [Imputed]' : ''}
          </li>
        ))}
      </ol>

      <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
        ▼ {mech.pipeline_step_names.join(' → ')} ({mech.n_transformed_features} transformed cols)
      </p>
      <ol className="font-mono-tabular flex flex-col gap-0.5 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {top16.map(([k, v]) => (
          <li key={k}>
            {k}: {v.toFixed(4)}
          </li>
        ))}
        {top16.length === 0 && <li style={{ color: 'var(--text-muted)' }}>All transformed values are zero.</li>}
      </ol>

      {waterfallMode === 'expanded' && (
        <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Expanded encoded view — {Object.keys(mech.transformed_row_top16 ?? {}).length} active columns (top-16 shown).
        </p>
      )}
    </div>
  );
}

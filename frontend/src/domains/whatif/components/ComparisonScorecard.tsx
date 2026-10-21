import { formatDelta, formatWhatIfValue, type WhatIfEvaluation } from '../types/whatif.types';

interface Props {
  evaluation: WhatIfEvaluation | null;
  isClassification: boolean;
  isEvaluating: boolean;
  lastEvalMs: number | null;
}

function classLabel(pred: number, conf: number, isClassification: boolean): string {
  if (!isClassification) return formatWhatIfValue(pred);
  return `Class ${Math.round(pred)} (Prob: ${conf.toFixed(2)})`;
}

/**
 * Phase 13 — before/after scorecard (spec §2.1).
 * `[Original #42] ➔ Δ +0.34 (🎉 DECISION FLIPPED!) [Simulated]`.
 */
export function ComparisonScorecard({ evaluation, isClassification, isEvaluating, lastEvalMs }: Props) {
  if (!evaluation) {
    return (
      <div data-testid="whatif-scorecard-empty" className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Adjust a slider to simulate — the before/after verdict lands here.
      </div>
    );
  }
  const delta = evaluation.mutated_prediction - evaluation.original_prediction;
  const deltaConf = evaluation.mutated_confidence - evaluation.original_confidence;
  const metric = isClassification ? deltaConf : delta;

  return (
    <div data-testid="whatif-scorecard" className="flex items-stretch gap-1.5">
      <div
        className="flex-1 rounded-md px-2 py-1.5 text-[11px]"
        style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
      >
        <div className="text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
          Original #{evaluation.dataset_row_index}
        </div>
        <div className="font-mono-tabular mt-0.5 font-semibold">
          {classLabel(evaluation.original_prediction, evaluation.original_confidence, isClassification)}
        </div>
      </div>

      <div className="flex flex-col items-center justify-center px-0.5 text-center">
        <span aria-hidden>➔</span>
        <span className="font-mono-tabular text-[10px]" style={{ color: metric >= 0 ? '#34d399' : '#fb7185' }}>
          {formatDelta(metric, 0).replace(/ \(.*/, '')}
        </span>
        {evaluation.decision_flipped && (
          <span
            data-testid="whatif-flipped-badge"
            className="mt-0.5 rounded px-1 py-px text-[10px] font-bold"
            style={{ background: 'rgba(52,211,153,0.15)', color: '#34d399' }}
          >
            🎉 DECISION FLIPPED!
          </span>
        )}
      </div>

      <div
        className="flex-1 rounded-md px-2 py-1.5 text-[11px]"
        style={{
          background: evaluation.decision_flipped ? 'rgba(52,211,153,0.08)' : 'var(--surface-elevated)',
          border: `1px solid ${evaluation.decision_flipped ? '#34d399' : 'var(--border-subtle)'}`,
        }}
      >
        <div className="text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
          What-If Simulated{isEvaluating ? '…' : ''}
        </div>
        <div className="font-mono-tabular mt-0.5 font-semibold">
          {classLabel(evaluation.mutated_prediction, evaluation.mutated_confidence, isClassification)}
        </div>
      </div>

      {lastEvalMs !== null && (
        <div
          className="font-mono-tabular self-center text-[10px]"
          style={{ color: 'var(--text-muted)' }}
          title="Last backend evaluate round-trip"
        >
          {lastEvalMs.toFixed(1)}ms
        </div>
      )}
    </div>
  );
}

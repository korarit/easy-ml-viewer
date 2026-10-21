import type { GlobalEvaluationSummary } from '../api/evaluationApi';

function r2Tone(r2: number): string {
  if (r2 >= 0.85) return 'var(--ml-success)';
  if (r2 >= 0.6) return 'var(--ml-anomaly)';
  return 'var(--ml-class-1)';
}

function Card({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: string }) {
  return (
    <div
      className="min-w-[120px] flex-1 rounded-md px-3 py-2"
      style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
    >
      <p className="text-[10px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
        {label}
      </p>
      <p className="font-mono-tabular text-base font-semibold" style={{ color: accent ?? 'var(--text-primary)' }}>
        {value}
      </p>
      {sub && (
        <p className="font-mono-tabular text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {sub}
        </p>
      )}
    </div>
  );
}

/** KPI ribbon: regression / classification / unlabeled (§2.1). */
export function GlobalMetricsRibbon({ summary }: { summary: GlobalEvaluationSummary }) {
  const reg = summary.regression;
  const cls = summary.classification;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap gap-2" role="status" aria-label="Global evaluation metrics">
        {reg && (
          <>
            <Card label="R² Score" value={reg.r2_score.toFixed(4)} sub={reg.r2_score >= 0.85 ? 'Excellent' : reg.r2_score >= 0.6 ? 'Fair' : 'Poor fit'} accent={r2Tone(reg.r2_score)} />
            <Card label="RMSE" value={reg.rmse.toLocaleString()} />
            <Card label="MAE" value={reg.mae.toLocaleString()} />
            <Card label="Max Abs Error" value={reg.max_abs_error.toLocaleString()} />
            <Card label="Evaluated Rows" value={reg.evaluated_rows.toLocaleString()} />
          </>
        )}
        {cls && (
          <>
            <Card label="Accuracy" value={`${(cls.accuracy * 100).toFixed(2)}%`} accent="var(--ml-success)" />
            <Card label="Macro F1" value={cls.macro_f1.toFixed(4)} />
            <Card label="Macro Precision" value={cls.macro_precision.toFixed(4)} />
            <Card label="Macro Recall" value={cls.macro_recall.toFixed(4)} />
            <Card
              label="Correct / Wrong"
              value={`${cls.correct_count.toLocaleString()} / ${cls.misclassified_count.toLocaleString()}`}
              sub={`${cls.evaluated_rows.toLocaleString()} evaluated`}
            />
          </>
        )}
        {!reg && !cls && summary.prediction_distribution && (
          <>
            <Card label="Pred Min" value={summary.prediction_distribution.pred_min.toLocaleString()} />
            <Card label="Pred Max" value={summary.prediction_distribution.pred_max.toLocaleString()} />
            <Card label="Pred Mean" value={summary.prediction_distribution.pred_mean.toLocaleString()} />
            <Card label="Pred SD" value={summary.prediction_distribution.pred_std.toLocaleString()} />
            <Card label="Evaluated Rows" value={summary.evaluated_row_count.toLocaleString()} />
          </>
        )}
      </div>
      {summary.ood_outlier_count > 0 && (
        <p className="text-[11px]" style={{ color: 'var(--ml-anomaly)' }}>
          {summary.ood_outlier_count.toLocaleString()} statistical outlier rows (|z| &gt; 3.0) detected in feature space.
        </p>
      )}
      {reg?.scale_mismatch_warning && (
        <p
          className="rounded-md px-3 py-2 text-[11px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--ml-anomaly)' }}
          role="alert"
        >
          {reg.scale_mismatch_warning}
        </p>
      )}
    </div>
  );
}

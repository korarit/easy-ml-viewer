import type { RegressionReport } from '../api/evaluationApi';
import type { ErrorSliceType } from '../store/useEvaluationStore';

interface Props {
  report: RegressionReport;
  onSlice: (t: ErrorSliceType) => void;
  disabled: boolean;
}

function SliceButton({
  title,
  hint,
  onClick,
  disabled,
}: {
  title: string;
  hint: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={hint}
      className="rounded-md px-3 py-2 text-left text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40"
      style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
    >
      {title}
    </button>
  );
}

/** Residual histogram (pure SVG) + 1-click error auto-queue presets (§2.2). */
export function ResidualSlicerPanel({ report, onSlice, disabled }: Props) {
  const counts = report.residual_hist_counts;
  const max = Math.max(1, ...counts);
  const W = 280;
  const H = 96;
  const n = Math.max(1, counts.length);
  const bw = W / n;

  return (
    <div
      className="flex flex-col gap-2 rounded-md p-3"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <p className="text-xs font-medium">Residual Error Distribution (e = y − ŷ)</p>
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H + 18}`}
        role="img"
        aria-label="Residual error histogram"
        style={{ display: 'block' }}
      >
        {counts.map((c, i) => {
          const h = (c / max) * H;
          const tail = i < 2 || i >= n - 2;
          return (
            <rect
              key={i}
              x={i * bw + 1}
              y={H - h}
              width={Math.max(1, bw - 2)}
              height={h}
              rx={1}
              fill={tail && c > 0 ? 'var(--ml-class-1)' : 'var(--ml-primary)'}
              opacity={c === 0 ? 0.25 : 0.9}
            >
              <title>{`Bin ${i}: ${c} rows`}</title>
            </rect>
          );
        })}
        <line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="var(--text-muted)" strokeDasharray="3 2" strokeWidth={1} />
        <text x={4} y={H + 13} fontSize={9} fill="var(--text-muted)">Under-predicted (ŷ&lt;y)</text>
        <text x={W - 4} y={H + 13} fontSize={9} fill="var(--text-muted)" textAnchor="end">Over-predicted (ŷ&gt;y)</text>
      </svg>
      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        <SliceButton title="🔥 Queue Top 50 Worst Errors (|e|)" hint=" worst absolute residuals first (Alt+E)" onClick={() => onSlice('worst_abs')} disabled={disabled} />
        <SliceButton title="⬆️ Queue Top 50 Over-predicted (ŷ≫y)" hint="model estimated too high" onClick={() => onSlice('over_pred')} disabled={disabled} />
        <SliceButton title="⬇️ Queue Top 50 Under-predicted (ŷ≪y)" hint="model estimated too low" onClick={() => onSlice('under_pred')} disabled={disabled} />
        <SliceButton title="🎯 Queue Top 50 Most Accurate (e≈0)" hint="near-perfect predictions for comparison" onClick={() => onSlice('accurate')} disabled={disabled} />
      </div>
    </div>
  );
}

import type { ClassificationReport } from '../api/evaluationApi';

interface Props {
  report: ClassificationReport;
  onCellClick: (actualIdx: number, predictedIdx: number) => void;
  onSliceUncertain: () => void;
  onSliceMisclassified: () => void;
  disabled: boolean;
}

function tone(actual: number, predicted: number): { bg: string; fg: string } {
  if (actual === predicted) return { bg: 'rgba(16,185,129,0.16)', fg: 'var(--ml-success)' };
  return { bg: 'rgba(244,63,94,0.14)', fg: 'var(--ml-class-1)' };
}

/** Interactive K×K confusion matrix with click-to-queue cells (§2.3). */
export function InteractiveConfusionMatrix({
  report,
  onCellClick,
  onSliceUncertain,
  onSliceMisclassified,
  disabled,
}: Props) {
  const labels = report.class_labels;
  const k = labels.length;
  const byPair = new Map(report.cells.map((c) => [`${c.actual_class_idx}:${c.predicted_class_idx}`, c]));
  const wide = k > 10;

  return (
    <div
      className="flex flex-col gap-2 rounded-md p-3"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium">
          Interactive Confusion Matrix {wide && <span style={{ color: 'var(--text-muted)' }}>(top-15 confused pairs mode)</span>}
        </p>
        <div className="flex gap-1.5">
          <button
            onClick={onSliceMisclassified}
            disabled={disabled || report.misclassified_count === 0}
            className="rounded-md px-2 py-1 text-[11px] disabled:cursor-not-allowed disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)' }}
            title="Queue every misclassified row"
          >
            Queue all {report.misclassified_count.toLocaleString()} errors
          </button>
          <button
            onClick={onSliceUncertain}
            disabled={disabled || report.most_uncertain_indices.length === 0}
            className="rounded-md px-2 py-1 text-[11px] disabled:cursor-not-allowed disabled:opacity-40"
            style={{ border: '1px solid var(--border-strong)' }}
            title="Rows nearest the decision boundary (prob ≈ 0.50)"
          >
            ⚖️ Queue 50 most uncertain
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="border-collapse text-center text-[11px]" aria-label="Confusion matrix">
          <thead>
            <tr>
              <th className="p-1 font-mono-tabular" style={{ color: 'var(--text-muted)' }}>actual ↓ / pred →</th>
              {labels.map((l, j) => (
                <th key={j} className="font-mono-tabular max-w-[110px] truncate p-1" style={{ color: 'var(--text-secondary)' }} title={l}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((actual, i) => (
              <tr key={i}>
                <th className="font-mono-tabular max-w-[110px] truncate p-1 text-left" style={{ color: 'var(--text-secondary)' }} title={actual}>
                  {actual}
                </th>
                {labels.map((_, j) => {
                  const cell = byPair.get(`${i}:${j}`);
                  if (!cell) return <td key={j} className="p-1" style={{ color: 'var(--text-muted)' }}>–</td>;
                  const { bg, fg } = tone(i, j);
                  const empty = cell.row_count === 0;
                  return (
                    <td key={j} className="p-0.5">
                      <button
                        onClick={() => onCellClick(i, j)}
                        disabled={disabled || empty}
                        title={
                          empty
                            ? 'No rows in this cell'
                            : `Queue ${cell.row_count} rows: actual ${cell.actual_label} → predicted ${cell.predicted_label}`
                        }
                        className="font-mono-tabular w-full rounded px-1.5 py-1.5 disabled:cursor-not-allowed disabled:opacity-35"
                        style={{ background: bg, border: '1px solid var(--border-subtle)', color: fg }}
                      >
                        <span className="block text-xs font-semibold">{cell.row_count.toLocaleString()}</span>
                        <span className="block text-[10px] opacity-80">{cell.percentage.toFixed(1)}%</span>
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Click a cell to queue its rows into the 3D trajectory player. Empty cells are not clickable.
      </p>
    </div>
  );
}

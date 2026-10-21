import { useLinearSliceStore } from '../store/useLinearSliceStore';

function fmtSigned(v: number): string {
  const sign = v < 0 ? '−' : '+';
  return `${sign}${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmt(v: number): string {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Phase 04 §2.2 — term-by-term waterfall breakdown for the traced row. */
export function LinearContributionCard() {
  const { traces, featureNames } = useLinearSliceStore();
  const trace = traces[0];
  if (!trace) return null;

  const entries = Object.entries(trace.feature_contributions);
  const denom = entries.reduce((s, [, v]) => s + Math.abs(v), 0) || 1;
  const residual = trace.residual_error;
  const under = residual !== null && residual > 0;

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <p className="mb-2 text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
        Row #{trace.row_index} internal flow breakdown
      </p>
      <table className="font-mono-tabular w-full text-[11px]">
        <thead>
          <tr style={{ color: 'var(--text-muted)' }}>
            <th className="py-1 text-left font-medium">Term / Feature</th>
            <th className="py-1 text-right font-medium">Input (xᵢ)</th>
            <th className="py-1 text-right font-medium">× Weight (wᵢ)</th>
            <th className="py-1 text-right font-medium">= Contribution</th>
            <th className="py-1 text-left font-medium">% Impact</th>
          </tr>
        </thead>
        <tbody style={{ color: 'var(--text-primary)' }}>
          <tr style={{ borderTop: '1px solid var(--border-subtle)' }}>
            <td className="py-1">Base Intercept (b)</td>
            <td className="py-1 text-right">1.00</td>
            <td className="py-1 text-right">{fmtSigned(trace.base_value)}</td>
            <td className="py-1 text-right">{fmtSigned(trace.base_value)}</td>
            <td className="py-1">
              <Bar pct={(Math.abs(trace.base_value) / (denom + Math.abs(trace.base_value))) * 100} />
            </td>
          </tr>
          {entries.map(([name, value]) => {
            const idx = featureNames.indexOf(name);
            const input = trace.extra_mechanics.transformed_inputs?.[idx];
            const weight = trace.extra_mechanics.weights?.[idx];
            const neg = value < 0;
            return (
              <tr key={name} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                <td className="py-1">{name}</td>
                <td className="py-1 text-right">{input !== undefined ? fmt(input) : '—'}</td>
                <td className="py-1 text-right">{weight !== undefined ? fmtSigned(weight) : '—'}</td>
                <td
                  className="py-1 text-right font-semibold"
                  style={{ color: neg ? 'var(--ml-class-1)' : 'var(--ml-class-0)' }}
                >
                  {fmtSigned(value)}
                </td>
                <td className="py-1">
                  <Bar pct={(Math.abs(value) / denom) * 100} negative={neg} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div
        className="font-mono-tabular mt-2 flex flex-wrap gap-x-4 gap-y-1 rounded px-2 py-1.5"
        style={{ background: 'var(--surface-elevated)', color: 'var(--text-primary)' }}
      >
        <span>
          Predicted (ŷ): <strong>{fmt(Number(trace.final_prediction))}</strong>
        </span>
        {trace.actual_target !== null && (
          <span>
            Actual (y): <strong>{fmt(trace.actual_target)}</strong>
          </span>
        )}
        {residual !== null && (
          <span style={{ color: under ? 'var(--ml-class-1)' : 'var(--ml-class-0)' }}>
            Residual (y − ŷ): <strong>{fmtSigned(residual)}</strong>{' '}
            {trace.actual_target ? `(${under ? 'Under' : 'Over'}-predicted ${Math.abs((residual / trace.actual_target) * 100).toFixed(2)}%)` : ''}
          </span>
        )}
      </div>
    </div>
  );
}

function Bar({ pct, negative = false }: { pct: number; negative?: boolean }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <span
      className="ml-2 inline-block h-2 w-24 overflow-hidden rounded align-middle"
      style={{ background: 'var(--surface-elevated)' }}
      title={`${clamped.toFixed(1)}%`}
    >
      <span
        className="block h-full rounded"
        style={{
          width: `${clamped}%`,
          background: negative ? 'var(--ml-class-1)' : 'var(--ml-class-0)',
        }}
      />
    </span>
  );
}

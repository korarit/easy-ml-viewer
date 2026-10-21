import { useLinearSliceStore } from '../store/useLinearSliceStore';

function fmtSigned(v: number): string {
  const sign = v < 0 ? '−' : '+';
  return `${sign}${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
}

function fmt(v: number): string {
  return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
}

/** Phase 04 §2.1 — interactive learned-equation banner with live row substitution. */
export function LinearFormulaBanner() {
  const { featureNames, weights, intercept, rowInput, targetIdx, targetCount, setTargetIdx } =
    useLinearSliceStore();

  if (featureNames.length === 0) return null;

  const live = rowInput
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const hasLive = live.length === featureNames.length && live.every(Number.isFinite);

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          Learned linear equation
        </p>
        {targetCount > 1 && (
          <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            Target output
            <select
              aria-label="Target output"
              value={targetIdx}
              onChange={(e) => setTargetIdx(Number(e.target.value))}
              className="rounded px-1.5 py-0.5"
              style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
            >
              {Array.from({ length: targetCount }, (_, i) => (
                <option key={i} value={i}>
                  Target {i}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      <p className="font-mono-tabular text-[13px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>
        <span>ŷ = </span>
        {featureNames.map((name, i) => {
          const w = weights[i] ?? 0;
          const neg = w < 0;
          return (
            <span key={name}>
              <span style={{ color: neg ? 'var(--ml-class-1)' : 'var(--ml-class-0)' }}>
                ({fmtSigned(w)} × {name}
                {hasLive && (
                  <span style={{ color: 'var(--text-secondary)' }}> = {fmtSigned(w * live[i])}</span>
                )}
                )
              </span>
              <span style={{ color: 'var(--text-muted)' }}> + </span>
            </span>
          );
        })}
        <span style={{ color: 'var(--text-secondary)' }}>
          {fmtSigned(intercept ?? 0)}
          {intercept === 0 && (
            <span className="ml-1 rounded px-1 py-0.5" style={{ background: 'var(--surface-elevated)' }}>
              No Intercept (Origin-Locked: b = 0)
            </span>
          )}
        </span>
      </p>
    </div>
  );
}

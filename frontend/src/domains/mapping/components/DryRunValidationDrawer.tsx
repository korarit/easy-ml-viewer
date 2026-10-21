import { useMappingStore } from '../store/useMappingStore';

export function DryRunValidationDrawer() {
  const { previewMatrixHead, testedRows, outputShape, isValid, featureErrors } = useMappingStore();
  if (previewMatrixHead.length === 0) return null;
  const errs = Object.entries(featureErrors);

  return (
    <section
      className="flex flex-col gap-2 rounded-lg p-3"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      aria-label="Dry-run float32 matrix preview"
    >
      <h2 className="text-xs font-semibold tracking-wide" style={{ color: 'var(--text-primary)' }}>
        DRY-RUN FLOAT32 MATRIX PREVIEW{' '}
        <span className="font-mono-tabular font-normal" style={{ color: isValid ? 'var(--ml-success, #10b981)' : 'var(--ml-class-1)' }}>
          ({testedRows} rows verified {isValid ? 'OK' : '— BLOCKED'} → Shape: ({outputShape?.[0] ?? '?'}, {outputShape?.[1] ?? '?'}), DType: float32)
        </span>
      </h2>
      {errs.length > 0 && (
        <ul className="flex flex-col gap-1 text-[11px]" style={{ color: 'var(--ml-class-1)' }}>
          {errs.map(([idx, msg]) => (
            <li key={idx} role="alert">Feature #{idx}: {msg}</li>
          ))}
        </ul>
      )}
      <div className="overflow-x-auto">
        <table className="font-mono-tabular w-full text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <thead>
            <tr>
              <th className="px-2 py-1 text-left" style={{ color: 'var(--text-muted)' }}>row</th>
              {previewMatrixHead[0]?.map((_, c) => (
                <th key={c} className="px-2 py-1 text-right" style={{ color: 'var(--text-muted)' }}>f{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {previewMatrixHead.map((row, r) => (
              <tr key={r} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                <td className="px-2 py-1">#{r}</td>
                {row.map((v, c) => (
                  <td key={c} className="px-2 py-1 text-right">
                    {v === null ? 'NaN' : Number(v).toLocaleString('en-US', { maximumFractionDigits: 4 })}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

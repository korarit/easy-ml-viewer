import { useMemo, useState } from 'react';

interface Props {
  open: boolean;
  rawLabels: string[];
  expectedClasses: string[];
  hint: string | null;
  isComputing: boolean;
  onConfirm: (labelMap: Record<string, number | string>) => void;
  onClose: () => void;
}

function coerceClassValue(raw: string, expected: string[]): number | string {
  // Prefer the numeric / original class value when the text matches.
  const hit = expected.find((e) => e === raw);
  if (hit !== undefined) {
    const n = Number(hit);
    return Number.isFinite(n) && hit.trim() !== '' ? n : hit;
  }
  const n = Number(raw);
  if (Number.isFinite(n)) return n;
  return raw;
}

/** Label-mismatch modal: map CSV answer labels onto model.classes_ (§2.5). */
export function TargetLabelMapperModal({
  open,
  rawLabels,
  expectedClasses,
  hint,
  isComputing,
  onConfirm,
  onClose,
}: Props) {
  const [draft, setDraft] = useState<Record<string, string>>({});

  const resolved = useMemo(() => {
    const out: Record<string, number | string> = {};
    for (const raw of rawLabels) {
      const picked = draft[raw] ?? '';
      if (picked !== '') out[raw] = coerceClassValue(picked, expectedClasses);
    }
    return out;
  }, [draft, rawLabels, expectedClasses]);

  if (!open) return null;
  const complete = rawLabels.length > 0 && rawLabels.every((r) => (draft[r] ?? '') !== '');

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.55)' }}
      role="dialog"
      aria-modal="true"
      aria-label="Map target labels"
    >
      <div
        className="w-full max-w-md rounded-lg p-4"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-strong)' }}
      >
        <h2 className="text-sm font-semibold">Map Target Labels → Model Classes</h2>
        {hint && (
          <p className="mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            {hint}
          </p>
        )}
        <div className="mt-3 flex max-h-64 flex-col gap-2 overflow-auto">
          {rawLabels.map((raw) => (
            <label key={raw} className="flex items-center justify-between gap-2 text-xs">
              <span className="font-mono-tabular truncate" style={{ color: 'var(--text-primary)' }} title={raw}>
                CSV: “{raw}”
              </span>
              <select
                value={draft[raw] ?? ''}
                onChange={(e) => setDraft((d) => ({ ...d, [raw]: e.target.value }))}
                className="rounded px-2 py-1 text-xs"
                style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
              >
                <option value="">— pick model class —</option>
                {expectedClasses.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-xs"
            style={{ border: '1px solid var(--border-strong)', color: 'var(--text-secondary)' }}
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(resolved)}
            disabled={!complete || isComputing}
            className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-40"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            {isComputing ? 'Evaluating…' : 'Apply Mapping & Re-run'}
          </button>
        </div>
      </div>
    </div>
  );
}

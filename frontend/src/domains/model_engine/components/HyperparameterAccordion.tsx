import { useEffect, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useModelStore } from '../store/useModelStore';

/** Collapsible get_params() inspector — toggled with Alt+H. */
export function HyperparameterAccordion() {
  const { currentModel } = useModelStore();
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey && (e.key === 'h' || e.key === 'H')) {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  if (!currentModel) return null;
  const entries = Object.entries(currentModel.hyperparameters);

  return (
    <div className="rounded-lg" style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="font-mono-tabular flex w-full items-center gap-1.5 px-3 py-2 text-xs"
        aria-expanded={open}
        title="Toggle with Alt+H"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Inspect model hyperparameters ({entries.length} params · Alt+H)
      </button>
      {open && (
        <div className="border-t px-3 py-2" style={{ borderColor: 'var(--border-subtle)' }}>
          {entries.length === 0 ? (
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              No hyperparameters exposed.
            </p>
          ) : (
            <dl className="font-mono-tabular grid grid-cols-1 gap-x-6 text-[11px] sm:grid-cols-2">
              {entries.map(([k, v]) => (
                <div key={k} className="flex justify-between gap-3 border-b py-1" style={{ borderColor: 'var(--border-subtle)' }}>
                  <dt style={{ color: 'var(--text-secondary)' }}>{k}</dt>
                  <dd className="truncate" style={{ color: 'var(--text-primary)' }} title={String(v)}>
                    {String(v)}
                  </dd>
                </div>
              ))}
            </dl>
          )}
          {currentModel.classes && (
            <p className="font-mono-tabular mt-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              classes_: [{currentModel.classes.map((c) => String(c)).join(', ')}]
              {currentModel.intercept_or_base !== null && (
                <> · intercept: {currentModel.intercept_or_base}</>
              )}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

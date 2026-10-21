import { useWhatIfStore } from '../store/useWhatIfStore';
import { buildRecipeSummary } from '../types/whatif.types';

/**
 * Phase 13 — 1-click nearest counterfactual finder (spec §2.2).
 * "How do I flip the model's decision with the smallest change?"
 */
export function CounterfactualFinderCard() {
  const isClassification = useWhatIfStore((s) => s.isClassification);
  const targetClass = useWhatIfStore((s) => s.targetClass);
  const targetThreshold = useWhatIfStore((s) => s.targetThreshold);
  const setTargetClass = useWhatIfStore((s) => s.setTargetClass);
  const setTargetThreshold = useWhatIfStore((s) => s.setTargetThreshold);
  const lockedIndices = useWhatIfStore((s) => s.lockedIndices);
  const featureNames = useWhatIfStore((s) => s.featureNames);
  const counterfactual = useWhatIfStore((s) => s.counterfactual);
  const isFinding = useWhatIfStore((s) => s.isFinding);
  const findError = useWhatIfStore((s) => s.findError);
  const findCounterfactual = useWhatIfStore((s) => s.findCounterfactual);
  const applyRecipe = useWhatIfStore((s) => s.applyRecipe);
  const evaluation = useWhatIfStore((s) => s.evaluation);

  const lockedNames = lockedIndices.map((i) => featureNames[i] ?? `#${i}`);

  return (
    <div
      data-testid="whatif-counterfactual-card"
      className="flex flex-col gap-2 rounded-md px-2 py-2"
      style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="text-[11px] font-semibold">1-Click Counterfactual Finder</div>

      {isClassification ? (
        <label className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          Flip to class
          <input
            type="number"
            step={1}
            placeholder="auto"
            value={targetClass ?? ''}
            onChange={(e) =>
              setTargetClass(e.target.value === '' ? null : Math.max(0, Math.round(Number(e.target.value))))
            }
            aria-label="Target class (empty = auto flip)"
            className="font-mono-tabular w-24 rounded border px-1.5 py-0.5"
            style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-canvas)' }}
          />
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            empty = nearest other class
          </span>
        </label>
      ) : (
        <label className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          Target prediction ≥
          <input
            type="number"
            step="any"
            placeholder="e.g. 500.0"
            value={targetThreshold ?? ''}
            onChange={(e) => setTargetThreshold(e.target.value === '' ? null : Number(e.target.value))}
            aria-label="Target prediction threshold"
            className="font-mono-tabular w-28 rounded border px-1.5 py-0.5"
            style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-canvas)' }}
          />
        </label>
      )}

      {lockedNames.length > 0 && (
        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          🔒 Locked (untouchable): {lockedNames.join(', ')}
        </div>
      )}

      <button
        type="button"
        onClick={() => void findCounterfactual()}
        disabled={isFinding}
        className="rounded-md px-3 py-1.5 text-[12px] font-semibold disabled:opacity-50"
        style={{ background: 'var(--ml-primary)', color: '#fff' }}
      >
        {isFinding ? 'Searching…' : '✨ Find Minimum Change to Flip Decision'}
      </button>

      {findError && (
        <div data-testid="whatif-find-error" className="rounded px-2 py-1 text-[11px]" role="alert" style={{ background: 'rgba(244,63,94,0.10)', color: '#f43e5e' }}>
          {findError}
        </div>
      )}

      {counterfactual?.found && counterfactual.evaluation && (
        <div data-testid="whatif-recipe" className="rounded px-2 py-1.5 text-[11px]" style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.4)' }}>
          <div className="font-semibold" style={{ color: '#34d399' }}>
            ✅ {counterfactual.target_description || 'Counterfactual found'}
          </div>
          <div className="mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            {buildRecipeSummary(counterfactual.evaluation.changed_features)}
          </div>
          <button
            type="button"
            onClick={applyRecipe}
            className="mt-1.5 rounded px-2 py-1 text-[11px] font-medium"
            style={{ border: '1px solid #34d399', color: '#34d399' }}
          >
            Apply Recipe to Sliders
          </button>
        </div>
      )}

      {evaluation && !counterfactual && (
        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          Tip: lock 🔒 columns you cannot change (e.g. age), then search.
        </div>
      )}
    </div>
  );
}

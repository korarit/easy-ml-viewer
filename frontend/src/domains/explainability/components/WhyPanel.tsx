import { useMemo } from 'react';
import { useExplainabilityStore } from '../store/useExplainabilityStore';
import {
  formatContribution,
  formatRawValue,
  isOutlierPercentile,
  TOP_K_OPTIONS,
  type FrameExplanation,
  type TopKOption,
} from '../types/explainability.types';

/** One plain-language sentence: which features pushed the prediction up/down. */
function VerdictSentence({ explanation, isFlat }: { explanation: FrameExplanation; isFlat: boolean }) {
  const ranked = useMemo(
    () => [...explanation.attributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)),
    [explanation],
  );

  const out = explanation.output_value;
  const outText = Number.isInteger(out) ? String(out) : out.toFixed(3);

  if (isFlat) {
    return (
      <p className="text-xs leading-5" style={{ color: 'var(--text-primary)' }} aria-live="polite">
        Prediction is rock-solid at <strong className="font-mono-tabular">{outText}</strong> — moving any
        single feature to its typical value does not move it. The model is fully confident on this frame,
        so there are no push/pull bars; where this row stands is below.
      </p>
    );
  }

  const pushUp = ranked.find((a) => a.contribution > 0) ?? null;
  const pullDown = ranked.find((a) => a.contribution < 0) ?? null;

  return (
    <p className="text-xs leading-5" style={{ color: 'var(--text-primary)' }} aria-live="polite">
      Prediction <strong className="font-mono-tabular">{outText}</strong>
      {pushUp && (
        <>
          {' '}— mainly pushed <strong>up</strong> by{' '}
          <strong>{pushUp.feature_name}</strong>{' '}
          <span className="font-mono-tabular" style={{ color: '#34d399' }}>
            ({formatContribution(pushUp.contribution)})
          </span>
        </>
      )}
      {pullDown && (
        <>
          {pushUp ? ';' : ' —'} pulled <strong>down</strong> by{' '}
          <strong>{pullDown.feature_name}</strong>{' '}
          <span className="font-mono-tabular" style={{ color: '#fb7185' }}>
            ({formatContribution(pullDown.contribution)})
          </span>
        </>
      )}
      {!pushUp && !pullDown && ' — no single feature dominates this frame'}
      .
    </p>
  );
}

/**
 * Dock "Why" panel — the per-frame reason behind the prediction.
 *
 * Top: plain-language verdict + baseline → output. Below: contribution bars
 * (green pushes up, red pulls down) with the raw value and percentile of
 * each feature, so "why" and "what the input was" sit in one glance.
 */
export function WhyPanel() {
  const explanation = useExplainabilityStore((s) => s.explanation);
  const explanationFrame = useExplainabilityStore((s) => s.explanationFrame);
  const isLoading = useExplainabilityStore((s) => s.isLoading);
  const explainError = useExplainabilityStore((s) => s.explainError);
  const topK = useExplainabilityStore((s) => s.topK);
  const setTopK = useExplainabilityStore((s) => s.setTopK);
  const clearError = useExplainabilityStore((s) => s.clearError);
  const featureCount = useExplainabilityStore((s) => s.featureCount);

  const maxAbs = useMemo(() => {
    if (!explanation || explanation.attributions.length === 0) return 1;
    return Math.max(1e-9, ...explanation.attributions.map((a) => Math.abs(a.contribution)));
  }, [explanation]);

  // Locally flat model (e.g. unanimous kNN neighborhoods, saturated
  // probability): every contribution is ~zero, so contribution bars would
  // be invisible. Rank by deviation from typical instead and say so.
  const isFlat =
    !!explanation &&
    explanation.attributions.length > 0 &&
    explanation.attributions.every((a) => Math.abs(a.contribution) < 1e-9);

  const positionalRanked = useMemo(() => {
    if (!explanation) return [];
    return [...explanation.attributions].sort(
      (a, b) => Math.abs(b.z_score) - Math.abs(a.z_score),
    );
  }, [explanation]);

  const hiddenCount = Math.max(0, featureCount - (explanation?.attributions.length ?? 0));

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto" aria-label="Why this prediction">
      {explainError && (
        <div className="rounded px-2 py-1 text-[11px]" style={{ background: 'rgba(244,63,94,0.10)', color: '#f43e5e' }} role="alert">
          Could not explain this frame: {explainError}{' '}
          <button type="button" onClick={clearError} className="underline">
            dismiss
          </button>
        </div>
      )}

      {isLoading && !explanation ? (
        <div className="flex flex-col gap-2" role="status" aria-label="Loading explanation">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse rounded" style={{ height: 34, background: 'var(--surface-elevated)' }} />
          ))}
        </div>
      ) : !explanation ? (
        <div className="rounded-md px-3 py-4 text-center text-xs" style={{ color: 'var(--text-secondary)' }}>
          Press play or scrub to a frame — the reason behind each prediction shows up here.
        </div>
      ) : (
        <>
          <div
            className="rounded-lg px-3 py-2"
            style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
          >
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <span className="font-mono-tabular text-[10px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
                Frame {explanationFrame ?? explanation.frame_index} · Row #{explanation.dataset_row_index}
              </span>
              <div className="flex gap-1" role="group" aria-label="How many top features to show">
                {TOP_K_OPTIONS.map((k) => (
                  <button
                    key={k}
                    onClick={() => setTopK(k as TopKOption)}
                    aria-pressed={topK === k}
                    className="font-mono-tabular min-h-[28px] rounded px-1.5 text-[11px]"
                    style={{
                      background: topK === k ? 'var(--ml-primary)' : 'transparent',
                      color: topK === k ? '#fff' : 'var(--text-muted)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    {k}
                  </button>
                ))}
                <button
                  onClick={() => setTopK('all')}
                  aria-pressed={topK === 'all'}
                  className="font-mono-tabular min-h-[28px] rounded px-1.5 text-[11px]"
                  style={{
                    background: topK === 'all' ? 'var(--ml-primary)' : 'transparent',
                    color: topK === 'all' ? '#fff' : 'var(--text-muted)',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  All
                </button>
              </div>
            </div>
            <div className="font-mono-tabular mb-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              Baseline {explanation.base_value.toFixed(3)} → Output {explanation.output_value.toFixed(3)}
            </div>
            <VerdictSentence explanation={explanation} isFlat={isFlat} />
          </div>

          {isFlat ? (
            <>
              <p className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                Where this row stands vs typical rows
              </p>
              <ol className="flex flex-col gap-1.5">
                {positionalRanked.map((attr) => (
                  <li key={attr.feature_index}>
                    <div className="flex items-baseline justify-between gap-2 text-xs">
                      <span className="min-w-0 flex-1 truncate" style={{ color: 'var(--text-primary)' }} title={attr.feature_name}>
                        {attr.feature_name}
                        {isOutlierPercentile(attr.percentile) && (
                          <span className="ml-1 text-[10px]" style={{ color: '#f59e0b' }}>
                            ⚠ outlier
                          </span>
                        )}
                      </span>
                      <span className="font-mono-tabular shrink-0" style={{ color: 'var(--text-secondary)' }}>
                        {formatRawValue(attr.raw_value)} · P{Math.round(attr.percentile)}
                      </span>
                    </div>
                    <div
                      className="mt-0.5"
                      style={{ position: 'relative', height: 8, background: 'rgba(148,163,184,0.15)', borderRadius: 4, overflow: 'hidden' }}
                    >
                      <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'rgba(226,232,240,0.6)' }} />
                      <div
                        style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: `${Math.max(0, Math.min(100, attr.percentile)).toFixed(1)}%`,
                          background: isOutlierPercentile(attr.percentile) ? '#f59e0b' : '#38bdf8',
                          borderRadius: 4,
                        }}
                      />
                    </div>
                    <div className="font-mono-tabular mt-0.5 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {Number.isFinite(attr.z_score) ? `z=${attr.z_score.toFixed(2)} vs typical` : 'typical value unavailable'}
                    </div>
                  </li>
                ))}
              </ol>
              <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                Blue = where this row sits in the data (median tick in the middle). Scrub to a frame where
                the model hesitates and the push/pull bars will appear.
              </p>
            </>
          ) : (
          <>
          <ol className="flex flex-col gap-1.5">
            {explanation.attributions.map((attr) => {
              const frac = Math.abs(attr.contribution) / maxAbs;
              const positive = attr.contribution >= 0;
              return (
                <li key={attr.feature_index}>
                  <div className="flex items-baseline justify-between gap-2 text-xs">
                    <span className="min-w-0 flex-1 truncate" style={{ color: 'var(--text-primary)' }} title={attr.feature_name}>
                      {attr.feature_name}
                      {isOutlierPercentile(attr.percentile) && (
                        <span className="ml-1 text-[10px]" style={{ color: '#f59e0b' }}>
                          ⚠ outlier
                        </span>
                      )}
                    </span>
                    <span className="font-mono-tabular shrink-0" style={{ color: positive ? '#34d399' : '#fb7185' }}>
                      {formatContribution(attr.contribution)}
                    </span>
                  </div>
                  <div
                    className="mt-0.5"
                    style={{ height: 8, background: 'rgba(148,163,184,0.15)', borderRadius: 4, overflow: 'hidden' }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: '100%',
                        transform: `scaleX(${frac.toFixed(4)})`,
                        transformOrigin: positive ? 'left center' : 'right center',
                        background: positive ? '#34d399' : '#fb7185',
                      }}
                    />
                  </div>
                  <div className="font-mono-tabular mt-0.5 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    value {formatRawValue(attr.raw_value)} · P{Math.round(attr.percentile)} of data
                    {Number.isFinite(attr.z_score) && ` · z=${attr.z_score.toFixed(2)}`}
                  </div>
                </li>
              );
            })}
          </ol>
          {Math.abs(explanation.other_contributions_sum) > 1e-12 && (
            <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
              {hiddenCount > 0 ? `Other ${hiddenCount} features` : 'Other features'} combined:{' '}
              {formatContribution(explanation.other_contributions_sum)}
            </p>
          )}
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            Green pushed the prediction up, red pulled it down. Bars scale to the strongest feature —
            scrub to another frame to compare.
          </p>
          </>
          )}
        </>
      )}
    </div>
  );
}

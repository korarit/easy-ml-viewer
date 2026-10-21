import { useEffect, useMemo } from 'react';
import { useExplainabilityStore } from '../store/useExplainabilityStore';
import {
  formatContribution,
  isOutlierPercentile,
  TOP_K_OPTIONS,
  type FrameExplanation,
  type TopKOption,
} from '../types/explainability.types';

/**
 * Phase 12 — real-time feature-contribution waterfall.
 *
 * GPU-composited bars: width is set via `transform: scaleX(...)` with
 * `will-change: transform` (never `width`) so 60 FPS scrubbing never
 * triggers layout thrash (§5.2). Self-contained `H` / `E` shortcuts so
 * the panel works mounted standalone as well as inside the studio.
 */
export function FeatureWaterfallPanel({ explanation }: { explanation: FrameExplanation | null }) {
  const isLoading = useExplainabilityStore((s) => s.isLoading);
  const topK = useExplainabilityStore((s) => s.topK);
  const setTopK = useExplainabilityStore((s) => s.setTopK);
  const hudVisible = useExplainabilityStore((s) => s.hudVisible);
  const panelCollapsed = useExplainabilityStore((s) => s.panelCollapsed);
  const toggleHud = useExplainabilityStore((s) => s.toggleHud);
  const togglePanel = useExplainabilityStore((s) => s.togglePanel);
  const featureCount = useExplainabilityStore((s) => s.featureCount);

  useEffect(() => {
    const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (INPUT_TAGS.has(target.tagName) || target.isContentEditable)) return;
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        useExplainabilityStore.getState().toggleHud();
      } else if (e.key === 'e' || e.key === 'E') {
        e.preventDefault();
        useExplainabilityStore.getState().togglePanel();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const maxAbs = useMemo(() => {
    if (!explanation || explanation.attributions.length === 0) return 1;
    return Math.max(1e-9, ...explanation.attributions.map((a) => Math.abs(a.contribution)));
  }, [explanation]);

  if (panelCollapsed) {
    return (
      <div data-hud-hidden={hudVisible ? 'false' : 'true'} data-testid="waterfall-collapsed">
        <button onClick={togglePanel} aria-label="Expand explainability panel">
          Expand explainability (E)
        </button>
      </div>
    );
  }

  if (isLoading && !explanation) {
    return (
      <div data-hud-hidden={hudVisible ? 'false' : 'true'} data-testid="waterfall-skeleton">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="animate-pulse rounded"
            style={{ height: 18, marginBottom: 8, background: 'var(--bg-elevated, #1a2230)' }}
          />
        ))}
      </div>
    );
  }

  if (!explanation) {
    return (
      <div data-hud-hidden={hudVisible ? 'false' : 'true'} data-testid="waterfall-empty">
        เลือกแถวข้อมูลหรือกด Play เพื่อดูการแยกส่วนเหตุผลของโมเดล (Local Explainability)
      </div>
    );
  }

  const delta = explanation.output_value - explanation.base_value;
  const hasOther = Math.abs(explanation.other_contributions_sum) > 1e-12;
  const hiddenCount = Math.max(0, featureCount - explanation.attributions.length);

  return (
    <div data-hud-hidden={hudVisible ? 'false' : 'true'} data-testid="waterfall-panel">
      <div className="flex items-center justify-between gap-2">
        <div>
          Base E[f(x)]: {explanation.base_value.toFixed(3)} → Output:{' '}
          {explanation.output_value.toFixed(3)} (Δ {formatContribution(delta)})
        </div>
        <div className="flex gap-1">
          <button onClick={toggleHud} aria-label="Toggle 3D HUD (H)">
            H
          </button>
          <button onClick={togglePanel} aria-label="Collapse explainability panel (E)">
            E
          </button>
        </div>
      </div>

      <div className="flex gap-1" role="group" aria-label="Top-K selector">
        {TOP_K_OPTIONS.map((k) => (
          <button
            key={k}
            onClick={() => setTopK(k as TopKOption)}
            aria-pressed={topK === k}
            style={{ fontWeight: topK === k ? 700 : 400 }}
          >
            Top {k}
          </button>
        ))}
        <button onClick={() => setTopK('all')} aria-pressed={topK === 'all'}>
          All
        </button>
      </div>

      <div>
        {explanation.attributions.map((attr) => {
          const frac = Math.abs(attr.contribution) / maxAbs;
          const positive = attr.contribution >= 0;
          return (
            <div key={attr.feature_index} data-testid={`waterfall-row-${attr.feature_name}`}>
              <span>
                {attr.feature_name} {formatContribution(attr.contribution)}
                {isOutlierPercentile(attr.percentile) && (
                  <span style={{ color: '#f59e0b', marginLeft: 4 }}>⚠ Outlier</span>
                )}
              </span>
              <div
                style={{
                  height: 10,
                  background: 'rgba(148,163,184,0.15)',
                  borderRadius: 4,
                  overflow: 'hidden',
                }}
              >
                <div
                  data-testid={`waterfall-bar-${attr.feature_name}`}
                  style={{
                    height: '100%',
                    width: '100%',
                    transform: `scaleX(${frac.toFixed(4)})`,
                    transformOrigin: positive ? 'left center' : 'right center',
                    willChange: 'transform',
                    background: positive ? '#34d399' : '#fb7185',
                  }}
                />
              </div>
            </div>
          );
        })}
        {hasOther && (
          <div data-testid="waterfall-row-other">
            <span>
              Other{hiddenCount > 0 ? ` (${hiddenCount})` : ''} features{' '}
              {formatContribution(explanation.other_contributions_sum)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

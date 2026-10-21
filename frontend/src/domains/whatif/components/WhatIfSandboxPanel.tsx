import { useMemo } from 'react';
import { useWhatIfStore } from '../store/useWhatIfStore';
import { WhatIfFeatureSliderRow } from './WhatIfFeatureSliderRow';
import { ComparisonScorecard } from './ComparisonScorecard';
import { CounterfactualFinderCard } from './CounterfactualFinderCard';
import { isSandboxActive, type FeatureChangeDelta } from '../types/whatif.types';

/**
 * Phase 13 — What-If Sandbox control panel (spec §2.1).
 * Live mutation sliders + before/after scorecard + counterfactual finder.
 */
export function WhatIfSandboxPanel() {
  const sessionId = useWhatIfStore((s) => s.sessionId);
  const isSessionLoading = useWhatIfStore((s) => s.isSessionLoading);
  const sessionError = useWhatIfStore((s) => s.sessionError);
  const panelOpen = useWhatIfStore((s) => s.panelOpen);
  const togglePanel = useWhatIfStore((s) => s.togglePanel);
  const featureRanges = useWhatIfStore((s) => s.featureRanges);
  const featureNames = useWhatIfStore((s) => s.featureNames);
  const isClassification = useWhatIfStore((s) => s.isClassification);
  const baseRowValues = useWhatIfStore((s) => s.baseRowValues);
  const mutatedValues = useWhatIfStore((s) => s.mutatedValues);
  const lockedIndices = useWhatIfStore((s) => s.lockedIndices);
  const evaluation = useWhatIfStore((s) => s.evaluation);
  const isEvaluating = useWhatIfStore((s) => s.isEvaluating);
  const evalError = useWhatIfStore((s) => s.evalError);
  const lastEvalMs = useWhatIfStore((s) => s.lastEvalMs);
  const activeFrame = useWhatIfStore((s) => s.activeFrame);
  const rowIndices = useWhatIfStore((s) => s.rowIndices);
  const setMutatedValue = useWhatIfStore((s) => s.setMutatedValue);
  const toggleLock = useWhatIfStore((s) => s.toggleLock);
  const resetAll = useWhatIfStore((s) => s.resetAll);
  const clearErrors = useWhatIfStore((s) => s.clearErrors);

  const changeByIndex = useMemo(() => {
    const map = new Map<number, FeatureChangeDelta>();
    for (const c of evaluation?.changed_features ?? []) map.set(c.feature_index, c);
    return map;
  }, [evaluation]);

  const sandboxActive = useMemo(
    () => (baseRowValues.length > 0 ? isSandboxActive(mutatedValues, baseRowValues) : false),
    [mutatedValues, baseRowValues],
  );

  if (!panelOpen) {
    return (
      <div data-testid="whatif-panel-collapsed">
        <button
          type="button"
          onClick={togglePanel}
          aria-label="Open What-If Sandbox panel (W)"
          className="rounded-md px-3 py-2 text-xs font-medium"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          🧪 Open What-If Sandbox (W)
        </button>
      </div>
    );
  }

  return (
    <div
      data-testid="whatif-sandbox-panel"
      className="flex max-h-full flex-col gap-2 overflow-y-auto rounded-lg px-3 py-2.5"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      aria-label="What-If Sandbox"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
          🧪 What-If & Counterfactual Sandbox
          {sandboxActive && (
            <span
              data-testid="whatif-active-badge"
              className="ml-2 rounded px-1.5 py-0.5"
              style={{ background: 'rgba(56,189,248,0.15)', color: '#38bdf8' }}
            >
              [ 🧪 WHAT-IF SANDBOX ACTIVE ]
            </span>
          )}
        </span>
        <div className="flex gap-1">
          <button
            type="button"
            onClick={resetAll}
            disabled={!sessionId || (!sandboxActive && !evaluation)}
            aria-label="Reset all to original (R)"
            title="Reset all to original (R)"
            className="rounded px-1.5 py-0.5 text-[11px] disabled:opacity-40"
            style={{ border: '1px solid var(--border-subtle)' }}
          >
            ↺ Reset All to Original
          </button>
          <button
            type="button"
            onClick={togglePanel}
            aria-label="Close What-If Sandbox panel (W)"
            className="rounded px-1.5 py-0.5 text-[11px]"
            style={{ border: '1px solid var(--border-subtle)' }}
          >
            W
          </button>
        </div>
      </div>

      {isSessionLoading && !sessionId ? (
        <div className="flex flex-col gap-2" role="status" aria-label="Loading what-if session">
          {[0, 1, 2].map((i) => (
            <div key={i} className="animate-pulse rounded" style={{ height: 44, background: 'var(--surface-elevated)' }} />
          ))}
        </div>
      ) : sessionError ? (
        <div className="rounded px-2 py-1 text-[11px]" role="alert" style={{ background: 'rgba(244,63,94,0.10)', color: '#f43e5e' }}>
          What-If failed: {sessionError}{' '}
          <button type="button" onClick={clearErrors} className="underline">
            dismiss
          </button>
        </div>
      ) : !sessionId ? (
        <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Load a mapped queue to open the sandbox.
        </div>
      ) : (
        <>
          <div className="font-mono-tabular text-[10px]" style={{ color: 'var(--text-muted)' }}>
            Frame {activeFrame}
            {rowIndices[activeFrame] !== undefined && ` · Original Row #${rowIndices[activeFrame]}`}
          </div>

          <ComparisonScorecard
            evaluation={evaluation}
            isClassification={isClassification}
            isEvaluating={isEvaluating}
            lastEvalMs={lastEvalMs}
          />

          {evalError && (
            <div className="rounded px-2 py-1 text-[11px]" role="alert" style={{ background: 'rgba(244,63,94,0.10)', color: '#f43e5e' }}>
              Evaluate failed: {evalError}{' '}
              <button type="button" onClick={clearErrors} className="underline">
                dismiss
              </button>
            </div>
          )}

          <div className="flex flex-col gap-1.5" role="group" aria-label="Feature mutation sliders">
            {featureRanges.map((range) => (
              <WhatIfFeatureSliderRow
                key={range.feature_index}
                range={range}
                baseValue={baseRowValues[range.feature_index] ?? range.mean_value}
                mutatedValue={mutatedValues[range.feature_index]}
                locked={lockedIndices.includes(range.feature_index)}
                change={changeByIndex.get(range.feature_index)}
                disabled={!sessionId || baseRowValues.length === 0}
                onChange={setMutatedValue}
                onToggleLock={toggleLock}
              />
            ))}
          </div>
          {featureNames.length === 0 && (
            <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              No mapped features in this session.
            </div>
          )}

          <CounterfactualFinderCard />
        </>
      )}
    </div>
  );
}

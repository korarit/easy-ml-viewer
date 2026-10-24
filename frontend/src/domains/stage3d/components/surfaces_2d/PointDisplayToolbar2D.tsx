import { useEffect } from 'react';
import { useStage3DStore } from '../../store/useStage3DStore';
import type { HexbinMode } from './HexbinOverlay2D';

function Toggle({
  kbd,
  label,
  on,
  onClick,
  title,
}: {
  kbd: string;
  label: string;
  on: boolean;
  onClick: () => void;
  title: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      title={`${title} (shortcut: ${kbd})`}
      className="font-mono-tabular rounded px-2 py-1 text-[11px]"
      style={{
        background: on ? 'rgba(34,211,238,0.18)' : 'rgba(10,15,22,0.72)',
        border: `1px solid ${on ? '#22d3ee' : 'var(--border-subtle)'}`,
        color: on ? '#22d3ee' : 'var(--text-secondary)',
        cursor: 'pointer',
      }}
    >
      {label} <span style={{ opacity: 0.6 }}>({kbd})</span>
    </button>
  );
}

export const TRAIL_CYCLE: Array<5 | 10 | 20 | 'all'> = [5, 10, 20, 'all'];
export const HEXBIN_CYCLE: HexbinMode[] = ['count', 'error', 'class'];

/**
 * Phase 56 — unified Point Intelligence toolbar + global hotkeys.
 *
 * Matrix (spec §3): T toggle trails · Shift+T cycle 5→10→20→All ·
 * R dual-ring · E error lens · H hexbin · Shift+H cycle Count→Error%→Class.
 * Keys are ignored while typing in inputs/selects so scrubbing never
 * hijacks forms, and Shift variants never collide with bare-letter toggles.
 */
export function PointDisplayToolbar2D({
  hexbinMode,
  onHexbinMode,
}: {
  hexbinMode: HexbinMode;
  onHexbinMode: (m: HexbinMode) => void;
}) {
  const showCometTrails = useStage3DStore((s) => s.showCometTrails);
  const showDualRings = useStage3DStore((s) => s.showDualRings);
  const showErrorsOnly = useStage3DStore((s) => s.showErrorsOnly);
  const showHexbins = useStage3DStore((s) => s.showHexbins);
  const showResidualSpokes = useStage3DStore((s) => s.showResidualSpokes);
  const trailLength = useStage3DStore((s) => s.trailLength);
  const storeMetric = useStage3DStore((s) => s.hexbinMetric);

  // Keep the legacy local hexbinMode prop in sync with the global store so
  // Stage2DCanvas (prop owner) and hotkeys (store owner) never diverge.
  useEffect(() => {
    if (hexbinMode !== storeMetric) onHexbinMode(storeMetric);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storeMetric]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const st = useStage3DStore.getState();
      const k = e.key.toLowerCase();
      if (k === 't') {
        if (e.shiftKey) {
          st.cycleTrailLength();
          e.preventDefault();
        } else {
          st.setShowCometTrails(!st.showCometTrails);
        }
      } else if (k === 'r' && !e.shiftKey) {
        st.setShowDualRings(!st.showDualRings);
      } else if (k === 'e' && !e.shiftKey) {
        st.setShowErrorsOnly(!st.showErrorsOnly);
      } else if (k === 'h') {
        if (e.shiftKey) {
          st.cycleHexbinMetric();
          e.preventDefault();
        } else {
          st.setShowHexbins(!st.showHexbins);
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const st = useStage3DStore.getState();
  const trailLabel = trailLength === 'all' ? 'All' : `${trailLength}f`;
  return (
    <div
      className="pointer-events-auto absolute top-1 left-1/2 flex -translate-x-1/2 items-center gap-1"
      role="toolbar"
      aria-label="Point display toolbar"
    >
      <Toggle kbd="T" label="Trails" on={showCometTrails} onClick={() => st.setShowCometTrails(!showCometTrails)} title="Comet trails + ghost dots" />
      <button
        type="button"
        onClick={() => st.cycleTrailLength()}
        title="Cycle trail length: 5 → 10 → 20 → All frames (shortcut: Shift+T)"
        aria-label={`Trail length ${trailLabel}, activate to cycle`}
        className="font-mono-tabular rounded px-2 py-1 text-[11px]"
        style={{
          background: 'rgba(10,15,22,0.72)',
          border: '1px solid var(--border-subtle)',
          color: 'var(--text-secondary)',
          cursor: 'pointer',
        }}
      >
        {trailLabel} <span style={{ opacity: 0.6 }}>(⇧T)</span>
      </button>
      <Toggle kbd="R" label="Dual-Ring" on={showDualRings} onClick={() => st.setShowDualRings(!showDualRings)} title="Dual-ring actual/prediction glyphs" />
      <Toggle kbd="E" label="Errors" on={showErrorsOnly} onClick={() => st.setShowErrorsOnly(!showErrorsOnly)} title="Error lens: mispredictions only" />
      <Toggle kbd="H" label="Hexbin" on={showHexbins} onClick={() => st.setShowHexbins(!showHexbins)} title="Hexbin density overlay" />
      <button
        type="button"
        onClick={() => st.setShowResidualSpokes(!showResidualSpokes)}
        aria-pressed={showResidualSpokes}
        title="Residual error spokes"
        className="font-mono-tabular rounded px-2 py-1 text-[11px]"
        style={{
          background: showResidualSpokes ? 'rgba(251,146,60,0.18)' : 'rgba(10,15,22,0.72)',
          border: `1px solid ${showResidualSpokes ? '#fb923c' : 'var(--border-subtle)'}`,
          color: showResidualSpokes ? '#fb923c' : 'var(--text-secondary)',
          cursor: 'pointer',
        }}
      >
        Spokes
      </button>
      {showHexbins && (
        <select
          aria-label="Hexbin display mode"
          value={hexbinMode}
          onChange={(e) => {
            const m = e.target.value as HexbinMode;
            onHexbinMode(m);
            st.setHexbinMetric(m);
          }}
          title="Cycle hexbin metric: Count → Error % → Class (shortcut: Shift+H)"
          className="font-mono-tabular rounded px-1 py-1 text-[11px]"
          style={{ background: 'rgba(10,15,22,0.9)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
        >
          <option value="count">Density</option>
          <option value="error">Error %</option>
          <option value="class">Class</option>
        </select>
      )}
    </div>
  );
}

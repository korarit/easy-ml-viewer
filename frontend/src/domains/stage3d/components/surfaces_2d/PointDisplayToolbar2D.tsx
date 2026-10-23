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

/**
 * Phase 49 — quick toggles for the point-intelligence layer.
 *
 * `[Trails (T)] [Dual-Ring (R)] [Errors (E)] [Hexbin (H)]`. Keys are
 * ignored while typing in inputs so scrubbing never hijacks forms.
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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      const st = useStage3DStore.getState();
      const k = e.key.toLowerCase();
      if (k === 't') st.setShowCometTrails(!st.showCometTrails);
      else if (k === 'r') st.setShowDualRings(!st.showDualRings);
      else if (k === 'e') st.setShowErrorsOnly(!st.showErrorsOnly);
      else if (k === 'h') st.setShowHexbins(!st.showHexbins);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const st = useStage3DStore.getState();
  return (
    <div
      className="pointer-events-auto absolute top-1 left-1/2 flex -translate-x-1/2 items-center gap-1"
      role="toolbar"
      aria-label="Point display toolbar"
    >
      <Toggle kbd="T" label="Trails" on={showCometTrails} onClick={() => st.setShowCometTrails(!showCometTrails)} title="Comet trails + ghost dots" />
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
          onChange={(e) => onHexbinMode(e.target.value as HexbinMode)}
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

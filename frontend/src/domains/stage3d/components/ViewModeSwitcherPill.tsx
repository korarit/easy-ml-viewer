import { useEffect } from 'react';
import { useStage3DStore, type ViewMode } from '../store/useStage3DStore';

/**
 * Phase 41 — dual-mode viewport switcher ([2D Planar] / [3D Spatial]).
 *
 * The 2D planar view is an orthographic-equivalent top-down projection:
 * world X/Y map 1:1 onto the screen with no perspective divide, so decision
 * boundaries and isolines read without distortion or occlusion.
 * Global shortcut: `V` (ignored while typing).
 */
export function ViewModeSwitcherPill() {
  const viewMode = useStage3DStore((s) => s.viewMode);
  const setViewMode = useStage3DStore((s) => s.setViewMode);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'SELECT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable);
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === 'v' || e.key === 'V') {
        e.preventDefault();
        useStage3DStore.getState().toggleViewMode();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const seg = (mode: ViewMode, label: string, title: string) => {
    const on = viewMode === mode;
    return (
      <button
        role="tab"
        aria-selected={on}
        title={title}
        onClick={() => setViewMode(mode)}
        className="min-h-[32px] px-2.5 py-1 text-xs font-medium"
        style={
          on
            ? mode === '2d'
              ? { background: '#0ea5e9', color: '#fff' }
              : { background: 'var(--ml-primary)', color: '#fff' }
            : { background: 'transparent', color: 'var(--text-secondary)' }
        }
      >
        {label}
      </button>
    );
  };

  return (
    <div
      className="flex shrink-0 items-center gap-2 rounded-lg px-2 py-1"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
      role="tablist"
      aria-label="Viewport mode"
    >
      <div className="flex overflow-hidden rounded" style={{ border: '1px solid var(--border-subtle)' }}>
        {seg('2d', '2D Planar', 'Orthographic top-down view — decision boundaries (V)')}
        {seg('3d', '3D Spatial', 'Perspective orbit view (V)')}
      </div>
      <span className="hidden font-mono-tabular text-[11px] xl:inline" style={{ color: 'var(--text-muted)' }}>
        {viewMode === '2d' ? 'Ortho · no distortion' : 'Perspective'}
      </span>
    </div>
  );
}

/**
 * Phase 41 — 2D layer visibility toggles (Scene dock tab card).
 */
export function LayerVisibilityControls2D() {
  const showBoundaryLines = useStage3DStore((s) => s.showBoundaryLines);
  const showConfidenceIsolines = useStage3DStore((s) => s.showConfidenceIsolines);
  const showHeatmap = useStage3DStore((s) => s.showHeatmap);
  const showRulers = useStage3DStore((s) => s.showRulers);
  const showLinearExact = useStage3DStore((s) => s.showLinearExact);
  const showNormalVector = useStage3DStore((s) => s.showNormalVector);
  const showMarginCorridor = useStage3DStore((s) => s.showMarginCorridor);
  const showSparsityBadges = useStage3DStore((s) => s.showSparsityBadges);
  const showKnnRadar = useStage3DStore((s) => s.showKnnRadar);
  const showSpiderLines = useStage3DStore((s) => s.showSpiderLines);
  const showSvHalos = useStage3DStore((s) => s.showSvHalos);
  const showSvmMargins = useStage3DStore((s) => s.showSvmMargins);
  const st = useStage3DStore.getState();

  const row = (label: string, checked: boolean, onChange: (v: boolean) => void, hint: string) => (
    <label
      className="flex min-h-[32px] cursor-pointer items-center gap-2 text-xs"
      style={{ color: 'var(--text-secondary)' }}
      title={hint}
    >
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );

  return (
    <div
      className="rounded-lg px-3 py-2"
      style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
      aria-label="2D layer visibility"
    >
      <p className="mb-1 text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
        2D Layers
      </p>
      <div className="grid grid-cols-2 gap-x-3">
        {row('Boundary lines', showBoundaryLines, (v) => st.setShowBoundaryLines(v), 'Decision boundary (P = 0.5)')}
        {row('Confidence isolines', showConfidenceIsolines, (v) => st.setShowConfidenceIsolines(v), '60 / 70 / 80 / 90% isolines')}
        {row('Density heatmap', showHeatmap, (v) => st.setShowHeatmap(v), 'Probability / prediction background')}
        {row('Precision rulers', showRulers, (v) => st.setShowRulers(v), 'Raw-value axis rulers')}
        {row('Exact linear line', showLinearExact, (v) => st.setShowLinearExact(v), 'Analytic linear boundary + confidence parallels (Phase 42)')}
        {row('Normal vector', showNormalVector, (v) => st.setShowNormalVector(v), 'Weight ascent arrow on the exact line')}
        {row('Margin corridor', showMarginCorridor, (v) => st.setShowMarginCorridor(v), 'SGD/Ridge z = ±1 band (W = 2/‖w‖)')}
        {row('Sparsity badges', showSparsityBadges, (v) => st.setShowSparsityBadges(v), 'L1 zeroed-axis warnings')}
        {row('KNN radar circle', showKnnRadar, (v) => st.setShowKnnRadar(v), 'Expanding 2D radar around the active frame')}
        {row('Spider lines', showSpiderLines, (v) => st.setShowSpiderLines(v), 'Laser links to the K reference rows')}
        {row('SV halos', showSvHalos, (v) => st.setShowSvHalos(v), 'Diamond rings sized by |α|/C')}
        {row('SVM margins', showSvmMargins, (v) => st.setShowSvmMargins(v), 'Curved f = ±1 parallels (RBF/poly)')}
      </div>
    </div>
  );
}

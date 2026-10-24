import { describe, expect, it } from 'vitest';
import { HEXBIN_CYCLE, TRAIL_CYCLE } from '../../components/surfaces_2d/PointDisplayToolbar2D';
import { useStage3DStore } from '../../store/useStage3DStore';

describe('Phase 56 unified point-intelligence controls', () => {
  it('cycles trail length 5 → 10 → 20 → All → 5 (Shift+T)', () => {
    expect(TRAIL_CYCLE).toEqual([5, 10, 20, 'all']);
    useStage3DStore.getState().setTrailLength(5);
    useStage3DStore.getState().cycleTrailLength();
    expect(useStage3DStore.getState().trailLength).toBe(10);
    useStage3DStore.getState().cycleTrailLength();
    expect(useStage3DStore.getState().trailLength).toBe(20);
    useStage3DStore.getState().cycleTrailLength();
    expect(useStage3DStore.getState().trailLength).toBe('all');
    useStage3DStore.getState().cycleTrailLength();
    expect(useStage3DStore.getState().trailLength).toBe(5);
  });

  it('cycles hexbin metric Count → Error% → Class → Count (Shift+H)', () => {
    expect(HEXBIN_CYCLE).toEqual(['count', 'error', 'class']);
    useStage3DStore.getState().setHexbinMetric('count');
    useStage3DStore.getState().cycleHexbinMetric();
    expect(useStage3DStore.getState().hexbinMetric).toBe('error');
    useStage3DStore.getState().cycleHexbinMetric();
    expect(useStage3DStore.getState().hexbinMetric).toBe('class');
    useStage3DStore.getState().cycleHexbinMetric();
    expect(useStage3DStore.getState().hexbinMetric).toBe('count');
  });

  it('keeps T/R/E/H defaults conflict-free (trails+dual on, errors off)', () => {
    const st = useStage3DStore.getState();
    st.setShowCometTrails(true);
    st.setShowDualRings(true);
    st.setShowErrorsOnly(false);
    st.setShowHexbins(false);
    expect(useStage3DStore.getState().showCometTrails).toBe(true);
    expect(useStage3DStore.getState().showDualRings).toBe(true);
    expect(useStage3DStore.getState().showErrorsOnly).toBe(false);
  });
});

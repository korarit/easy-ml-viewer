import { useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { useStage3DStore } from '../store/useStage3DStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';
import { DecisionSurfaceMesh } from '../../projection3d/components/DecisionSurfaceMesh';
import { CameraRigController } from './CameraRigController';
import { CoordinateCage3D } from './CoordinateCage3D';
import { ParametricSurfaceMesh } from './ParametricSurfaceMesh';
import { InstancedPointCloud } from './InstancedPointCloud';
import { ActiveDataPointSphere } from './ActiveDataPointSphere';
import { ActiveBallHudOverlay } from '../../explainability/components/ActiveBallHudOverlay';
import { ActiveSampleSphere3D } from '../../playback/components/ActiveSampleSphere3D';
import { MotionTrail3D } from '../../playback/components/MotionTrail3D';
import { GhostComparisonGroup3D } from '../../whatif/components/GhostComparisonGroup3D';
import { CameraPresetHUD } from './CameraPresetHUD';

/** Reports FPS + draw calls to the store ~2x per second (no per-frame setState). */
function PerfReporter() {
  const gl = useThree((s) => s.gl);
  useFrame(() => {
    const now = performance.now();
    const last = (PerfReporter as { _last?: number })._last ?? now;
    const frames = ((PerfReporter as { _frames?: number })._frames ?? 0) + 1;
    (PerfReporter as { _frames?: number })._frames = frames;
    if (now - last >= 500) {
      const fps = (frames * 1000) / Math.max(now - last, 1);
      (PerfReporter as { _last?: number })._last = now;
      (PerfReporter as { _frames?: number })._frames = 0;
      const st = useStage3DStore.getState();
      const calls = gl.info.render.calls;
      if (Math.abs(st.fps - fps) > 0.5 || st.drawCalls !== calls) st.reportPerf(Math.round(fps * 10) / 10, calls);
    }
  });
  return null;
}

/**
 * Phase 08 — WebGL stage canvas. 100% offline: local lights only
 * (no drei Environment HDR fetch, no remote Text fonts — labels use
 * DOM-based Html with bundled fonts).
 */
export function Stage3DCanvas() {
  const contextLost = useStage3DStore((s) => s.contextLost);
  const setContextLost = useStage3DStore((s) => s.setContextLost);
  const restoreContext = useStage3DStore((s) => s.restoreContext);
  const sceneKey = useStage3DStore((s) => s.sceneKey);
  const fps = useStage3DStore((s) => s.fps);
  const drawCalls = useStage3DStore((s) => s.drawCalls);
  const pointCount = useStage3DStore((s) => s.points.length);
  // Phase 09: the LOD projection surface takes over when baked; otherwise
  // the Phase-08 parametric surface renders (offline fallback path).
  const hasProjectionSurface = useProjectionStore((s) => s.surfaceData !== null);

  useEffect(() => {
    const canvas = document.querySelector('[data-stage3d-canvas="true"] canvas');
    if (!canvas) return;
    const onLost = (e: Event) => {
      e.preventDefault();
      useStage3DStore.getState().setContextLost(true);
    };
    canvas.addEventListener('webglcontextlost', onLost);
    return () => canvas.removeEventListener('webglcontextlost', onLost);
  }, [sceneKey]);

  return (
    <div className="relative min-h-[420px] w-full flex-1" data-stage3d-canvas="true" style={{ minHeight: 420 }}>
      <Canvas
        key={sceneKey}
        dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
        camera={{ position: [10.5, 8.5, 12.5], fov: 45, near: 0.1, far: 200 }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        onCreated={({ gl }) => {
          const el = gl.domElement;
          el.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            useStage3DStore.getState().setContextLost(true);
          });
        }}
      >
        <color attach="background" args={['#0b0f14']} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[10, 15, 10]} intensity={1.2} />
        <hemisphereLight args={['#bcd2f5', '#1a2230', 0.5]} />
        <CameraRigController />
        <CoordinateCage3D />
        {hasProjectionSurface ? <DecisionSurfaceMesh /> : <ParametricSurfaceMesh />}
        <InstancedPointCloud />
        <ActiveDataPointSphere />
        <ActiveBallHudOverlay />
        <ActiveSampleSphere3D />
        <MotionTrail3D />
        <GhostComparisonGroup3D />
        <PerfReporter />
      </Canvas>

      <CameraPresetHUD />

      <div
        className="font-mono-tabular pointer-events-none absolute bottom-2 left-2 rounded px-2 py-1 text-[11px]"
        style={{ background: 'rgba(10,15,22,0.72)', color: 'var(--text-secondary)' }}
      >
        FPS: {fps.toFixed(1)} | Draw Calls: {drawCalls} | Instanced Points: {pointCount} | WebGL2
      </div>

      {contextLost && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center gap-3"
          style={{ background: 'rgba(8,12,18,0.82)' }}
          role="alert"
        >
          <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            3D context was lost (GPU reset)
          </p>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            Buffers were released safely — restore to rebuild the scene.
          </p>
          <button
            onClick={() => {
              setContextLost(false);
              restoreContext();
            }}
            className="rounded-md px-4 py-2 text-xs font-medium"
            style={{ background: 'var(--ml-primary)', color: '#fff' }}
          >
            Restore 3D Context
          </button>
        </div>
      )}
    </div>
  );
}

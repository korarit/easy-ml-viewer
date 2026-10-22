import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { usePipelineSliceStore } from '../store/usePipelineSliceStore';
import {
  normalizePipelineHeight,
  pipelineGridToStage,
  pipelinePosteriorHeight,
} from '../../stage3d/components/surfaces/pipelineSurfaceMath';
import { useDisposableThreeResources } from '../../stage3d/hooks/useDisposableThreeResources';

/** Phase 37 — raw vs transformed space toggle + delegated decision surface. */
export function PipelineSurfaceSandbox() {
  const {
    featureNames,
    axisX,
    axisY,
    coordSpace,
    surface,
    isSurfacing,
    showSurface,
    showWireframe,
    showTelemetry,
    setAxisX,
    setAxisY,
    setCoordSpace,
    setShowSurface,
    setShowWireframe,
    traces,
  } = usePipelineSliceStore();

  const mech = traces[0]?.extra_mechanics ?? null;
  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const telemetry = surface?.telemetry;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'S' || e.key === 's')) {
        e.preventDefault();
        const cur = usePipelineSliceStore.getState().showTelemetry;
        usePipelineSliceStore.getState().setShowTelemetry(!cur);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const cells = useMemo(() => {
    if (!surface || res < 2 || surface.z_values.length < res * res) return [];
    const isClassifier = overlays?.is_classifier !== false;
    const out: Array<{ x: number; y: number; z: number; sx: number; sy: number; sz: number; shade: number }> = [];
    const cell = 5 / res;
    const lo = telemetry?.z_min ?? 0;
    const hi = telemetry?.z_max ?? 1;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const raw = surface.z_values[r * res + c] ?? 0;
        const h = isClassifier ? pipelinePosteriorHeight(raw) : normalizePipelineHeight(lo, hi, raw);
        out.push({
          x: -2.5 + (c + 0.5) * cell,
          y: Math.max(h, 0.02) / 2,
          z: -2.5 + (r + 0.5) * cell,
          sx: cell * 0.92,
          sy: Math.max(h, 0.02),
          sz: cell * 0.92,
          shade: isClassifier ? Math.min(1, Math.max(0, raw)) : h / 2,
        });
      }
    }
    return out;
  }, [surface, res, overlays, telemetry]);

  const meshRef = useRef<THREE.InstancedMesh>(null!);
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || cells.length === 0) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    const col = new THREE.Color();
    cells.forEach((cell, i) => {
      pos.set(cell.x, cell.y, cell.z);
      scl.set(cell.sx, cell.sy, cell.sz);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      col.setRGB(0.3 + cell.shade * 0.4, 0.55 + cell.shade * 0.2, 0.7 + cell.shade * 0.25);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells]);

  useDisposableThreeResources([], []);
  if (featureNames.length === 0) return null;

  const x0 = surface?.x_grid[0] ?? 0;
  const x1 = surface?.x_grid[(surface?.x_grid.length ?? 1) - 1] ?? 1;
  const y0 = surface?.y_grid[0] ?? 0;
  const y1 = surface?.y_grid[(surface?.y_grid.length ?? 1) - 1] ?? 1;

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          3D delegated stage — {overlays?.delegated_estimator ?? 'terminal estimator'}
        </span>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Axis X
          <select
            aria-label="Axis X"
            value={axisX}
            onChange={(e) => setAxisX(Number(e.target.value))}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {featureNames.map((n, i) => (
              <option key={n} value={i}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          Axis Y
          <select
            aria-label="Axis Y"
            value={axisY}
            onChange={(e) => setAxisY(Number(e.target.value))}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            {featureNames.map((n, i) => (
              <option key={n} value={i}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          3D space
          <select
            aria-label="Coordinate space"
            value={coordSpace}
            onChange={(e) => setCoordSpace(e.target.value as 'raw' | 'transformed')}
            className="rounded px-1.5 py-0.5"
            style={{ border: '1px solid var(--border-strong)', background: 'var(--surface-elevated)' }}
          >
            <option value="raw">Raw Domain Units (THB, Years)</option>
            <option value="transformed">Transformed Space (Scaled Z-Score)</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showSurface} onChange={(e) => setShowSurface(e.target.checked)} />
          Surface
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showWireframe} onChange={(e) => setShowWireframe(e.target.checked)} />
          Wireframe
        </label>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing raw-space surface…</span>}
      </div>

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} voxels ready | Z [{telemetry.z_min.toFixed(3)} ..{' '}
          {telemetry.z_max.toFixed(3)}] | {telemetry.compute_ms.toFixed(1)} ms
          {mech ? ` · tunnel ${mech.pipeline_n_steps} steps → ${mech.n_transformed_features} cols` : ''}
        </p>
      )}

      <div className="mt-2">
        {showSurface && surface && res >= 2 ? (
          <div className="flex flex-col gap-1">
            <div
              className="relative w-full overflow-hidden rounded"
              style={{ height: 320, border: '1px solid var(--border-subtle)', background: '#0b0f14' }}
            >
              <Canvas
                dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
                camera={{ position: [5.2, 4.4, 6.4], fov: 42, near: 0.1, far: 100 }}
                gl={{ antialias: true, powerPreference: 'high-performance' }}
              >
                <ambientLight intensity={0.8} />
                <directionalLight position={[6, 8, 4]} intensity={1.15} />
                <instancedMesh ref={meshRef} args={[undefined, undefined, Math.max(cells.length, 1)]} frustumCulled={false}>
                  <boxGeometry args={[1, 1, 1]} />
                  <meshStandardMaterial roughness={0.6} metalness={0.05} wireframe={showWireframe} />
                </instancedMesh>
                <OrbitControls enableDamping={false} />
              </Canvas>
              <span
                className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
                style={{ background: 'rgba(34,211,238,0.12)', color: '#a5f3fc', border: '1px solid rgba(34,211,238,0.4)' }}
              >
                ◆ Pipeline tunnel ({res}×{res} · {coordSpace === 'raw' ? 'raw units' : 'z-score'} · X [
                {x0.toFixed(1)}..{x1.toFixed(1)}] · Y [{y0.toFixed(1)}..{y1.toFixed(1)}])
              </span>
              <span className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]" style={{ color: '#94a3b8' }}>
                X [{pipelineGridToStage(x0, x0, x1).toFixed(1)}..{pipelineGridToStage(x1, x0, x1).toFixed(1)}] · drag to orbit
              </span>
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>Run the surface pass to grow the delegated decision terrain.</p>
        )}
      </div>

      {showTelemetry && surface && (
        <pre
          className="font-mono-tabular mt-2 max-h-48 overflow-auto rounded p-2 text-[11px]"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
        >
          {JSON.stringify(
            {
              visual_type: surface.visual_type,
              resolution: surface.resolution,
              delegated_estimator: surface.extra_overlays?.delegated_estimator,
              coordinate_space: surface.extra_overlays?.coordinate_space,
              telemetry: surface.telemetry,
              z_sample: surface.z_values.slice(0, 8),
            },
            null,
            2,
          )}
        </pre>
      )}
    </div>
  );
}

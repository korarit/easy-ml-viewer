import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { useOnnxDeepSliceStore } from '../store/useOnnxDeepSliceStore';
import { ONNXLatentManifold3D } from '../../stage3d/components/surfaces/ONNXLatentManifold3D';
import { deepPosteriorHeight, deepGridToStage } from '../../stage3d/components/surfaces/onnxDeepSurfaceMath';
import { useDisposableThreeResources } from '../../stage3d/hooks/useDisposableThreeResources';

/**
 * Phase 39 — dual 3D stage router: `[Input Feature Surface 3D]` on the left
 * and `[3D Latent Space PCA Manifold]` on the right. The surface renders the
 * decision terrain from the probed session; the manifold renders the
 * penultimate-layer `(PC1, PC2, PC3)` star-cluster with the active VCR row
 * glowing. Both viewports share one voxel/instanced renderer each
 * (≤ 2 draw calls for geometry) with optional camera synchronization.
 */
export function OnnxDeepStageRouter() {
  const {
    featureNames,
    axisX,
    axisY,
    surface,
    isSurfacing,
    showSurface,
    showWireframe,
    showLatent,
    syncCameras,
    showTrail,
    setAxisX,
    setAxisY,
    setShowSurface,
    setShowWireframe,
    setShowLatent,
    setSyncCameras,
    setShowTrail,
    showTelemetry,
    traces,
  } = useOnnxDeepSliceStore();

  const mech = traces[0]?.extra_mechanics ?? null;
  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const telemetry = surface?.telemetry;

  const leftControls = useRef<OrbitControlsImpl>(null!);
  const rightControls = useRef<OrbitControlsImpl>(null!);

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
        const h = isClassifier ? deepPosteriorHeight(raw) : deepGridToStage(raw, lo, hi) + 1;
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
    const col = new THREE.Color('#818cf8');
    cells.forEach((cell, i) => {
      pos.set(cell.x, cell.y, cell.z);
      scl.set(cell.sx, cell.sy, cell.sz);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      col.set('#818cf8').multiplyScalar(0.45 + cell.shade * 0.75);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells]);

  useDisposableThreeResources([], []);

  const latentFlat = useMemo(() => {
    const sample = overlays?.latent_manifold_sample ?? [];
    const flat = new Float32Array(sample.length * 3 + 3);
    sample.forEach((p, i) => {
      flat[i * 3] = p[0] ?? 0;
      flat[i * 3 + 1] = p[1] ?? 0;
      flat[i * 3 + 2] = p[2] ?? 0;
    });
    const active = mech?.latent_coords_3d ?? [0, 0, 0];
    flat[sample.length * 3] = active[0] ?? 0;
    flat[sample.length * 3 + 1] = active[1] ?? 0;
    flat[sample.length * 3 + 2] = active[2] ?? 0;
    return flat;
  }, [overlays, mech]);
  const latentClasses = useMemo(() => {
    const n = Math.floor(latentFlat.length / 3);
    const arr = new Uint16Array(n);
    const activeCls = typeof mech?.predicted_class === 'number' ? mech.predicted_class : 0;
    if (n > 0) arr[n - 1] = Math.max(0, activeCls);
    return arr;
  }, [latentFlat, mech]);
  const latentTelemetry = useMemo(
    () =>
      (mech?.layer_telemetry ?? []).map((t) => ({
        layerName: t.layer_name,
        meanActivation: t.mean_activation,
        sparsity: t.sparsity,
      })),
    [mech],
  );

  const syncLeftToRight = () => {
    if (!syncCameras) return;
    const l = leftControls.current;
    const r = rightControls.current;
    if (!l || !r) return;
    try {
      r.object.position.copy(l.object.position);
      r.target.copy(l.target);
      r.update();
    } catch {
      /* controls not mounted yet */
    }
  };

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
        <span className="text-[11px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
          Dual 3D stage — input surface ↔ latent manifold
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
          <input type="checkbox" checked={showSurface} onChange={(e) => setShowSurface(e.target.checked)} />
          Surface
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showWireframe} onChange={(e) => setShowWireframe(e.target.checked)} />
          Wireframe
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showLatent} onChange={(e) => setShowLatent(e.target.checked)} />
          Latent manifold
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={syncCameras} onChange={(e) => setSyncCameras(e.target.checked)} />
          Sync cameras
        </label>
        <label className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <input type="checkbox" checked={showTrail} onChange={(e) => setShowTrail(e.target.checked)} />
          Trail
        </label>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing dual 3D stage…</span>}
      </div>

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} voxels ready | Z [{telemetry.z_min.toFixed(3)} ..{' '}
          {telemetry.z_max.toFixed(3)}] | {telemetry.compute_ms.toFixed(1)} ms
          {overlays?.n_probed_layers != null ? ` · ${overlays.n_probed_layers} probed layers` : ''}
          {overlays?.batch_mode ? ` · ${overlays.batch_mode}` : ''}
        </p>
      )}

      <div className="mt-2 grid gap-2 lg:grid-cols-2">
        <div className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
            3D input feature surface
          </span>
          {showSurface && surface && res >= 2 ? (
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
                <OrbitControls ref={leftControls} enableDamping={false} onChange={syncLeftToRight} />
              </Canvas>
              <span
                className="font-mono-tabular pointer-events-none absolute left-2 top-1 rounded px-1.5 py-0.5 text-[10px]"
                style={{ background: 'rgba(129,140,248,0.12)', color: '#c7d2fe', border: '1px solid rgba(129,140,248,0.4)' }}
              >
                ◆ deep surface ({res}×{res} · X [{x0.toFixed(1)}..{x1.toFixed(1)}] · Y [{y0.toFixed(1)}..{y1.toFixed(1)}])
              </span>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>Run the surface pass to grow the deep decision terrain.</p>
          )}
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
            3D latent manifold (penultimate)
          </span>
          {showLatent && latentFlat.length >= 3 ? (
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
                <ONNXLatentManifold3D
                  latentCoords3D={latentFlat}
                  predictedClasses={latentClasses}
                  activeFrameIndex={Math.floor(latentFlat.length / 3) - 1}
                  layerTelemetry={latentTelemetry}
                />
                <OrbitControls ref={rightControls} enableDamping={false} />
              </Canvas>
              <span
                className="font-mono-tabular pointer-events-none absolute left-2 top-1 rounded px-1.5 py-0.5 text-[10px]"
                style={{ background: 'rgba(244,114,182,0.12)', color: '#fbcfe8', border: '1px solid rgba(244,114,182,0.4)' }}
              >
                ◆ latent cluster ({Math.floor(latentFlat.length / 3)} pts · {overlays?.penultimate_layer ?? 'penultimate'})
              </span>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>Run the trace pass to project the latent star-cluster.</p>
          )}
        </div>
      </div>

      {showTrail && mech?.latent_coords_3d && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          (*) Active row in latent space: [{mech.latent_coords_3d.map((v) => v.toFixed(2)).join(', ')}]
        </p>
      )}

      {showTelemetry && surface && (
        <pre
          className="font-mono-tabular mt-2 max-h-48 overflow-auto rounded p-2 text-[11px]"
          style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
        >
          {JSON.stringify(
            {
              visual_type: surface.visual_type,
              resolution: surface.resolution,
              operator_family: surface.extra_overlays?.operator_family,
              node_chain: surface.extra_overlays?.node_chain,
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

import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useOnnxClassicalSliceStore } from '../store/useOnnxClassicalSliceStore';
import { useDisposableThreeResources } from '../../stage3d/hooks/useDisposableThreeResources';

const MODE_META: Record<string, { title: string; accent: string }> = {
  tree_ensemble: { title: 'TreeEnsemble Voxel Terraces', accent: '#34d399' },
  linear: { title: 'Linear Hyperplane / Sigmoid Wave', accent: '#38bdf8' },
  svm: { title: 'RBF Kernel Saddle + SV Pillars', accent: '#e879f9' },
};

/**
 * Phase 38 — adaptive 3D stage router. Switches terrain shading/labels by
 * `detected_operator_family` (TreeEnsemble → stepped voxels, Linear →
 * hyperplane/sigmoid, SVM → kernel saddle) while sharing one InstancedMesh
 * voxel renderer (≤ 1 draw call) and split-plane threshold markers.
 */
export function OnnxClassicalStageRouter() {
  const {
    featureNames,
    axisX,
    axisY,
    surface,
    isSurfacing,
    showSurface,
    showWireframe,
    showSplitPlanes,
    setAxisX,
    setAxisY,
    setShowSurface,
    setShowWireframe,
    setShowSplitPlanes,
    showTelemetry,
    traces,
  } = useOnnxClassicalSliceStore();

  const mech = traces[0]?.extra_mechanics ?? null;
  const overlays = surface?.extra_overlays;
  const family = overlays?.operator_family ?? mech?.operator_family ?? 'unknown';
  const meta = MODE_META[family] ?? { title: 'Auto-Adaptive ONNX Stage', accent: '#94a3b8' };
  const res = surface?.resolution ?? 0;
  const telemetry = surface?.telemetry;

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
        let h: number;
        if (isClassifier) h = Math.min(2, Math.max(0.02, raw * 2));
        else {
          const span = Math.max(hi - lo, 1e-9);
          h = Math.min(2, Math.max(0.02, ((raw - lo) / span) * 2));
        }
        // Stepped look for tree ensembles: quantize height into 8 terraces.
        const stepped = family === 'tree_ensemble' ? Math.round(h * 4) / 4 : h;
        out.push({
          x: -2.5 + (c + 0.5) * cell,
          y: Math.max(stepped, 0.02) / 2,
          z: -2.5 + (r + 0.5) * cell,
          sx: cell * 0.92,
          sy: Math.max(stepped, 0.02),
          sz: cell * 0.92,
          shade: isClassifier ? Math.min(1, Math.max(0, raw)) : h / 2,
        });
      }
    }
    return out;
  }, [surface, res, overlays, telemetry, family]);

  const meshRef = useRef<THREE.InstancedMesh>(null!);
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || cells.length === 0) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    const col = new THREE.Color(meta.accent);
    cells.forEach((cell, i) => {
      pos.set(cell.x, cell.y, cell.z);
      scl.set(cell.sx, cell.sy, cell.sz);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      const k = cell.shade;
      col.set(meta.accent).multiplyScalar(0.45 + k * 0.75);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells, meta.accent]);

  useDisposableThreeResources([], []);
  if (featureNames.length === 0) return null;

  const x0 = surface?.x_grid[0] ?? 0;
  const x1 = surface?.x_grid[(surface?.x_grid.length ?? 1) - 1] ?? 1;
  const y0 = surface?.y_grid[0] ?? 0;
  const y1 = surface?.y_grid[(surface?.y_grid.length ?? 1) - 1] ?? 1;
  const splits = overlays?.split_thresholds;
  const shownSplits = showSplitPlanes && splits && (family === 'tree_ensemble' || family === 'linear');

  return (
    <div
      className="rounded-lg px-4 py-3 text-xs"
      style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[11px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
          3D adaptive stage — {meta.title}
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
          <input type="checkbox" checked={showSplitPlanes} onChange={(e) => setShowSplitPlanes(e.target.checked)} />
          Split planes
        </label>
        {isSurfacing && <span style={{ color: 'var(--ml-primary)' }}>Computing adaptive surface…</span>}
      </div>

      {telemetry && (
        <p className="font-mono-tabular mt-2" style={{ color: 'var(--text-secondary)' }}>
          Status: {telemetry.grid_points.toLocaleString()} voxels ready | Z [{telemetry.z_min.toFixed(3)} ..{' '}
          {telemetry.z_max.toFixed(3)}] | {telemetry.compute_ms.toFixed(1)} ms
          {overlays?.n_trees != null ? ` · ${overlays.n_trees} trees / ${overlays.n_nodes ?? '?'} nodes` : ''}
          {overlays?.kernel_type ? ` · kernel ${overlays.kernel_type}` : ''}
          {overlays?.zipmap_bypass_active === false ? ' · ZipMap fallback unpacker' : ''}
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
                className="font-mono-tabular pointer-events-none absolute left-2 top-1 rounded px-1.5 py-0.5 text-[10px]"
                style={{ background: 'rgba(52,211,153,0.12)', color: '#a7f3d0', border: '1px solid rgba(52,211,153,0.4)' }}
              >
                ◆ {meta.title} ({res}×{res} · X [{x0.toFixed(1)}..{x1.toFixed(1)}] · Y [{y0.toFixed(1)}..{y1.toFixed(1)}])
              </span>
              {shownSplits && splits && (splits.axis_x_thresholds.length > 0 || splits.axis_y_thresholds.length > 0) && (
                <span
                  className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 max-w-[95%] truncate text-[10px]"
                  style={{ color: '#94a3b8' }}
                  title={`X splits: ${splits.axis_x_thresholds.join(', ')} | Y splits: ${splits.axis_y_thresholds.join(', ')}`}
                >
                  split planes X[{splits.axis_x_thresholds.length}] Y[{splits.axis_y_thresholds.length}] · drag to orbit
                </span>
              )}
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>Run the surface pass to grow the adaptive ONNX terrain.</p>
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

import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useSklearnGbmSliceStore } from '../../../model_engine/store/useSklearnGbmSliceStore';
import {
  gbmPosteriorHeight,
  gbmSigmoid,
  normalizeGbmHeight,
  spiralColor,
} from './sklearnGbmSurfaceMath';

/**
 * Phase 31 — 3D boosted terrain + residual spiral ladder.
 *
 * Own tiny WebGL canvas: fine-grained boosted terraces (single
 * InstancedMesh), a vertical residual spiral ladder at the query
 * position whose rungs track the top residual stages (color = push/pull
 * direction of η·h_m, height = cumulative raw score), and the live
 * query ball.
 */

export interface SklearnGbmResidualSpiral3DProps {
  liveFeatures?: number[] | null;
}

export function SklearnGbmResidualSpiral3D(props: SklearnGbmResidualSpiral3DProps) {
  const surface = useSklearnGbmSliceStore((s) => s.surface);
  const showSpiral = useSklearnGbmSliceStore((s) => s.showSpiral);
  const showWireframe = useSklearnGbmSliceStore((s) => s.showWireframe);
  const axisX = useSklearnGbmSliceStore((s) => s.axisX);
  const axisY = useSklearnGbmSliceStore((s) => s.axisY);
  const traces = useSklearnGbmSliceStore((s) => s.traces);
  const selectedStage = useSklearnGbmSliceStore((s) => s.selectedStage);
  const terrainRef = useRef<THREE.InstancedMesh>(null!);
  const rungsRef = useRef<THREE.InstancedMesh>(null!);

  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const isRegressor = overlays?.is_regressor === true;
  const zRange = (overlays?.regression_z_range ?? [0, 1]) as [number, number];
  const mech = traces[0]?.extra_mechanics ?? null;
  const stages = (mech?.top_residual_stages ?? []) as Array<{
    stage: number;
    leaf_index: number;
    delta_raw: number;
    cumulative_raw: number;
    cumulative_pred: number | null;
  }>;
  const baseRaw = (traces[0]?.base_value ?? 0) as number;
  const toH = (raw: number, pred: number | null): number => {
    if (isRegressor) return normalizeGbmHeight(zRange[0], zRange[1], raw);
    const p = pred ?? gbmSigmoid(raw);
    return gbmPosteriorHeight(p);
  };

  const cells = useMemo(() => {
    if (!surface || res < 2 || surface.z_values.length < res * res) return [];
    const out: Array<{ x: number; y: number; z: number; sx: number; sy: number; sz: number; shade: number }> = [];
    const cell = 5 / res;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const raw = surface.z_values[idx] ?? 0;
        const h = isRegressor ? normalizeGbmHeight(zRange[0], zRange[1], raw) : gbmPosteriorHeight(raw);
        out.push({
          x: -2.5 + (c + 0.5) * cell,
          y: Math.max(h, 0.02) / 2,
          z: -2.5 + (r + 0.5) * cell,
          sx: cell * 0.92,
          sy: Math.max(h, 0.02),
          sz: cell * 0.92,
          shade: isRegressor ? h / 2 : Math.min(1, Math.max(0, raw)),
        });
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surface, res, isRegressor, zRange]);

  const rungs = useMemo(() => {
    if (!showSpiral || stages.length === 0) return [];
    return stages.map((st) => {
      const hPrev = toH(st.cumulative_raw - st.delta_raw, null);
      const hCur = toH(st.cumulative_raw, st.cumulative_pred);
      const [cr, cg, cb] = spiralColor(st.delta_raw);
      return {
        stage: st.stage,
        leaf: st.leaf_index,
        y: (hPrev + hCur) / 2,
        sy: Math.max(Math.abs(hCur - hPrev), 0.035),
        w: 0.3,
        color: [cr, cg, cb] as [number, number, number],
        dimmed: st.stage > selectedStage,
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showSpiral, stages, baseRaw, isRegressor, selectedStage]);

  useEffect(() => {
    const mesh = terrainRef.current;
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
      // Teal/emerald boosted-terrain ramp (GBM residual identity).
      col.setRGB(0.15 + cell.shade * 0.35, 0.45 + cell.shade * 0.35, 0.5 + cell.shade * 0.35);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells]);

  useEffect(() => {
    const mesh = rungsRef.current;
    if (!mesh || rungs.length === 0) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    const col = new THREE.Color();
    rungs.forEach((r, i) => {
      pos.set(0, r.y, 0);
      scl.set(r.w, r.sy, r.w);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      col.setRGB(...r.color);
      if (r.dimmed) col.multiplyScalar(0.35);
      mesh.setColorAt(i, col);
    });
    mesh.count = rungs.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [rungs]);

  const queryPos = useMemo(() => {
    const live = props.liveFeatures;
    if (!surface || !live || live.length === 0) return new THREE.Vector3(0, 0.1, 0);
    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const x0 = xs[0];
    const y0 = ys[0];
    const spanX = xs[xs.length - 1] - x0 || 1;
    const spanY = ys[ys.length - 1] - y0 || 1;
    const lx = live[axisX];
    const ly = live[axisY];
    if (!Number.isFinite(lx) || !Number.isFinite(ly)) return new THREE.Vector3(0, 0.1, 0);
    return new THREE.Vector3(
      Math.min(2.5, Math.max(-2.5, ((lx - x0) / spanX) * 5 - 2.5)),
      0.12,
      Math.min(2.5, Math.max(-2.5, ((ly - y0) / spanY) * 5 - 2.5)),
    );
  }, [surface, props.liveFeatures, axisX, axisY]);

  const ballH = useMemo(() => {
    const last = stages.length > 0 ? stages[stages.length - 1] : null;
    const h = last ? toH(last.cumulative_raw, last.cumulative_pred) : toH(baseRaw, null);
    return Math.max(h, 0.12);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stages, baseRaw, isRegressor]);

  const ballMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#fbbf24'),
        emissive: new THREE.Color('#92400e'),
        emissiveIntensity: 0.9,
      }),
    [],
  );
  const poleMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#5eead4'),
        transparent: true,
        opacity: 0.5,
      }),
    [],
  );
  const ringGeo = useMemo(() => {
    const curve = new THREE.EllipseCurve(0, 0, 0.22, 0.22, 0, Math.PI * 2);
    return new THREE.BufferGeometry().setFromPoints(curve.getPoints(48).map((p) => new THREE.Vector3(p.x, 0, p.y)));
  }, []);
  const ringMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#fde68a'),
        transparent: true,
        opacity: 0.95,
      }),
    [],
  );
  useDisposableThreeResources([ringGeo], [ballMaterial, poleMaterial, ringMaterial]);

  if (!surface || res < 2) return null;
  const total = overlays?.n_stages_total ?? overlays?.n_estimators_actual ?? stages.length;
  const baseH = Math.max(toH(baseRaw, null), 0.02);

  return (
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
          <instancedMesh ref={terrainRef} args={[undefined, undefined, Math.max(cells.length, 1)]} frustumCulled={false}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial roughness={0.6} metalness={0.05} wireframe={showWireframe} />
          </instancedMesh>
          {rungs.length > 0 && (
            <group position={[queryPos.x, 0, queryPos.z]}>
              <mesh position={[0, (baseH + ballH) / 2, 0]} material={poleMaterial}>
                <boxGeometry args={[0.03, Math.max(ballH - baseH, 0.05), 0.03]} />
              </mesh>
              <instancedMesh ref={rungsRef} args={[undefined, undefined, Math.max(rungs.length, 1)]} frustumCulled={false}>
                <boxGeometry args={[1, 1, 1]} />
                <meshStandardMaterial roughness={0.35} metalness={0.15} emissiveIntensity={0.35} />
              </instancedMesh>
            </group>
          )}
          <lineLoop geometry={ringGeo} material={ringMaterial} position={[queryPos.x, 0.03, queryPos.z]} />
          <mesh position={[queryPos.x, ballH, queryPos.z]} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(45,212,191,0.12)', color: '#5eead4', border: '1px solid rgba(45,212,191,0.4)' }}
        >
          ▲ Boosted terrain + residual spiral ({rungs.length} rungs / {total} stages · 2 draw calls)
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          cyan = push up · orange = pull down · rungs dim past scrubber · drag to orbit
        </span>
      </div>
    </div>
  );
}

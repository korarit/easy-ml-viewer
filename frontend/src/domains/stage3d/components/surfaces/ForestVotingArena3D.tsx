import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useForestSliceStore } from '../../../model_engine/store/useForestSliceStore';
import {
  forestPosteriorHeight,
  normalizeForestHeight,
  pillarHeight,
} from './forestSurfaceMath';

/**
 * Phase 28 — 3D forest voting arena + ensemble smooth terrain.
 *
 * Own tiny WebGL canvas: fine-grained ensemble terraces drawn with a
 * single InstancedMesh (piecewise-constant, no smoothing), plus the
 * crystal voting ring (up to 100 pillars, one InstancedMesh draw call)
 * whose heights track per-tree predictions for the active frame, and
 * the live query ball.
 */

export interface ForestVotingArena3DProps {
  liveFeatures?: number[] | null;
}

export function ForestVotingArena3D(props: ForestVotingArena3DProps) {
  const surface = useForestSliceStore((s) => s.surface);
  const showVotes = useForestSliceStore((s) => s.showVotes);
  const showWireframe = useForestSliceStore((s) => s.showWireframe);
  const axisX = useForestSliceStore((s) => s.axisX);
  const axisY = useForestSliceStore((s) => s.axisY);
  const traces = useForestSliceStore((s) => s.traces);
  const selectedTree = useForestSliceStore((s) => s.selectedTree);
  const terrainRef = useRef<THREE.InstancedMesh>(null!);
  const votesRef = useRef<THREE.InstancedMesh>(null!);

  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const isRegressor = overlays?.is_regressor === true;
  const zRange = (overlays?.regression_z_range ?? [0, 1]) as [number, number];
  const mech = traces[0]?.extra_mechanics ?? null;
  const treePreds = (mech?.tree_predictions_sample ?? []) as number[];

  const cells = useMemo(() => {
    if (!surface || res < 2 || surface.z_values.length < res * res) return [];
    const out: Array<{ x: number; y: number; z: number; sx: number; sy: number; sz: number; shade: number }> = [];
    const cell = 5 / res;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const raw = surface.z_values[idx] ?? 0;
        const h = isRegressor
          ? normalizeForestHeight(zRange[0], zRange[1], raw)
          : forestPosteriorHeight(raw);
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
  }, [surface, res, isRegressor]);

  const pillars = useMemo(() => {
    if (!showVotes || treePreds.length === 0) return [];
    const n = Math.min(treePreds.length, 100);
    const R = 3.4;
    const zMin = Math.min(...treePreds);
    const zMax = Math.max(...treePreds);
    return Array.from({ length: n }, (_, k) => {
      const ang = (k / n) * Math.PI * 2;
      const h = pillarHeight(treePreds[k] ?? 0, zMin, zMax, isRegressor);
      return {
        x: Math.cos(ang) * R,
        z: Math.sin(ang) * R,
        h: Math.max(h, 0.05),
        y: Math.max(h, 0.05) / 2,
        pos: (treePreds[k] ?? 0) >= 0.5,
        selected: k === selectedTree,
      };
    });
  }, [showVotes, treePreds, selectedTree, isRegressor]);

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
      // Emerald shade ramp: deep teal (low) -> bright emerald (high).
      col.setRGB(0.04 + cell.shade * 0.1, 0.32 + cell.shade * 0.5, 0.3 + cell.shade * 0.42);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells]);

  useEffect(() => {
    const mesh = votesRef.current;
    if (!mesh || pillars.length === 0) return;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scl = new THREE.Vector3();
    const col = new THREE.Color();
    pillars.forEach((p, i) => {
      pos.set(p.x, p.y, p.z);
      scl.set(p.selected ? 0.16 : 0.1, p.h, p.selected ? 0.16 : 0.1);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      if (p.selected) col.setRGB(1.0, 0.85, 0.3);
      else if (p.pos) col.setRGB(0.13, 0.83, 0.93);
      else col.setRGB(0.98, 0.62, 0.12);
      mesh.setColorAt(i, col);
    });
    mesh.count = pillars.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [pillars]);

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

  const ballMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#34d399'),
        emissive: new THREE.Color('#065f46'),
        emissiveIntensity: 0.9,
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
        color: new THREE.Color('#a7f3d0'),
        transparent: true,
        opacity: 0.95,
      }),
    [],
  );
  useDisposableThreeResources([ringGeo], [ballMaterial, ringMaterial]);

  if (!surface || res < 2) return null;
  const total = overlays?.n_estimators_total ?? treePreds.length;

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
          {pillars.length > 0 && (
            <instancedMesh ref={votesRef} args={[undefined, undefined, Math.max(pillars.length, 1)]} frustumCulled={false}>
              <boxGeometry args={[1, 1, 1]} />
              <meshStandardMaterial roughness={0.35} metalness={0.15} emissiveIntensity={0.45} />
            </instancedMesh>
          )}
          <lineLoop geometry={ringGeo} material={ringMaterial} position={[queryPos.x, 0.03, queryPos.z]} />
          <mesh position={queryPos} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(52,211,153,0.12)', color: '#6ee7b7', border: '1px solid rgba(52,211,153,0.4)' }}
        >
          ▲ Forest consensus terraces + voting ring ({pillars.length}/{total} pillars · 2 draw calls)
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          cyan = pos vote · amber = neg vote · gold = selected tree · drag to orbit
        </span>
      </div>
    </div>
  );
}

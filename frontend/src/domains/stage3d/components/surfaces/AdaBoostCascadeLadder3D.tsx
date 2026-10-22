import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useAdaBoostSliceStore } from '../../../model_engine/store/useAdaBoostSliceStore';
import {
  adaBoostPosteriorHeight,
  gridToStage,
  normalizeAdaBoostHeight,
  rungColor,
  rungWeight,
} from './adaboostSurfaceMath';

/**
 * Phase 30 — 3D plaid terraces + sequential cascade ladder + stump curtains.
 *
 * Own tiny WebGL canvas: fine-grained plaid terraces (single
 * InstancedMesh), a vertical cascade ladder at the query position whose
 * rungs track the top influential weak learners (color = push/pull
 * direction, thickness = learner weight α), translucent stump split
 * curtain planes for the selected stage, and the live query ball.
 */

export interface AdaBoostCascadeLadder3DProps {
  liveFeatures?: number[] | null;
}

export function AdaBoostCascadeLadder3D(props: AdaBoostCascadeLadder3DProps) {
  const surface = useAdaBoostSliceStore((s) => s.surface);
  const showLadder = useAdaBoostSliceStore((s) => s.showLadder);
  const showCurtains = useAdaBoostSliceStore((s) => s.showCurtains);
  const showWireframe = useAdaBoostSliceStore((s) => s.showWireframe);
  const axisX = useAdaBoostSliceStore((s) => s.axisX);
  const axisY = useAdaBoostSliceStore((s) => s.axisY);
  const traces = useAdaBoostSliceStore((s) => s.traces);
  const selectedStage = useAdaBoostSliceStore((s) => s.selectedStage);
  const terrainRef = useRef<THREE.InstancedMesh>(null!);
  const rungsRef = useRef<THREE.InstancedMesh>(null!);

  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const isRegressor = overlays?.is_regressor === true;
  const zRange = (overlays?.regression_z_range ?? [0, 1]) as [number, number];
  const mech = traces[0]?.extra_mechanics ?? null;
  const learners = (mech?.top_influential_learners ?? []) as Array<{
    stage: number;
    weight: number;
    error: number;
    feature_idx: number;
    threshold: number;
    delta: number;
    cumulative: number;
  }>;
  const baseValue = (traces[0]?.base_value ?? 0.5) as number;
  const toH = (v: number): number =>
    isRegressor ? normalizeAdaBoostHeight(zRange[0], zRange[1], v) : adaBoostPosteriorHeight(v);

  const cells = useMemo(() => {
    if (!surface || res < 2 || surface.z_values.length < res * res) return [];
    const out: Array<{ x: number; y: number; z: number; sx: number; sy: number; sz: number; shade: number }> = [];
    const cell = 5 / res;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const raw = surface.z_values[idx] ?? 0;
        const h = toH(raw);
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
    if (!showLadder || learners.length === 0) return [];
    const maxAlpha = Math.max(1e-9, ...learners.map((l) => Math.abs(l.weight)));
    let prev = baseValue;
    return learners.map((l) => {
      const hPrev = toH(prev);
      const hCur = toH(l.cumulative);
      prev = l.cumulative;
      const [cr, cg, cb] = rungColor(l.delta);
      return {
        stage: l.stage,
        y: (hPrev + hCur) / 2,
        sy: Math.max(Math.abs(hCur - hPrev), 0.035),
        w: 0.22 + rungWeight(Math.abs(l.weight), maxAlpha) * 0.22,
        color: [cr, cg, cb] as [number, number, number],
        dimmed: l.stage > selectedStage,
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showLadder, learners, baseValue, isRegressor, selectedStage]);

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
      // Indigo/violet plaid ramp (boosting stumps identity).
      col.setRGB(0.25 + cell.shade * 0.45, 0.28 + cell.shade * 0.3, 0.55 + cell.shade * 0.4);
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
      // Ladder floats above the query ball position (set via group below).
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

  const curtains = useMemo(() => {
    if (!surface || !showCurtains) return [];
    const splits = overlays?.stump_splits ?? [];
    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const lo_x = xs[0];
    const hi_x = xs[xs.length - 1];
    const lo_y = ys[0];
    const hi_y = ys[ys.length - 1];
    return splits
      .filter((s) => s.feature_idx === axisX || s.feature_idx === axisY)
      .slice(0, 12)
      .map((s) => ({
        stage: s.stage,
        alongX: s.feature_idx === axisX,
        at: gridToStage(s.threshold, s.feature_idx === axisX ? lo_x : lo_y, s.feature_idx === axisX ? hi_x : hi_y),
        active: s.stage === Math.min(selectedStage, Math.max(0, (overlays?.n_estimators_actual ?? 1) - 1)),
      }));
  }, [surface, overlays, showCurtains, axisX, axisY, selectedStage]);

  const ballH = useMemo(() => {
    const final = learners.length > 0 ? learners[learners.length - 1].cumulative : baseValue;
    return Math.max(toH(final), 0.12);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [learners, baseValue, isRegressor]);

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
        color: new THREE.Color('#a5b4fc'),
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
  const curtainGeos = useMemo(
    () =>
      curtains.map((c) => {
        const g = new THREE.PlaneGeometry(5, 2.2);
        g.translate(0, 1.1, 0);
        return g;
      }),
    [curtains],
  );
  const curtainMaterials = useMemo(
    () =>
      curtains.map(
        (c) =>
          new THREE.MeshBasicMaterial({
            color: new THREE.Color(c.active ? '#fbbf24' : '#8b5cf6'),
            transparent: true,
            opacity: c.active ? 0.28 : 0.12,
            side: THREE.DoubleSide,
            depthWrite: false,
          }),
      ),
    [curtains],
  );
  useDisposableThreeResources(
    [...curtainGeos, ringGeo],
    [ballMaterial, poleMaterial, ringMaterial, ...curtainMaterials],
  );

  if (!surface || res < 2) return null;
  const total = overlays?.n_estimators_actual ?? learners.length;
  const baseH = Math.max(toH(baseValue), 0.02);

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
          {curtains.map((c, i) => (
            <mesh
              key={`curtain-${c.stage}-${i}`}
              geometry={curtainGeos[i]}
              material={curtainMaterials[i]}
              position={c.alongX ? [c.at, 0, 0] : [0, 0, c.at]}
              rotation={c.alongX ? [0, Math.PI / 2, 0] : [0, 0, 0]}
            />
          ))}
          <lineLoop geometry={ringGeo} material={ringMaterial} position={[queryPos.x, 0.03, queryPos.z]} />
          <mesh position={[queryPos.x, ballH, queryPos.z]} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(139,92,246,0.12)', color: '#c4b5fd', border: '1px solid rgba(139,92,246,0.4)' }}
        >
          ▲ Plaid terraces + cascade ladder ({rungs.length} rungs / {total} learners · 2 draw calls)
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          cyan = push up · rose = pull down · gold curtain = selected stage · drag to orbit
        </span>
      </div>
    </div>
  );
}

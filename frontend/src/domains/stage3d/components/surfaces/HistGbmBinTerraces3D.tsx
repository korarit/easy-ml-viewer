import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useHistGbmSliceStore } from '../../../model_engine/store/useHistGbmSliceStore';
import {
  downsampleBinEdges,
  histGridToStage,
  histPosteriorHeight,
  histSigmoid,
  histSpiralColor,
  normalizeHistHeight,
} from './histGbmSurfaceMath';

/**
 * Phase 32 — 3D quantized bin terraces + residual spiral + NaN portal.
 *
 * Own tiny WebGL canvas: boosted terraces (single InstancedMesh),
 * glowing histogram bin-grid lines from `axis_bin_thresholds`, a
 * vertical residual spiral ladder at the query position tracking the
 * top residual stages, and a violet/gold NaN routing portal ring when
 * the traced row carries missing values.
 */

export interface HistGbmBinTerraces3DProps {
  liveFeatures?: number[] | null;
}

export function HistGbmBinTerraces3D(props: HistGbmBinTerraces3DProps) {
  const surface = useHistGbmSliceStore((s) => s.surface);
  const showSpiral = useHistGbmSliceStore((s) => s.showSpiral);
  const showBinGrid = useHistGbmSliceStore((s) => s.showBinGrid);
  const showWireframe = useHistGbmSliceStore((s) => s.showWireframe);
  const axisX = useHistGbmSliceStore((s) => s.axisX);
  const axisY = useHistGbmSliceStore((s) => s.axisY);
  const traces = useHistGbmSliceStore((s) => s.traces);
  const selectedStage = useHistGbmSliceStore((s) => s.selectedStage);
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
    nan_routed_feature_idx: number | null;
  }>;
  const hasNan = mech?.has_missing_values === true;
  const baseRaw = (traces[0]?.base_value ?? 0) as number;
  const toH = (raw: number, pred: number | null): number => {
    if (isRegressor) return normalizeHistHeight(zRange[0], zRange[1], raw);
    const p = pred ?? histSigmoid(raw);
    return histPosteriorHeight(p);
  };

  const cells = useMemo(() => {
    if (!surface || res < 2 || surface.z_values.length < res * res) return [];
    const out: Array<{ x: number; y: number; z: number; sx: number; sy: number; sz: number; shade: number }> = [];
    const cell = 5 / res;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const raw = surface.z_values[idx] ?? 0;
        const h = isRegressor ? normalizeHistHeight(zRange[0], zRange[1], raw) : histPosteriorHeight(raw);
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
      const [cr, cg, cb] = histSpiralColor(st.delta_raw);
      return {
        stage: st.stage,
        leaf: st.leaf_index,
        y: (hPrev + hCur) / 2,
        sy: Math.max(Math.abs(hCur - hPrev), 0.035),
        w: 0.3,
        color: [cr, cg, cb] as [number, number, number],
        dimmed: st.stage > selectedStage,
        nanRouted: st.nan_routed_feature_idx != null,
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
      // Amber/teal quantized-terraces ramp (HistGBM bin identity).
      col.setRGB(0.5 + cell.shade * 0.4, 0.42 + cell.shade * 0.3, 0.28 + cell.shade * 0.3);
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

  const binLines = useMemo(() => {
    if (!surface || !showBinGrid) return { x: [] as number[], y: [] as number[] };
    const raw = overlays?.axis_bin_thresholds as { x: number[]; y: number[] } | undefined;
    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const x0 = xs[0];
    const x1 = xs[xs.length - 1];
    const y0 = ys[0];
    const y1 = ys[ys.length - 1];
    const mapX = (v: number) => histGridToStage(v, x0, x1);
    const mapY = (v: number) => histGridToStage(v, y0, y1);
    return {
      x: downsampleBinEdges(raw?.x ?? [], 64)
        .map(mapX)
        .filter((v) => Math.abs(v) <= 2.5),
      y: downsampleBinEdges(raw?.y ?? [], 64)
        .map(mapY)
        .filter((v) => Math.abs(v) <= 2.5),
    };
  }, [surface, overlays, showBinGrid]);

  const ballH = useMemo(() => {
    const last = stages.length > 0 ? stages[stages.length - 1] : null;
    const h = last ? toH(last.cumulative_raw, last.cumulative_pred) : toH(baseRaw, null);
    return Math.max(h, 0.12);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stages, baseRaw, isRegressor]);

  const ballMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(hasNan ? '#c4b5fd' : '#fbbf24'),
        emissive: new THREE.Color(hasNan ? '#6d28d9' : '#92400e'),
        emissiveIntensity: 0.9,
      }),
    [hasNan],
  );
  const poleMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#2dd4bf'),
        transparent: true,
        opacity: 0.5,
      }),
    [],
  );
  const ringGeo = useMemo(() => {
    const curve = new THREE.EllipseCurve(0, 0, 0.22, 0.22, 0, Math.PI * 2);
    return new THREE.BufferGeometry().setFromPoints(curve.getPoints(48).map((p) => new THREE.Vector3(p.x, 0, p.y)));
  }, []);
  const portalGeo = useMemo(() => {
    const curve = new THREE.EllipseCurve(0, 0, 0.34, 0.34, 0, Math.PI * 2);
    return new THREE.BufferGeometry().setFromPoints(curve.getPoints(64).map((p) => new THREE.Vector3(p.x, 0, p.y)));
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
  const portalMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#a78bfa'),
        transparent: true,
        opacity: 0.9,
      }),
    [],
  );
  const gridGeos = useMemo(
    () => [...binLines.x, ...binLines.y].map(() => new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-2.5, 0, 0), new THREE.Vector3(2.5, 0, 0)])),
    [binLines],
  );
  const gridMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#f59e0b'),
        transparent: true,
        opacity: 0.35,
      }),
    [],
  );
  useDisposableThreeResources(
    [...gridGeos, ringGeo, portalGeo],
    [ballMaterial, poleMaterial, ringMaterial, portalMaterial, gridMaterial],
  );

  if (!surface || res < 2) return null;
  const total = overlays?.n_iter_total ?? overlays?.n_estimators_actual ?? stages.length;
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
          {showBinGrid &&
            binLines.x.map((at, i) => (
              <lineSegments key={`binx-${i}`} geometry={gridGeos[i]} material={gridMaterial} position={[0, 0.02, at]} />
            ))}
          {showBinGrid &&
            binLines.y.map((at, i) => (
              <lineSegments
                key={`biny-${i}`}
                geometry={gridGeos[binLines.x.length + i]}
                material={gridMaterial}
                position={[at, 0.02, 0]}
                rotation={[0, Math.PI / 2, 0]}
              />
            ))}
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
          {hasNan && (
            <lineLoop geometry={portalGeo} material={portalMaterial} position={[queryPos.x, ballH, queryPos.z]} />
          )}
          <mesh position={[queryPos.x, ballH, queryPos.z]} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(245,158,11,0.12)', color: '#fbbf24', border: '1px solid rgba(245,158,11,0.4)' }}
        >
          ▲ Quantized bin terraces ({binLines.x.length + binLines.y.length} grid lines · {rungs.length} rungs / {total}{' '}
          iters)
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          amber = bin edges · teal = push up · orange = pull down{hasNan ? ' · ◆ NaN portal: missing routed' : ''} ·
          drag to orbit
        </span>
      </div>
    </div>
  );
}

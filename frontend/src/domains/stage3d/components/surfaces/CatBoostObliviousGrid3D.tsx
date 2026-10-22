import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useCatBoostSliceStore } from '../../../model_engine/store/useCatBoostSliceStore';
import {
  catboostForceColor,
  catboostForceLength,
  catboostGridToStage,
  catboostPosteriorHeight,
  normalizeCatboostHeight,
} from './catboostSurfaceMath';

export interface CatBoostObliviousGrid3DProps {
  liveFeatures?: Array<number | string> | null;
}

/**
 * Phase 35 — 3D symmetric oblivious hypercube grid + C++ SHAP force vectors.
 *
 * Stepped terraces with cyan/magenta symmetric-grid laser planes along the
 * active X/Y splits, categorical tick rails on CAT axes, TreeSHAP force
 * arrows at the query ball and the live query ball.
 */
export function CatBoostObliviousGrid3D(props: CatBoostObliviousGrid3DProps) {
  const surface = useCatBoostSliceStore((s) => s.surface);
  const showForceVectors = useCatBoostSliceStore((s) => s.showForceVectors);
  const showSplitLines = useCatBoostSliceStore((s) => s.showSplitLines);
  const showCategoricalTicks = useCatBoostSliceStore((s) => s.showCategoricalTicks);
  const showWireframe = useCatBoostSliceStore((s) => s.showWireframe);
  const axisX = useCatBoostSliceStore((s) => s.axisX);
  const axisY = useCatBoostSliceStore((s) => s.axisY);
  const traces = useCatBoostSliceStore((s) => s.traces);
  const selectedRound = useCatBoostSliceStore((s) => s.selectedRound);
  const terrainRef = useRef<THREE.InstancedMesh>(null!);

  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const isRegressor = overlays?.is_regressor === true;
  const zRange = (overlays?.regression_z_range ?? [0, 1]) as [number, number];
  const mech = traces[0]?.extra_mechanics ?? null;
  const contribs = (traces[0]?.feature_contributions ?? {}) as Record<string, number>;
  const featureNames = useCatBoostSliceStore((s) => s.featureNames);
  const toH = (raw: number): number =>
    isRegressor ? normalizeCatboostHeight(zRange[0], zRange[1], raw) : catboostPosteriorHeight(raw);

  const shapX = contribs[featureNames[axisX] ?? ''] ?? 0;
  const shapY = contribs[featureNames[axisY] ?? ''] ?? 0;

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
      // Symmetric oblivious identity: teal terraces, deep cells glow magenta.
      col.setRGB(0.2 + cell.shade * 0.45, 0.55 + cell.shade * 0.2, 0.6 + cell.shade * 0.3);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells]);

  const queryPos = useMemo(() => {
    const live = props.liveFeatures;
    if (!surface || !live || live.length === 0) return new THREE.Vector3(0, 0.1, 0);
    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const lx = live[axisX];
    const ly = live[axisY];
    const axisXCategorical = overlays?.axis_x_is_categorical === true;
    const axisYCategorical = overlays?.axis_y_is_categorical === true;
    const qx = typeof lx === 'number' && Number.isFinite(lx) && !axisXCategorical
      ? Math.min(2.5, Math.max(-2.5, ((lx - xs[0]) / ((xs[xs.length - 1] - xs[0]) || 1)) * 5 - 2.5))
      : 0;
    const qz = typeof ly === 'number' && Number.isFinite(ly) && !axisYCategorical
      ? Math.min(2.5, Math.max(-2.5, ((ly - ys[0]) / ((ys[ys.length - 1] - ys[0]) || 1)) * 5 - 2.5))
      : 0;
    return new THREE.Vector3(qx, 0.12, qz);
  }, [surface, props.liveFeatures, axisX, axisY, overlays]);

  const splitLines = useMemo(() => {
    if (!surface || !showSplitLines) return { x: [] as number[], y: [] as number[] };
    const lines = (overlays?.split_lines_sample ?? []) as Array<{
      feature_idx: number;
      threshold: number | null;
    }>;
    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const x0 = xs[0];
    const x1 = xs[xs.length - 1];
    const y0 = ys[0];
    const y1 = ys[ys.length - 1];
    const seenX = new Set<string>();
    const seenY = new Set<string>();
    const outX: number[] = [];
    const outY: number[] = [];
    for (const l of lines) {
      if (typeof l.threshold !== 'number') continue;
      if (l.feature_idx === axisX) {
        const at = catboostGridToStage(l.threshold, x0, x1);
        const key = at.toFixed(2);
        if (Math.abs(at) <= 2.5 && !seenX.has(key) && outX.length < 24) {
          seenX.add(key);
          outX.push(at);
        }
      } else if (l.feature_idx === axisY) {
        const at = catboostGridToStage(l.threshold, y0, y1);
        const key = at.toFixed(2);
        if (Math.abs(at) <= 2.5 && !seenY.has(key) && outY.length < 24) {
          seenY.add(key);
          outY.push(at);
        }
      }
    }
    return { x: outX, y: outY };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surface, overlays, showSplitLines, axisX, axisY]);

  const ballH = useMemo(() => {
    const staged = (mech?.staged_trajectory ?? []) as number[];
    const last = staged.length > 0 ? staged[staged.length - 1] : null;
    const base = (traces[0]?.base_value ?? 0) as number;
    const h =
      last != null
        ? isRegressor
          ? normalizeCatboostHeight(zRange[0], zRange[1], last)
          : catboostPosteriorHeight(last)
        : toH(base);
    return Math.max(h, 0.12);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mech, traces, isRegressor]);

  const maxShap = Math.max(Math.abs(shapX), Math.abs(shapY), 1e-9);
  const fxLen = showForceVectors ? catboostForceLength(shapX, maxShap) : 0;
  const fyLen = showForceVectors ? catboostForceLength(shapY, maxShap) : 0;
  const [fxr, fxg, fxb] = catboostForceColor(shapX);
  const [fyr, fyg, fyb] = catboostForceColor(shapY);

  const ballMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#22d3ee'),
        emissive: new THREE.Color('#0e7490'),
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
        color: new THREE.Color('#a5f3fc'),
        transparent: true,
        opacity: 0.95,
      }),
    [],
  );
  const lineGeos = useMemo(
    () => [...splitLines.x, ...splitLines.y].map(() => new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-2.5, 0, 0), new THREE.Vector3(2.5, 0, 0)])),
    [splitLines],
  );
  const lineMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#22d3ee'),
        transparent: true,
        opacity: 0.45,
      }),
    [],
  );
  const arrowXGeo = useMemo(() => new THREE.BoxGeometry(1, 0.03, 0.03), []);
  const arrowZGeo = useMemo(() => new THREE.BoxGeometry(0.03, 0.03, 1), []);
  const headGeo = useMemo(() => new THREE.ConeGeometry(0.05, 0.14, 12), []);
  const arrowXMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: new THREE.Color(fxr, fxg, fxb) }),
    [fxr, fxg, fxb],
  );
  const arrowYMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: new THREE.Color(fyr, fyg, fyb) }),
    [fyr, fyg, fyb],
  );
  useDisposableThreeResources(
    [...lineGeos, ringGeo, arrowXGeo, arrowZGeo, headGeo],
    [ballMaterial, ringMaterial, lineMaterial, arrowXMat, arrowYMat],
  );

  if (!surface || res < 2) return null;
  const total = overlays?.n_iterations ?? overlays?.n_boost_rounds ?? 0;
  const dirX = shapX >= 0 ? 1 : -1;
  const dirY = shapY >= 0 ? 1 : -1;
  const ticksX = (overlays?.categorical_ticks_x ?? []) as string[];
  const ticksY = (overlays?.categorical_ticks_y ?? []) as string[];

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
          {showSplitLines &&
            splitLines.x.map((at, i) => (
              <lineSegments key={`spx-${i}`} geometry={lineGeos[i]} material={lineMaterial} position={[0, 0.02, at]} />
            ))}
          {showSplitLines &&
            splitLines.y.map((at, i) => (
              <lineSegments
                key={`spy-${i}`}
                geometry={lineGeos[splitLines.x.length + i]}
                material={lineMaterial}
                position={[at, 0.02, 0]}
                rotation={[0, Math.PI / 2, 0]}
              />
            ))}
          {showForceVectors && fxLen > 1e-6 && (
            <group position={[queryPos.x, ballH, queryPos.z]}>
              <mesh geometry={arrowXGeo} material={arrowXMat} scale={[fxLen, 1, 1]} position={[dirX * (fxLen / 2), 0, 0]} />
              <mesh
                geometry={headGeo}
                material={arrowXMat}
                rotation={[0, 0, dirX > 0 ? -Math.PI / 2 : Math.PI / 2]}
                position={[dirX * (fxLen + 0.07), 0, 0]}
              />
            </group>
          )}
          {showForceVectors && fyLen > 1e-6 && (
            <group position={[queryPos.x, ballH, queryPos.z]}>
              <mesh geometry={arrowZGeo} material={arrowYMat} scale={[1, 1, fyLen]} position={[0, 0, dirY * (fyLen / 2)]} />
              <mesh
                geometry={headGeo}
                material={arrowYMat}
                rotation={[dirY > 0 ? Math.PI / 2 : -Math.PI / 2, 0, 0]}
                position={[0, 0, dirY * (fyLen + 0.07)]}
              />
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
          style={{ background: 'rgba(34,211,238,0.12)', color: '#a5f3fc', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          ◆ CatBoost oblivious grid ({total} trees · {splitLines.x.length + splitLines.y.length} split planes · iter ≤{' '}
          {Math.min(selectedRound + 1, Math.max(total, 1))})
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          teal = symmetric terraces · laser = oblivious splits · arrows = C++ SHAP push on X/Y · drag to orbit
        </span>
      </div>
      {showCategoricalTicks && (ticksX.length > 0 || ticksY.length > 0) && (
        <p className="font-mono-tabular text-[10px]" style={{ color: '#94a3b8' }}>
          {ticksX.length > 0 ? `X CAT: ${ticksX.slice(0, 8).join(' · ')}` : ''}
          {ticksX.length > 0 && ticksY.length > 0 ? '   |   ' : ''}
          {ticksY.length > 0 ? `Y CAT: ${ticksY.slice(0, 8).join(' · ')}` : ''}
        </p>
      )}
    </div>
  );
}

import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useTreeSliceStore } from '../../../model_engine/store/useTreeSliceStore';
import {
  normalizeRegressionHeight,
  posteriorHeight,
  splitLineWeight,
  splitsUpToDepth,
} from './treeSurfaceMath';

/**
 * Phase 27 — 3D stepped voxel terraces + split threshold lines.
 *
 * Own tiny WebGL canvas (no dataset/stage context needed): one flat
 * voxel box per grid cell (piecewise-constant terraces, never smoothed)
 * drawn with a single InstancedMesh draw call, glowing split-threshold
 * lines (shallow splits render thickest) on the X/Y axes, and the live
 * query ball with a gold leaf ring.
 */

export interface SteppedVoxelTerrain3DProps {
  liveFeatures?: number[] | null;
}

export function SteppedVoxelTerrain3D(props: SteppedVoxelTerrain3DProps) {
  const surface = useTreeSliceStore((s) => s.surface);
  const depthClip = useTreeSliceStore((s) => s.depthClip);
  const showSplits = useTreeSliceStore((s) => s.showSplits);
  const showWireframe = useTreeSliceStore((s) => s.showWireframe);
  const axisX = useTreeSliceStore((s) => s.axisX);
  const axisY = useTreeSliceStore((s) => s.axisY);
  const meshRef = useRef<THREE.InstancedMesh>(null!);

  const overlays = surface?.extra_overlays;
  const res = surface?.resolution ?? 0;
  const isRegressor = overlays?.is_regressor === true;
  const zRange = (overlays?.regression_z_range ?? [0, 1]) as [number, number];

  const cells = useMemo(() => {
    if (!surface || res < 2 || surface.z_values.length < res * res) return [];
    const out: Array<{ x: number; y: number; z: number; sx: number; sy: number; sz: number; shade: number }> = [];
    const cell = 5 / res;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const raw = surface.z_values[idx] ?? 0;
        const h = isRegressor ? normalizeRegressionHeight(zRange[0], zRange[1], raw) : posteriorHeight(raw);
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
  }, [surface, res, isRegressor, zRange]);

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
      // Cyan shade ramp: deep teal (low) -> bright cyan (high).
      col.setRGB(0.05 + cell.shade * 0.08, 0.35 + cell.shade * 0.48, 0.45 + cell.shade * 0.48);
      mesh.setColorAt(i, col);
    });
    mesh.count = cells.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [cells]);

  const splitLines = useMemo((): {
    x: Array<{ at: number; w: number }>;
    y: Array<{ at: number; w: number }>;
    maxDepth: number;
  } => {
    if (!surface || !showSplits) return { x: [], y: [], maxDepth: 1 };
    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const x0 = xs[0];
    const y0 = ys[0];
    const spanX = xs[xs.length - 1] - x0 || 1;
    const spanY = ys[ys.length - 1] - y0 || 1;
    const sxLines = splitsUpToDepth(overlays?.split_lines_x ?? [], depthClip);
    const syLines = splitsUpToDepth(overlays?.split_lines_y ?? [], depthClip);
    const maxDepth = Math.max(1, ...sxLines.map((l) => l.depth), ...syLines.map((l) => l.depth));
    return {
      x: sxLines.map((l) => ({ at: ((l.threshold - x0) / spanX) * 5 - 2.5, w: splitLineWeight(l.depth, maxDepth) })),
      y: syLines.map((l) => ({ at: ((l.threshold - y0) / spanY) * 5 - 2.5, w: splitLineWeight(l.depth, maxDepth) })),
      maxDepth,
    };
  }, [surface, overlays, depthClip, showSplits]);

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

  const splitGeos = useMemo(() => {
    const mk = (at: number, alongX: boolean): THREE.BufferGeometry => {
      const g = new THREE.BufferGeometry();
      const pts = alongX
        ? [new THREE.Vector3(at, 0, -2.5), new THREE.Vector3(at, 2.2, -2.5), new THREE.Vector3(at, 0, 2.5), new THREE.Vector3(at, 2.2, 2.5)]
        : [new THREE.Vector3(-2.5, 0, at), new THREE.Vector3(-2.5, 2.2, at), new THREE.Vector3(2.5, 0, at), new THREE.Vector3(2.5, 2.2, at)];
      // Two vertical quads per split (front + side) so cliffs read from any angle.
      const pos = new Float32Array([
        pts[0].x, pts[0].y, pts[0].z, pts[1].x, pts[1].y, pts[1].z,
        pts[2].x, pts[2].y, pts[2].z, pts[3].x, pts[3].y, pts[3].z,
      ]);
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      return g;
    };
    return {
      x: splitLines.x.map((l) => ({ geo: mk(l.at, true), w: l.w })),
      y: splitLines.y.map((l) => ({ geo: mk(l.at, false), w: l.w })),
    };
  }, [splitLines]);

  const splitMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#fbbf24'),
        transparent: true,
        opacity: 0.75,
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
  const ballMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#22d3ee'),
        emissive: new THREE.Color('#0e7490'),
        emissiveIntensity: 0.9,
      }),
    [],
  );
  useDisposableThreeResources(
    [...splitGeos.x.map((s) => s.geo), ...splitGeos.y.map((s) => s.geo), ringGeo],
    [splitMaterial, ringMaterial, ballMaterial],
  );

  if (!surface || res < 2) return null;

  return (
    <div className="flex flex-col gap-1">
      <div
        className="relative w-full overflow-hidden rounded"
        style={{ height: 300, border: '1px solid var(--border-subtle)', background: '#0b0f14' }}
      >
        <Canvas
          dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
          camera={{ position: [4.5, 3.8, 5.5], fov: 42, near: 0.1, far: 100 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
        >
          <ambientLight intensity={0.8} />
          <directionalLight position={[6, 8, 4]} intensity={1.15} />
          <instancedMesh ref={meshRef} args={[undefined, undefined, Math.max(cells.length, 1)]} frustumCulled={false}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial roughness={0.6} metalness={0.05} wireframe={showWireframe} />
          </instancedMesh>
          {showSplits &&
            splitGeos.x.map((s, i) => (
              <lineSegments key={`x-${i}`} geometry={s.geo} material={splitMaterial} />
            ))}
          {showSplits &&
            splitGeos.y.map((s, i) => (
              <lineSegments key={`y-${i}`} geometry={s.geo} material={splitMaterial} />
            ))}
          <lineLoop geometry={ringGeo} material={ringMaterial} position={[queryPos.x, 0.03, queryPos.z]} />
          <mesh position={queryPos} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#67e8f6', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          ▲ Stepped voxel terraces ({cells.length.toLocaleString()} voxels · no smoothing)
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          {splitGeos.x.length + splitGeos.y.length} split lines ≤ depth {depthClip} · drag to orbit
        </span>
      </div>
    </div>
  );
}

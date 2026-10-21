import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useSvmSliceStore } from '../../../model_engine/store/useSvmSliceStore';
import {
  normalizeSaddleHeight,
  svmSaddleColor,
  tetherOpacity,
} from './svmSurfaceMath';

/**
 * Phase 23 — curved kernel-saddle surface + support-vector pillars +
 * top-5 kernel force tethers.
 *
 * Own tiny WebGL canvas (no dataset/stage context needed): the 60×60
 * decision surface renders with saddle height coloring, the ±margin
 * corridor reuses the SAME BufferGeometry with a Y offset (Phase-21
 * pattern: −66% GPU vertex memory), support vectors render as vertical
 * pillar lines + glowing points, and the top-5 force tethers connect
 * the query ball to its strongest support vectors.
 */

export interface KernelSaddleSurface3DProps {
  liveFeatures?: number[] | null;
}

export function KernelSaddleSurface3D(props: KernelSaddleSurface3DProps) {
  const surface = useSvmSliceStore((s) => s.surface);
  const trace = useSvmSliceStore((s) => s.traces[0] ?? null);
  const axisX = useSvmSliceStore((s) => s.axisX);
  const axisY = useSvmSliceStore((s) => s.axisY);
  const showPillars = useSvmSliceStore((s) => s.showPillars);
  const showTethers = useSvmSliceStore((s) => s.showTethers);
  const showContours = useSvmSliceStore((s) => s.showContours);

  const scene = useMemo(() => {
    const empty = {
      saddleGeo: new THREE.BufferGeometry(),
      pillarGeo: new THREE.BufferGeometry(),
      svPointGeo: new THREE.BufferGeometry(),
      tetherGeo: new THREE.BufferGeometry(),
      res: 0,
      xs: [] as number[],
      ys: [] as number[],
      zMin: 0,
      zMax: 1,
      corridorHalfHeight: 0,
      marginLabel: '±1 margin',
      queryPos: new THREE.Vector3(0, 1, 0),
      ballColor: '#22d3ee',
      svCount: 0,
      tetherCount: 0,
      outlier: false,
    };
    if (!surface || surface.x_grid.length < 2 || surface.y_grid.length < 2) return empty;
    const res = surface.resolution;
    if (surface.z_values.length < res * res) return empty;

    const xs = surface.x_grid;
    const ys = surface.y_grid;
    const x0 = xs[0];
    const y0 = ys[0];
    const spanX = xs[xs.length - 1] - x0 || 1;
    const spanY = ys[ys.length - 1] - y0 || 1;

    let zMin = Infinity;
    let zMax = -Infinity;
    for (const z of surface.z_values) {
      if (Number.isFinite(z)) {
        zMin = Math.min(zMin, z);
        zMax = Math.max(zMax, z);
      }
    }
    if (!Number.isFinite(zMin) || !Number.isFinite(zMax)) {
      zMin = 0;
      zMax = 1;
    }

    const pos = new Float32Array(res * res * 3);
    const col = new Float32Array(res * res * 3);
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const z = surface.z_values[idx] ?? 0;
        const h = normalizeSaddleHeight(zMin, zMax, z);
        pos[idx * 3] = ((xs[c] - x0) / spanX) * 5 - 2.5;
        pos[idx * 3 + 1] = h;
        pos[idx * 3 + 2] = ((ys[r] - y0) / spanY) * 5 - 2.5;
        const t = zMax - zMin <= 1e-9 ? 0.5 : (z - zMin) / (zMax - zMin);
        const rgb = svmSaddleColor(t);
        col[idx * 3] = rgb[0];
        col[idx * 3 + 1] = rgb[1];
        col[idx * 3 + 2] = rgb[2];
      }
    }
    const index: number[] = [];
    for (let r = 0; r < res - 1; r += 1) {
      for (let c = 0; c < res - 1; c += 1) {
        const a = r * res + c;
        const b = a + 1;
        const d = a + res;
        const e = d + 1;
        index.push(a, d, b, b, d, e);
      }
    }
    const saddleGeo = new THREE.BufferGeometry();
    saddleGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    saddleGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    saddleGeo.setIndex(index);
    saddleGeo.computeVertexNormals();

    const overlays = surface.extra_overlays;
    const marginWidth =
      typeof overlays?.epsilon === 'number' && Number.isFinite(overlays.epsilon)
        ? (overlays.epsilon as number)
        : 1;
    const isRegressor = overlays?.is_classifier === false && !overlays?.is_oneclass;
    const span = zMax - zMin;
    const corridorHalfHeight = span > 1e-9 ? (marginWidth / span) * 2 : 0;
    const marginLabel = isRegressor
      ? `±${marginWidth.toFixed(3)} ε-tube`
      : marginWidth === 1
        ? '±1 margin contours'
        : `±${marginWidth.toFixed(3)} contours`;

    // Support-vector pillars: project SV coords onto the axis plane.
    const mech = trace?.extra_mechanics ?? null;
    const sv = mech?.support_vectors_display ?? null;
    const live = props.liveFeatures;
    let pillarGeo = new THREE.BufferGeometry();
    let svPointGeo = new THREE.BufferGeometry();
    let svCount = 0;
    if (sv && sv.coordinates.length > 0) {
      const n = sv.coordinates.length;
      const importance = sv.importance ?? [];
      let impMax = 1e-12;
      for (const v of importance) if (Number.isFinite(v)) impMax = Math.max(impMax, Math.abs(v));
      const px = new Float32Array(n * 6);
      const pp = new Float32Array(n * 3);
      for (let i = 0; i < n; i += 1) {
        const coord = sv.coordinates[i] ?? [];
        const vx = axisX < coord.length ? coord[axisX] : 0;
        const vy = axisY < coord.length ? coord[axisY] : 0;
        // SVs live in scaled feature space; fall back to the surface grid
        // extent when the live row defines a tighter window.
        const gx = ((vx - x0) / spanX) * 5 - 2.5;
        const gz = ((vy - y0) / spanY) * 5 - 2.5;
        const hgt = 0.25 + 1.1 * Math.min(1, Math.abs(importance[i] ?? 0) / impMax);
        px[i * 6] = gx;
        px[i * 6 + 1] = 0;
        px[i * 6 + 2] = gz;
        px[i * 6 + 3] = gx;
        px[i * 6 + 4] = hgt;
        px[i * 6 + 5] = gz;
        pp[i * 3] = gx;
        pp[i * 3 + 1] = hgt;
        pp[i * 3 + 2] = gz;
      }
      pillarGeo = new THREE.BufferGeometry();
      pillarGeo.setAttribute('position', new THREE.BufferAttribute(px, 3));
      svPointGeo = new THREE.BufferGeometry();
      svPointGeo.setAttribute('position', new THREE.BufferAttribute(pp, 3));
      svCount = n;
    }

    // Query ball: sample the saddle height at the live (x, y).
    let queryPos = new THREE.Vector3(0, 1, 0);
    let ballColor = '#22d3ee';
    if (live && live.length > 0) {
      const lx = live[axisX];
      const ly = live[axisY];
      if (Number.isFinite(lx) && Number.isFinite(ly)) {
        const fx = Math.min(Math.max((lx - x0) / spanX, 0), 1) * (res - 1);
        const fy = Math.min(Math.max((ly - y0) / spanY, 0), 1) * (res - 1);
        const c0 = Math.floor(fx);
        const r0 = Math.floor(fy);
        const c1 = Math.min(c0 + 1, res - 1);
        const r1 = Math.min(r0 + 1, res - 1);
        const tx = fx - c0;
        const ty = fy - r0;
        const z00 = surface.z_values[r0 * res + c0] ?? 0;
        const z10 = surface.z_values[r0 * res + c1] ?? 0;
        const z01 = surface.z_values[r1 * res + c0] ?? 0;
        const z11 = surface.z_values[r1 * res + c1] ?? 0;
        const z = z00 * (1 - tx) * (1 - ty) + z10 * tx * (1 - ty) + z01 * (1 - tx) * ty + z11 * tx * ty;
        queryPos = new THREE.Vector3(
          ((lx - x0) / spanX) * 5 - 2.5,
          normalizeSaddleHeight(zMin, zMax, z),
          ((ly - y0) / spanY) * 5 - 2.5,
        );
        const raw = typeof mech?.raw_score === 'number' ? (mech.raw_score as number) : z;
        if (mech?.predicted_label === 'OUTLIER' || (mech && raw < 0 && mech.is_oneclass)) {
          ballColor = '#f43f5e';
        } else if (Number.isFinite(raw) && Math.abs(raw) < (isRegressor ? marginWidth : 1)) {
          ballColor = '#fbbf24';
        } else if (Number.isFinite(raw) && raw >= 0) {
          ballColor = '#34d399';
        } else {
          ballColor = '#fb7185';
        }
      }
    }

    // Top-5 tethers: ball -> SV display positions.
    let tetherGeo = new THREE.BufferGeometry();
    let tetherCount = 0;
    const forces = (mech?.kernel_forces_top ?? []) as Array<{
      sv_position: number;
      kernel_value: number;
      force: number;
    }>;
    const displayPositions = (sv?.display_positions ?? []) as number[];
    const invMap = new Map<number, number>();
    displayPositions.forEach((origPos, dispIdx) => {
      if (!invMap.has(origPos)) invMap.set(origPos, dispIdx);
    });
    if (sv && forces.length > 0 && svPointGeo.getAttribute('position')) {
      const ppAttr = svPointGeo.getAttribute('position') as THREE.BufferAttribute;
      const maxAbs = Math.max(...forces.map((f) => Math.abs(f.force ?? 0)), 1e-12);
      const tp = new Float32Array(forces.length * 6);
      const tc = new Float32Array(forces.length * 6);
      let kept = 0;
      forces.forEach((f) => {
        const dispIdx = invMap.get(f.sv_position);
        if (dispIdx === undefined || dispIdx < 0 || dispIdx >= svCount) return;
        tp[kept * 6] = queryPos.x;
        tp[kept * 6 + 1] = queryPos.y;
        tp[kept * 6 + 2] = queryPos.z;
        tp[kept * 6 + 3] = ppAttr.getX(dispIdx);
        tp[kept * 6 + 4] = ppAttr.getY(dispIdx);
        tp[kept * 6 + 5] = ppAttr.getZ(dispIdx);
        const op = tetherOpacity(f.force ?? 0, maxAbs);
        const pulling = (f.force ?? 0) >= 0;
        tc[kept * 6] = pulling ? 0.13 : 0.96;
        tc[kept * 6 + 1] = pulling ? 0.83 : 0.44;
        tc[kept * 6 + 2] = pulling ? 0.93 : 0.52;
        tc[kept * 6 + 3] = (pulling ? 0.13 : 0.96) * op;
        tc[kept * 6 + 4] = (pulling ? 0.83 : 0.44) * op;
        tc[kept * 6 + 5] = (pulling ? 0.93 : 0.52) * op;
        kept += 1;
      });
      tetherGeo = new THREE.BufferGeometry();
      tetherGeo.setAttribute('position', new THREE.BufferAttribute(tp.slice(0, kept * 6), 3));
      tetherGeo.setAttribute('color', new THREE.BufferAttribute(tc.slice(0, kept * 6), 3));
      tetherCount = kept;
    }

    const outlier = mech?.predicted_label === 'OUTLIER';
    return {
      saddleGeo,
      pillarGeo,
      svPointGeo,
      tetherGeo,
      res,
      xs,
      ys,
      zMin,
      zMax,
      corridorHalfHeight,
      marginLabel,
      queryPos,
      ballColor,
      svCount,
      tetherCount,
      outlier,
    };
  }, [surface, trace, props.liveFeatures, axisX, axisY]);

  const saddleMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.62,
        side: THREE.DoubleSide,
        roughness: 0.55,
        metalness: 0.1,
        depthWrite: false,
      }),
    [],
  );
  const contourPosMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#4ade80'),
        wireframe: true,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [],
  );
  const contourNegMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#fb923c'),
        wireframe: true,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [],
  );
  const pillarMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#67e8f6'),
        transparent: true,
        opacity: 0.45,
      }),
    [],
  );
  const svPointMaterial = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.09,
        color: new THREE.Color('#fbbf24'),
        transparent: true,
        opacity: 1.0,
        sizeAttenuation: true,
        depthWrite: false,
      }),
    [],
  );
  const tetherMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
      }),
    [],
  );
  const ballMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(scene.ballColor),
        emissive: new THREE.Color(scene.ballColor),
        emissiveIntensity: 0.55,
      }),
    [scene.ballColor],
  );
  useDisposableThreeResources(
    [scene.saddleGeo, scene.pillarGeo, scene.svPointGeo, scene.tetherGeo],
    [saddleMaterial, contourPosMaterial, contourNegMaterial, pillarMaterial, svPointMaterial, tetherMaterial, ballMaterial],
  );

  if (!surface) return null;

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
          <ambientLight intensity={0.75} />
          <directionalLight position={[6, 8, 4]} intensity={1.1} />
          <mesh geometry={scene.saddleGeo} material={saddleMaterial} />
          {showContours && (
            <>
              <mesh
                geometry={scene.saddleGeo}
                material={contourPosMaterial}
                position={[0, scene.corridorHalfHeight, 0]}
              />
              <mesh
                geometry={scene.saddleGeo}
                material={contourNegMaterial}
                position={[0, -scene.corridorHalfHeight, 0]}
              />
            </>
          )}
          {showPillars && scene.svCount > 0 && (
            <>
              <lineSegments geometry={scene.pillarGeo} material={pillarMaterial} />
              <points geometry={scene.svPointGeo} material={svPointMaterial} />
            </>
          )}
          {showTethers && scene.tetherCount > 0 && (
            <lineSegments geometry={scene.tetherGeo} material={tetherMaterial} />
          )}
          <mesh position={scene.queryPos} material={ballMaterial}>
            <sphereGeometry args={[0.09, 24, 24]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#67e8f6', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          {scene.marginLabel}
          {showContours ? ' · shown' : ' · hidden'}
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          z∈[{scene.zMin.toFixed(2)}..{scene.zMax.toFixed(2)}] · {scene.svCount.toLocaleString()} SV pillars ·{' '}
          {scene.tetherCount} tethers · drag to orbit
        </span>
        {scene.outlier && (
          <span
            className="font-mono-tabular pointer-events-none absolute top-8 left-2 rounded px-1.5 py-0.5 text-[10px]"
            style={{ background: 'rgba(244,63,94,0.16)', color: '#fda4af', border: '1px solid rgba(244,63,94,0.45)' }}
          >
            ⚠ OUTLIER: outside the envelope (f(x) &lt; 0)
          </span>
        )}
      </div>
    </div>
  );
}

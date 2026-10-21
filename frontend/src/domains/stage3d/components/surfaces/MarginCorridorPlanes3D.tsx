import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useSgdSliceStore } from '../../../model_engine/store/useSgdSliceStore';
import {
  liveSgdHeight,
  sgdCorridorColor,
} from './sgdSurfaceMath';

/**
 * Phase 21 — SGD tilted-plane mini-preview with the ±margin corridor
 * (single-geometry offset rendering) + orthogonal laser drop-line.
 *
 * Renders the 60×60 analytical plane (own tiny WebGL canvas so the Model
 * step needs no dataset/stage context) with corridor-tinted height
 * coloring. The ±margin planes reuse the SAME BufferGeometry with a Y
 * offset of ±marginWidth (blueprint §6.1: −66% GPU vertex memory) and
 * render as wireframes: green (+m) / orange (−m). For regressors the
 * corridor is the ±ε tube. The amber laser drops vertically from the
 * live What-If ball to the decision plane with the d_geom readout.
 */

export interface MarginCorridorPlanes3DProps {
  liveFeatures?: number[] | null;
}

function normalizeHeights(zMin: number, zMax: number, z: number): number {
  const span = zMax - zMin;
  if (!Number.isFinite(span) || span <= 1e-9) return 1;
  return ((z - zMin) / span) * 2;
}

export function MarginCorridorPlanes3D(props: MarginCorridorPlanes3DProps) {
  const surface = useSgdSliceStore((s) => s.surface);
  const axisX = useSgdSliceStore((s) => s.axisX);
  const axisY = useSgdSliceStore((s) => s.axisY);
  const showCorridor = useSgdSliceStore((s) => s.showCorridor);
  const showLaser = useSgdSliceStore((s) => s.showLaser);
  const trace = useSgdSliceStore((s) => s.traces[0] ?? null);

  const plane = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const empty = {
      geometry: geo,
      actualMin: 0,
      actualMax: 1,
      liveRecompute: false,
      corridorHalfHeight: 0,
      marginWidth: 1,
      isRegressor: false,
      raw: new Float64Array(0),
      res: 0,
      xs: [] as number[],
      ys: [] as number[],
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

    const overlays = surface.extra_overlays;
    const weightRow = Array.isArray(overlays?.weight_row) ? (overlays.weight_row as number[]) : null;
    const baseIntercept = typeof overlays?.base_intercept === 'number' ? (overlays.base_intercept as number) : null;
    const marginWidth =
      typeof overlays?.margin_width === 'number' && Number.isFinite(overlays.margin_width)
        ? (overlays.margin_width as number)
        : 1;
    const isRegressor = overlays?.is_classifier === false;
    const live = props.liveFeatures;
    const canLive =
      overlays?.gpu_formula_valid === true &&
      weightRow !== null &&
      baseIntercept !== null &&
      live !== null &&
      live !== undefined &&
      live.length > 0 &&
      live.every((v) => Number.isFinite(v));

    const raw = new Float64Array(res * res);
    let zMin = Infinity;
    let zMax = -Infinity;
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const z =
          canLive && weightRow !== null && baseIntercept !== null && live
            ? liveSgdHeight(weightRow, baseIntercept, live, axisX, axisY, xs[c], ys[r])
            : (surface.z_values[idx] ?? 0);
        const safe = Number.isFinite(z) ? z : 0;
        raw[idx] = safe;
        if (safe < zMin) zMin = safe;
        if (safe > zMax) zMax = safe;
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
        const h = normalizeHeights(zMin, zMax, raw[idx]);
        pos[idx * 3] = ((xs[c] - x0) / spanX) * 5 - 2.5;
        pos[idx * 3 + 1] = h;
        pos[idx * 3 + 2] = ((ys[r] - y0) / spanY) * 5 - 2.5;
        const t = zMax - zMin <= 1e-9 ? 0.5 : (raw[idx] - zMin) / (zMax - zMin);
        const rgb = sgdCorridorColor(t);
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
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(index);
    geo.computeVertexNormals();

    const span = zMax - zMin;
    const half = span > 1e-9 ? (marginWidth / span) * 2 : 0;
    return {
      geometry: geo,
      actualMin: zMin,
      actualMax: zMax,
      liveRecompute: canLive,
      corridorHalfHeight: half,
      marginWidth,
      isRegressor,
      raw,
      res,
      xs,
      ys,
    };
  }, [surface, props.liveFeatures, axisX, axisY]);

  const laser = useMemo(() => {
    if (!showLaser || !trace || plane.res < 2 || plane.raw.length === 0) return null;
    const mech = trace.extra_mechanics;
    const rawScore = typeof mech.raw_score === 'number' ? mech.raw_score : null;
    const geomDist = typeof mech.geometric_distance === 'number' ? mech.geometric_distance : null;
    const live = props.liveFeatures;
    if (rawScore === null || live === null || live === undefined) return null;
    const lx = live[axisX];
    const ly = live[axisY];
    if (!Number.isFinite(lx) || !Number.isFinite(ly)) return null;
    const { xs, ys, res, raw, actualMin, actualMax } = plane;
    const x0 = xs[0];
    const y0 = ys[0];
    const spanX = xs[xs.length - 1] - x0 || 1;
    const spanY = ys[ys.length - 1] - y0 || 1;
    // Bilinear sample of the plane height at the live (x, y).
    const fx = Math.min(Math.max((lx - x0) / spanX, 0), 1) * (res - 1);
    const fy = Math.min(Math.max((ly - y0) / spanY, 0), 1) * (res - 1);
    const c0 = Math.floor(fx);
    const r0 = Math.floor(fy);
    const c1 = Math.min(c0 + 1, res - 1);
    const r1 = Math.min(r0 + 1, res - 1);
    const tx = fx - c0;
    const ty = fy - r0;
    const z00 = raw[r0 * res + c0] ?? 0;
    const z10 = raw[r0 * res + c1] ?? 0;
    const z01 = raw[r1 * res + c0] ?? 0;
    const z11 = raw[r1 * res + c1] ?? 0;
    const planeZ = z00 * (1 - tx) * (1 - ty) + z10 * tx * (1 - ty) + z01 * (1 - tx) * ty + z11 * tx * ty;
    const wx = ((lx - x0) / spanX) * 5 - 2.5;
    const wz = ((ly - y0) / spanY) * 5 - 2.5;
    const ballH = normalizeHeights(actualMin, actualMax, rawScore);
    const surfH = normalizeHeights(actualMin, actualMax, planeZ);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([wx, ballH, wz, wx, surfH, wz]), 3),
    );
    return {
      line: new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: new THREE.Color('#fbbf24'), transparent: true, opacity: 0.95 }),
      ),
      ballPos: new THREE.Vector3(wx, ballH, wz),
      geomDist,
      inside: mech.inside_margin === true,
    };
  }, [showLaser, trace, plane, props.liveFeatures, axisX, axisY]);

  const mainMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        side: THREE.DoubleSide,
        roughness: 0.55,
        metalness: 0.1,
        depthWrite: false,
      }),
    [],
  );
  const corridorPosMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#4ade80'),
        wireframe: true,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [],
  );
  const corridorNegMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#fb923c'),
        wireframe: true,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [],
  );
  useDisposableThreeResources(
    [plane.geometry, ...(laser ? [laser.line.geometry] : [])],
    [
      mainMaterial,
      corridorPosMaterial,
      corridorNegMaterial,
      ...(laser ? [laser.line.material as THREE.Material] : []),
    ],
  );

  if (!surface) return null;
  const corridorLabel =
    plane.marginWidth === 1 && !plane.isRegressor
      ? '±1 margin corridor'
      : `±${plane.marginWidth.toFixed(3)} ${plane.isRegressor ? 'ε-tube' : 'corridor'}`;

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
          <mesh geometry={plane.geometry} material={mainMaterial} />
          {showCorridor && (
            <>
              <mesh
                geometry={plane.geometry}
                material={corridorPosMaterial}
                position={[0, plane.corridorHalfHeight, 0]}
              />
              <mesh
                geometry={plane.geometry}
                material={corridorNegMaterial}
                position={[0, -plane.corridorHalfHeight, 0]}
              />
            </>
          )}
          {laser && (
            <>
              <primitive object={laser.line} />
              <mesh position={laser.ballPos}>
                <sphereGeometry args={[0.09, 24, 24]} />
                <meshStandardMaterial
                  color={laser.inside ? '#fbbf24' : '#22d3ee'}
                  emissive={laser.inside ? '#b45309' : '#0e7490'}
                  emissiveIntensity={0.9}
                />
              </mesh>
            </>
          )}
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          z∈[{plane.actualMin.toFixed(2)}..{plane.actualMax.toFixed(2)}]
          {plane.liveRecompute ? ' · live phase-shift' : ''} · drag to orbit
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#67e8f6', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          {corridorLabel}
          {showCorridor ? ' · shown' : ' · hidden'}
        </span>
        {laser && laser.geomDist !== null && (
          <span
            className="font-mono-tabular pointer-events-none absolute top-8 left-2 rounded px-1.5 py-0.5 text-[10px]"
            style={{
              background: laser.inside ? 'rgba(251,191,36,0.16)' : 'rgba(52,211,153,0.14)',
              color: laser.inside ? '#fcd34d' : '#6ee7b7',
              border: `1px solid ${laser.inside ? 'rgba(251,191,36,0.45)' : 'rgba(52,211,153,0.4)'}`,
            }}
          >
            d_geom = {laser.geomDist >= 0 ? '+' : ''}{laser.geomDist.toFixed(3)}
            {laser.inside ? ' ⚠ INSIDE CORRIDOR' : ' · safe'}
          </span>
        )}
      </div>
    </div>
  );
}

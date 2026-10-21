import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useKnnSliceStore } from '../../../model_engine/store/useKnnSliceStore';
import { spiderOpacity, type KnnRadarShape } from './knnSurfaceMath';

/**
 * Phase 22 — KNN expanding radar bubble + spider-web laser lines.
 *
 * Own tiny WebGL canvas (no dataset/stage context needed): the reference
 * cloud renders as THREE.Points (1 draw call), the radar shell as a
 * wireframe sphere / octahedron / cube sized to touch the K-th neighbor
 * in projected 3D, and the top-15 spider lines as LineSegments with
 * weight-scaled opacity. The query ball sits at y=0.6; the cloud plane
 * sits at y=0.
 */

export interface KnnRadarSphere3DProps {
  liveFeatures?: number[] | null;
}

const LABEL_COLORS: Array<[number, number, number]> = [
  [0x22 / 255, 0xd3 / 255, 0xee / 255],
  [0xf4 / 255, 0x3f / 255, 0x5e / 255],
  [0x34 / 255, 0xd3 / 255, 0x99 / 255],
  [0xf5 / 255, 0x9e / 255, 0x0b / 255],
  [0xe8 / 255, 0x79 / 255, 0xf9 / 255],
  [0x81 / 255, 0x8c / 255, 0xf8 / 255],
];

function labelColor(label: number | string): [number, number, number] {
  const idx =
    typeof label === 'number' && Number.isFinite(label)
      ? Math.abs(Math.floor(label)) % LABEL_COLORS.length
      : Math.abs(
          String(label)
            .split('')
            .reduce((a, c) => a + c.charCodeAt(0), 0),
        ) % LABEL_COLORS.length;
  return LABEL_COLORS[idx];
}

function targetColor(t: number, tMin: number, tMax: number): [number, number, number] {
  const span = tMax - tMin;
  const u = span > 1e-9 ? Math.min(1, Math.max(0, (t - tMin) / span)) : 0.5;
  // cyan -> amber ramp for regression targets
  const lo: [number, number, number] = [0x22 / 255, 0xd3 / 255, 0xee / 255];
  const hi: [number, number, number] = [0xf5 / 255, 0x9e / 255, 0x0b / 255];
  return [lo[0] + (hi[0] - lo[0]) * u, lo[1] + (hi[1] - lo[1]) * u, lo[2] + (hi[2] - lo[2]) * u];
}

export function KnnRadarSphere3D(props: KnnRadarSphere3DProps) {
  const surface = useKnnSliceStore((s) => s.surface);
  const trace = useKnnSliceStore((s) => s.traces[0] ?? null);
  const axisX = useKnnSliceStore((s) => s.axisX);
  const axisY = useKnnSliceStore((s) => s.axisY);
  const showRadar = useKnnSliceStore((s) => s.showRadar);
  const showSpiders = useKnnSliceStore((s) => s.showSpiders);
  const cloudOpacity = useKnnSliceStore((s) => s.cloudOpacity);

  const scene = useMemo(() => {
    const empty = {
      cloudGeo: new THREE.BufferGeometry(),
      neighborGeo: new THREE.BufferGeometry(),
      spiderGeo: new THREE.BufferGeometry(),
      cloudCount: 0,
      neighborCount: 0,
      spiderCount: 0,
      queryPos: new THREE.Vector3(0, 0.6, 0),
      radarRadius: 0,
      radarShape: 'sphere' as KnnRadarShape,
      radarLabel: '—',
      outlier: false,
    };
    if (!trace) return empty;
    const mech = trace.extra_mechanics;
    const cloud = mech.reference_cloud;
    if (!cloud || cloud.coordinates.length === 0) return empty;
    const coords = cloud.coordinates;
    const targets = cloud.targets;
    const n = coords.length;
    const gx = (i: number): number => (i < coords.length && axisX < coords[i].length ? coords[i][axisX] : 0);
    const gy = (i: number): number => (i < coords.length && axisY < coords[i].length ? coords[i][axisY] : 0);
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (let i = 0; i < n; i += 1) {
      const vx = gx(i);
      const vy = gy(i);
      if (Number.isFinite(vx)) {
        xMin = Math.min(xMin, vx);
        xMax = Math.max(xMax, vx);
      }
      if (Number.isFinite(vy)) {
        yMin = Math.min(yMin, vy);
        yMax = Math.max(yMax, vy);
      }
    }
    if (surface && surface.x_grid.length >= 2) {
      xMin = Math.min(xMin, surface.x_grid[0], surface.x_grid[surface.x_grid.length - 1]);
      xMax = Math.max(xMax, surface.x_grid[0], surface.x_grid[surface.x_grid.length - 1]);
    }
    if (surface && surface.y_grid.length >= 2) {
      yMin = Math.min(yMin, surface.y_grid[0], surface.y_grid[surface.y_grid.length - 1]);
      yMax = Math.max(yMax, surface.y_grid[0], surface.y_grid[surface.y_grid.length - 1]);
    }
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMax - xMin <= 1e-9) {
      xMin = -3;
      xMax = 3;
    }
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax - yMin <= 1e-9) {
      yMin = -3;
      yMax = 3;
    }
    const nx = (v: number): number => ((v - xMin) / (xMax - xMin)) * 5 - 2.5;
    const nz = (v: number): number => ((v - yMin) / (yMax - yMin)) * 5 - 2.5;

    const isClassifier = mech.is_classifier !== false;
    const numTargets = targets.map((t) => (typeof t === 'number' ? t : Number.NaN)).filter(Number.isFinite);
    const tMin = numTargets.length > 0 ? Math.min(...numTargets) : 0;
    const tMax = numTargets.length > 0 ? Math.max(...numTargets) : 1;

    const cloudPos = new Float32Array(n * 3);
    const cloudCol = new Float32Array(n * 3);
    for (let i = 0; i < n; i += 1) {
      cloudPos[i * 3] = nx(gx(i));
      cloudPos[i * 3 + 1] = 0;
      cloudPos[i * 3 + 2] = nz(gy(i));
      const rgb = isClassifier
        ? labelColor(targets[i] as number | string)
        : targetColor(typeof targets[i] === 'number' ? (targets[i] as number) : tMin, tMin, tMax);
      cloudCol[i * 3] = rgb[0];
      cloudCol[i * 3 + 1] = rgb[1];
      cloudCol[i * 3 + 2] = rgb[2];
    }
    const cloudGeo = new THREE.BufferGeometry();
    cloudGeo.setAttribute('position', new THREE.BufferAttribute(cloudPos, 3));
    cloudGeo.setAttribute('color', new THREE.BufferAttribute(cloudCol, 3));

    // Query ball position from live features (fallback: cloud centroid).
    const live = props.liveFeatures;
    const qxRaw =
      live && Number.isFinite(live[axisX])
        ? (live[axisX] as number)
        : (xMin + xMax) / 2;
    const qyRaw =
      live && Number.isFinite(live[axisY])
        ? (live[axisY] as number)
        : (yMin + yMax) / 2;
    const queryPos = new THREE.Vector3(nx(qxRaw), 0.6, nz(qyRaw));

    // Displayed neighbors: top-15 closest (blueprint edge #6).
    const localIdx = (mech.neighbor_local_indices ?? []) as number[];
    const dists = (mech.neighbor_distances ?? []) as number[];
    const weights = (mech.neighbor_weights ?? []) as number[];
    const order = localIdx.map((_, i) => i).sort((a, b) => (dists[a] ?? 0) - (dists[b] ?? 0));
    const cap = Math.min(15, order.length);
    const shown = order.slice(0, cap);
    const maxW = shown.length > 0 ? Math.max(...shown.map((i) => weights[i] ?? 0), 1e-12) : 1;

    const neighborGeo = new THREE.BufferGeometry();
    let neighborCount = 0;
    if (shown.length > 0) {
      const np = new Float32Array(shown.length * 3);
      shown.forEach((src, k) => {
        const loc = localIdx[src] ?? -1;
        if (loc >= 0 && loc < n) {
          np[k * 3] = cloudPos[loc * 3];
          np[k * 3 + 1] = 0.12;
          np[k * 3 + 2] = cloudPos[loc * 3 + 2];
        } else {
          np[k * 3] = queryPos.x;
          np[k * 3 + 1] = 0.12;
          np[k * 3 + 2] = queryPos.z;
        }
      });
      neighborGeo.setAttribute('position', new THREE.BufferAttribute(np, 3));
      neighborCount = shown.length;
    }

    const spiderGeo = new THREE.BufferGeometry();
    let spiderCount = 0;
    if (shown.length > 0) {
      const sp = new Float32Array(shown.length * 2 * 3);
      const scol = new Float32Array(shown.length * 2 * 3);
      shown.forEach((src, k) => {
        const loc = localIdx[src] ?? -1;
        const ex = loc >= 0 && loc < n ? cloudPos[loc * 3] : queryPos.x;
        const ez = loc >= 0 && loc < n ? cloudPos[loc * 3 + 2] : queryPos.z;
        sp[k * 6] = queryPos.x;
        sp[k * 6 + 1] = queryPos.y;
        sp[k * 6 + 2] = queryPos.z;
        sp[k * 6 + 3] = ex;
        sp[k * 6 + 4] = 0.12;
        sp[k * 6 + 5] = ez;
        const op = spiderOpacity(weights[src] ?? 0, maxW);
        scol[k * 6] = 0.13;
        scol[k * 6 + 1] = 0.83;
        scol[k * 6 + 2] = 0.93;
        scol[k * 6 + 3] = 0.13 * op;
        scol[k * 6 + 4] = 0.83 * op;
        scol[k * 6 + 5] = 0.93 * op;
      });
      spiderGeo.setAttribute('position', new THREE.BufferAttribute(sp, 3));
      spiderGeo.setAttribute('color', new THREE.BufferAttribute(scol, 3));
      spiderCount = shown.length;
    }

    // Projected radar radius: touch the K-th SHOWN neighbor in 3D view.
    let radarRadius = 0;
    if (shown.length > 0) {
      let furthest = 0;
      shown.forEach((src) => {
        const loc = localIdx[src] ?? -1;
        const ex = loc >= 0 && loc < n ? cloudPos[loc * 3] : queryPos.x;
        const ez = loc >= 0 && loc < n ? cloudPos[loc * 3 + 2] : queryPos.z;
        const dx = ex - queryPos.x;
        const dz = ez - queryPos.z;
        const d = Math.sqrt(dx * dx + dz * dz + 0.6 * 0.6);
        furthest = Math.max(furthest, d);
      });
      radarRadius = furthest;
    }

    const shape = (mech.radar_geometry_shape ?? 'sphere') as KnnRadarShape;
    const outlier = mech.is_outlier === true;
    const radarLabel =
      typeof mech.radar_radius === 'number' ? `R_t = ${mech.radar_radius.toFixed(3)} units` : '—';
    return {
      cloudGeo,
      neighborGeo,
      spiderGeo,
      cloudCount: n,
      neighborCount,
      spiderCount: spiderCount,
      queryPos,
      radarRadius,
      radarShape: shape,
      radarLabel,
      outlier,
    };
  }, [trace, surface, props.liveFeatures, axisX, axisY]);

  const cloudMaterial = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.055,
        vertexColors: true,
        transparent: true,
        opacity: cloudOpacity,
        sizeAttenuation: true,
        depthWrite: false,
      }),
    [cloudOpacity],
  );
  const neighborMaterial = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.14,
        color: new THREE.Color('#fbbf24'),
        transparent: true,
        opacity: 1.0,
        sizeAttenuation: true,
        depthWrite: false,
      }),
    [],
  );
  const spiderMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
      }),
    [],
  );
  const radarMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color('#22d3ee'),
        wireframe: true,
        transparent: true,
        opacity: 0.4,
        depthWrite: false,
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
    [scene.cloudGeo, scene.neighborGeo, scene.spiderGeo],
    [cloudMaterial, neighborMaterial, spiderMaterial, radarMaterial, ballMaterial],
  );

  if (!trace) return null;
  const radarArgs: Array<number> =
    scene.radarShape === 'octahedron'
      ? [Math.max(scene.radarRadius, 1e-6), 0]
      : scene.radarShape === 'cube'
        ? [Math.max(scene.radarRadius, 1e-6) * 1.35, Math.max(scene.radarRadius, 1e-6) * 1.35, Math.max(scene.radarRadius, 1e-6) * 1.35]
        : [Math.max(scene.radarRadius, 1e-6), 20, 14];

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
          <directionalLight position={[6, 8, 4]} intensity={1.1} />
          {scene.cloudCount > 0 && <points geometry={scene.cloudGeo} material={cloudMaterial} />}
          {scene.neighborCount > 0 && <points geometry={scene.neighborGeo} material={neighborMaterial} />}
          {showSpiders && scene.spiderCount > 0 && (
            <lineSegments geometry={scene.spiderGeo} material={spiderMaterial} />
          )}
          {showRadar && scene.radarRadius > 1e-9 && !scene.outlier && (
            <mesh position={scene.queryPos} material={radarMaterial}>
              {scene.radarShape === 'octahedron' ? (
                <octahedronGeometry args={[radarArgs[0], radarArgs[1]]} />
              ) : scene.radarShape === 'cube' ? (
                <boxGeometry args={[radarArgs[0], radarArgs[1], radarArgs[2]]} />
              ) : (
                <sphereGeometry args={[radarArgs[0], radarArgs[1], radarArgs[2]]} />
              )}
            </mesh>
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
          {scene.radarShape === 'octahedron'
            ? '◆ Octahedron radar (L1)'
            : scene.radarShape === 'cube'
              ? '■ Cube radar (L∞)'
              : '● Sphere radar (L2)'}
          {showRadar ? ` · ${scene.radarLabel}` : ' · hidden'}
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          {scene.cloudCount.toLocaleString()} ref pts · {scene.spiderCount} spider lines · drag to orbit
        </span>
        {scene.outlier && (
          <span
            className="font-mono-tabular pointer-events-none absolute top-8 left-2 rounded px-1.5 py-0.5 text-[10px]"
            style={{ background: 'rgba(244,63,94,0.16)', color: '#fda4af', border: '1px solid rgba(244,63,94,0.45)' }}
          >
            ⚠ OUTLIER: 0 neighbors in radius
          </span>
        )}
      </div>
    </div>
  );
}

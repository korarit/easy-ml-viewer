import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useDiscriminantSliceStore } from '../../../model_engine/store/useDiscriminantSliceStore';
import {
  ellipsoidColor,
  ellipsePoint,
  sigmaRadii,
} from './discriminantSurfaceMath';

/**
 * Phase 26 — tilted covariance ellipsoid shells + Mahalanobis tethers.
 *
 * Own tiny WebGL canvas (no dataset/stage context needed): the posterior
 * surface of the resolved target class, one rotated 1σ inner ring and one
 * m-σ outer wireframe ring per class (radii from the 2×2 sub-covariance
 * eigendecomposition, det(V) = +1 guaranteed backend-side), principal
 * eigen-axes through each center, and laser tethers from the live query
 * ball to every class mean. The paradox banner (Mahalanobis winner ≠
 * Euclidean winner) is raised by the inspector panel.
 */

export interface CovarianceEllipsoids3DProps {
  liveFeatures?: number[] | null;
}

interface EllipsoidScene {
  surface: THREE.BufferGeometry;
  surfaceMat: THREE.Material;
  rings: Array<{ geo: THREE.BufferGeometry; k: number }>;
  axes: THREE.BufferGeometry;
  centers: THREE.BufferGeometry;
  tethers: THREE.BufferGeometry;
  queryPos: THREE.Vector3;
  ringCount: number;
  tetherCount: number;
}

export function CovarianceEllipsoids3D(props: CovarianceEllipsoids3DProps) {
  const surface = useDiscriminantSliceStore((s) => s.surface);
  const axisX = useDiscriminantSliceStore((s) => s.axisX);
  const axisY = useDiscriminantSliceStore((s) => s.axisY);
  const sigmaLevel = useDiscriminantSliceStore((s) => s.sigmaLevel);
  const showAxes = useDiscriminantSliceStore((s) => s.showAxes);
  const showTethers = useDiscriminantSliceStore((s) => s.showTethers);

  const scene: EllipsoidScene = useMemo(() => {
    const empty: EllipsoidScene = {
      surface: new THREE.BufferGeometry(),
      surfaceMat: new THREE.MeshStandardMaterial(),
      rings: [],
      axes: new THREE.BufferGeometry(),
      centers: new THREE.BufferGeometry(),
      tethers: new THREE.BufferGeometry(),
      queryPos: new THREE.Vector3(0, 0.1, 0),
      ringCount: 0,
      tetherCount: 0,
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
    const toSX = (x: number): number => ((x - x0) / spanX) * 5 - 2.5;
    const toSZ = (y: number): number => ((y - y0) / spanY) * 5 - 2.5;

    const posArr = new Float32Array(res * res * 3);
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const z = Number.isFinite(surface.z_values[idx]) ? Math.min(1, Math.max(0, surface.z_values[idx])) : 0;
        posArr[idx * 3] = toSX(xs[c]);
        posArr[idx * 3 + 1] = z * 2;
        posArr[idx * 3 + 2] = toSZ(ys[r]);
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
    const surfaceGeo = new THREE.BufferGeometry();
    surfaceGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    surfaceGeo.setIndex(index);
    surfaceGeo.computeVertexNormals();
    const surfaceMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#22d3ee'),
      transparent: true,
      opacity: 0.45,
      side: THREE.DoubleSide,
      roughness: 0.55,
      metalness: 0.05,
      depthWrite: false,
    });

    const ells = ((surface.extra_overlays?.ellipsoids_2d ?? []) as Array<{
      class_idx: number;
      center_xy: [number, number];
      radii_1sigma: [number, number];
      rotation_angle_rad: number;
    }>);
    const rings: EllipsoidScene['rings'] = [];
    const axesPts: number[] = [];
    const centerPts: number[] = [];
    ells.forEach((e) => {
      const k = e.class_idx;
      const [cx, cy] = e.center_xy;
      const sx = toSX(cx);
      const sz = toSZ(cy);
      const [r1a, r1b] = e.radii_1sigma;
      const [rmA, rmB] = sigmaRadii([r1a, r1b], sigmaLevel);
      const rxS = (rmA / spanX) * 5;
      const rzS = (rmB / spanY) * 5;
      const rx1 = (r1a / spanX) * 5;
      const rz1 = (r1b / spanY) * 5;
      const ang = e.rotation_angle_rad;
      const mk = (rx: number, rz: number, y: number): THREE.BufferGeometry => {
        const pts: THREE.Vector3[] = [];
        for (let s = 0; s <= 72; s += 1) {
          const [ex, ey] = ellipsePoint(0, 0, rx, rz, ang, (s / 72) * Math.PI * 2);
          pts.push(new THREE.Vector3(sx + ex, y, sz + ey));
        }
        return new THREE.BufferGeometry().setFromPoints(pts);
      };
      rings.push({ geo: mk(rxS, rzS, 0.06), k });
      rings.push({ geo: mk(rx1, rz1, 0.03), k });
      // Principal eigen-axes (major + minor) through the center.
      const dirMajor: [number, number] = [Math.cos(ang), Math.sin(ang)];
      const dirMinor: [number, number] = [-Math.sin(ang), Math.cos(ang)];
      axesPts.push(
        sx - dirMajor[0] * rxS, 0.06, sz - dirMajor[1] * rxS,
        sx + dirMajor[0] * rxS, 0.06, sz + dirMajor[1] * rxS,
        sx - dirMinor[0] * rzS, 0.06, sz - dirMinor[1] * rzS,
        sx + dirMinor[0] * rzS, 0.06, sz + dirMinor[1] * rzS,
      );
      centerPts.push(sx, 0.1, sz);
    });
    const axesGeo = new THREE.BufferGeometry();
    axesGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(axesPts), 3));
    const centersGeo = new THREE.BufferGeometry();
    centersGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(centerPts), 3));

    // Laser tethers from the live query ball to every class mean.
    let queryPos = new THREE.Vector3(0, 0.1, 0);
    let tethersGeo = new THREE.BufferGeometry();
    let tetherCount = 0;
    const live = props.liveFeatures;
    if (live && live.length > 0) {
      const lx = live[axisX];
      const ly = live[axisY];
      if (Number.isFinite(lx) && Number.isFinite(ly)) {
        queryPos = new THREE.Vector3(
          Math.min(2.5, Math.max(-2.5, toSX(lx))),
          0.12,
          Math.min(2.5, Math.max(-2.5, toSZ(ly))),
        );
        if (showTethers && ells.length > 0) {
          const tp = new Float32Array(ells.length * 6);
          ells.forEach((e, i) => {
            tp[i * 6] = queryPos.x;
            tp[i * 6 + 1] = queryPos.y;
            tp[i * 6 + 2] = queryPos.z;
            tp[i * 6 + 3] = toSX(e.center_xy[0]);
            tp[i * 6 + 4] = 0.1;
            tp[i * 6 + 5] = toSZ(e.center_xy[1]);
          });
          tethersGeo = new THREE.BufferGeometry();
          tethersGeo.setAttribute('position', new THREE.BufferAttribute(tp, 3));
          tetherCount = ells.length;
        }
      }
    }

    return {
      surface: surfaceGeo,
      surfaceMat,
      rings,
      axes: axesGeo,
      centers: centersGeo,
      tethers: tethersGeo,
      queryPos,
      ringCount: rings.length,
      tetherCount,
    };
  }, [surface, props.liveFeatures, axisX, axisY, sigmaLevel, showAxes, showTethers]);

  const ringMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#f8fafc'),
        transparent: true,
        opacity: 0.55,
      }),
    [],
  );
  const axesMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#fbbf24'),
        transparent: true,
        opacity: 0.8,
      }),
    [],
  );
  const tetherMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color('#22d3ee'),
        transparent: true,
        opacity: 0.6,
      }),
    [],
  );
  const centerMaterial = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.12,
        color: new THREE.Color('#ffffff'),
        transparent: true,
        opacity: 0.95,
        sizeAttenuation: true,
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
  const ringColors = useMemo(() => scene.rings.map((r) => {
    const rgb = ellipsoidColor(r.k);
    return new THREE.LineBasicMaterial({
      color: new THREE.Color(rgb[0], rgb[1], rgb[2]),
      transparent: true,
      opacity: 0.7,
    });
  }), [scene]);
  useDisposableThreeResources(
    [scene.surface, ...scene.rings.map((r) => r.geo), scene.axes, scene.centers, scene.tethers],
    [scene.surfaceMat, ringMaterial, axesMaterial, tetherMaterial, centerMaterial, ballMaterial, ...ringColors],
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
          <ambientLight intensity={0.8} />
          <directionalLight position={[6, 8, 4]} intensity={1.15} />
          <mesh geometry={scene.surface} material={scene.surfaceMat} />
          {scene.rings.map((r, i) => (
            <lineLoop key={`${r.k}-${i}`} geometry={r.geo} material={ringColors[i] ?? ringMaterial} />
          ))}
          {showAxes && <lineSegments geometry={scene.axes} material={axesMaterial} />}
          <points geometry={scene.centers} material={centerMaterial} />
          {showTethers && scene.tetherCount > 0 && (
            <lineSegments geometry={scene.tethers} material={tetherMaterial} />
          )}
          <mesh position={scene.queryPos} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#67e8f6', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          ▲ Tilted covariance ellipsoids ({sigmaLevel}σ shells + eigen-axes)
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          {scene.ringCount} rings · {scene.tetherCount} tethers · drag to orbit
        </span>
      </div>
    </div>
  );
}

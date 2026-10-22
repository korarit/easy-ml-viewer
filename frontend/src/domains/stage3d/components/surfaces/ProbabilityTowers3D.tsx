import { useMemo } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useDisposableThreeResources } from '../../hooks/useDisposableThreeResources';
import { useDiscreteNbSliceStore } from '../../../model_engine/store/useDiscreteNbSliceStore';
import { towerColor, towerHeight } from './discreteNbSurfaceMath';

/**
 * Phase 25 — non-negative count-domain probability surface + per-class
 * 3D probability towers (one crystal column per class).
 *
 * Own tiny WebGL canvas (no dataset/stage context needed): the posterior
 * surface of the resolved target class spans the X>=0, Y>=0 quadrant
 * with translucent guard walls on the X=0 / Y=0 edges, while a row of
 * towers on the right encodes the frame posterior P(C_k | x_t) — the
 * winner tower glows with a floating crown marker.
 */

export interface ProbabilityTowers3DProps {
  liveFeatures?: number[] | null;
}

interface TowerScene {
  surface: THREE.BufferGeometry;
  surfaceMat: THREE.Material;
  walls: THREE.BufferGeometry[];
  towers: Array<{ geo: THREE.BufferGeometry; mat: THREE.Material; k: number }>;
  crowns: THREE.BufferGeometry;
  queryPos: THREE.Vector3;
  towerCount: number;
  winner: number;
}

export function ProbabilityTowers3D(props: ProbabilityTowers3DProps) {
  const surface = useDiscreteNbSliceStore((s) => s.surface);
  const trace = useDiscreteNbSliceStore((s) => s.traces[0] ?? null);
  const showTowers = useDiscreteNbSliceStore((s) => s.showTowers);
  const showWalls = useDiscreteNbSliceStore((s) => s.showWalls);
  const axisX = useDiscreteNbSliceStore((s) => s.axisX);
  const axisY = useDiscreteNbSliceStore((s) => s.axisY);

  const scene: TowerScene = useMemo(() => {
    const empty: TowerScene = {
      surface: new THREE.BufferGeometry(),
      surfaceMat: new THREE.MeshStandardMaterial(),
      walls: [],
      towers: [],
      crowns: new THREE.BufferGeometry(),
      queryPos: new THREE.Vector3(0, 0.1, 0),
      towerCount: 0,
      winner: 0,
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

    // Posterior surface of the resolved target class (already P(C|x) in [0, 1]).
    const posArr = new Float32Array(res * res * 3);
    for (let r = 0; r < res; r += 1) {
      for (let c = 0; c < res; c += 1) {
        const idx = r * res + c;
        const z = Number.isFinite(surface.z_values[idx]) ? Math.min(1, Math.max(0, surface.z_values[idx])) : 0;
        posArr[idx * 3] = ((xs[c] - x0) / spanX) * 5 - 2.5;
        posArr[idx * 3 + 1] = z * 2;
        posArr[idx * 3 + 2] = ((ys[r] - y0) / spanY) * 5 - 2.5;
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
      opacity: 0.5,
      side: THREE.DoubleSide,
      roughness: 0.55,
      metalness: 0.05,
      depthWrite: false,
    });

    // Guard walls on the X=0 / Y=0 count-domain edges.
    const walls: THREE.BufferGeometry[] = [];
    if (showWalls) {
      const mkWall = (alongX: boolean): THREE.BufferGeometry => {
        const g = new THREE.BufferGeometry();
        const h = 2.2;
        const v = alongX
          ? new Float32Array([-2.5, 0, -2.5, 2.5, 0, -2.5, 2.5, h, -2.5, -2.5, 0, -2.5, 2.5, h, -2.5, -2.5, h, -2.5])
          : new Float32Array([-2.5, 0, -2.5, -2.5, 0, 2.5, -2.5, h, 2.5, -2.5, 0, -2.5, -2.5, h, 2.5, -2.5, h, -2.5]);
        g.setAttribute('position', new THREE.BufferAttribute(v, 3));
        g.computeVertexNormals();
        return g;
      };
      walls.push(mkWall(true), mkWall(false));
    }

    // Probability towers: one box per class, base pivot at Z=0.
    const mech = trace?.extra_mechanics as unknown as
      | { tower_heights?: number[]; winning_class_idx?: number; classes?: string[] }
      | null
      | undefined;
    const heights = Array.isArray(mech?.tower_heights) ? (mech?.tower_heights as number[]) : [];
    const nClasses = heights.length > 0 ? heights.length : 0;
    const winner = Number.isFinite(mech?.winning_class_idx) ? (mech?.winning_class_idx as number) : 0;
    const towers: TowerScene['towers'] = [];
    const laneX = 3.4;
    heights.forEach((hRaw, k) => {
      const h = towerHeight(hRaw) * 2;
      const geo = new THREE.BoxGeometry(0.55, 1, 0.55);
      geo.translate(0, 0, 0.5);
      const rgb = towerColor(k);
      const isWin = k === winner;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgb[0], rgb[1], rgb[2]),
        emissive: new THREE.Color(rgb[0], rgb[1], rgb[2]),
        emissiveIntensity: isWin ? 0.9 : 0.25,
        transparent: true,
        opacity: isWin ? 0.95 : 0.7,
        roughness: 0.3,
        metalness: 0.1,
      });
      const m = new THREE.Mesh(geo, mat);
      m.position.set(laneX + k * 0.85, 0, 2.2);
      m.scale.set(1, Math.max(h, 0.02), 1);
      m.rotation.x = -Math.PI / 2;
      // Bake transform into geometry so no per-frame updates are needed.
      m.updateMatrix();
      geo.applyMatrix4(m.matrix);
      towers.push({ geo, mat, k });
    });

    // Crown marker floating above the winner tower.
    let crowns = new THREE.BufferGeometry();
    if (showTowers && nClasses > 0) {
      const cp = new Float32Array(3);
      cp[0] = laneX + winner * 0.85;
      cp[1] = towerHeight(heights[winner] ?? 0) * 2 + 0.35;
      cp[2] = 2.2;
      crowns = new THREE.BufferGeometry();
      crowns.setAttribute('position', new THREE.BufferAttribute(cp, 3));
    }

    // Query ball at the live feature point.
    let queryPos = new THREE.Vector3(0, 0.1, 0);
    const live = props.liveFeatures;
    if (live && live.length > 0) {
      const lx = live[axisX];
      const ly = live[axisY];
      if (Number.isFinite(lx) && Number.isFinite(ly)) {
        const qx = ((lx - x0) / spanX) * 5 - 2.5;
        const qz = ((ly - y0) / spanY) * 5 - 2.5;
        queryPos = new THREE.Vector3(
          Math.min(2.5, Math.max(-2.5, qx)),
          0.12,
          Math.min(2.5, Math.max(-2.5, qz)),
        );
      }
    }

    return { surface: surfaceGeo, surfaceMat, walls, towers, crowns, queryPos, towerCount: towers.length, winner };
  }, [surface, trace, props.liveFeatures, axisX, axisY, showTowers, showWalls]);

  const wallMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#f59e0b'),
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [],
  );
  const crownMaterial = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.22,
        color: new THREE.Color('#fef08a'),
        transparent: true,
        opacity: 1.0,
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
  useDisposableThreeResources(
    [scene.surface, ...scene.walls, ...scene.towers.map((t) => t.geo), scene.crowns],
    [scene.surfaceMat, wallMaterial, crownMaterial, ballMaterial, ...scene.towers.map((t) => t.mat)],
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
          camera={{ position: [5.5, 4.2, 6.5], fov: 42, near: 0.1, far: 100 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
        >
          <ambientLight intensity={0.8} />
          <directionalLight position={[6, 8, 4]} intensity={1.15} />
          <mesh geometry={scene.surface} material={scene.surfaceMat} />
          {showWalls && scene.walls.map((w, i) => <mesh key={i} geometry={w} material={wallMaterial} />)}
          {showTowers && scene.towers.map((t) => <mesh key={t.k} geometry={t.geo} material={t.mat} />)}
          {showTowers && scene.towerCount > 0 && <points geometry={scene.crowns} material={crownMaterial} />}
          <mesh position={scene.queryPos} material={ballMaterial}>
            <sphereGeometry args={[0.07, 20, 20]} />
          </mesh>
          <OrbitControls enableDamping={false} />
        </Canvas>
        <span
          className="font-mono-tabular pointer-events-none absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px]"
          style={{ background: 'rgba(34,211,238,0.12)', color: '#67e8f6', border: '1px solid rgba(34,211,238,0.4)' }}
        >
          ▲ Non-negative count surface P(C|x) · X≥0 Y≥0
          {showTowers && scene.towerCount > 0 ? ` · ${scene.towerCount} towers · winner ${scene.winner} ♛` : ''}
        </span>
        <span
          className="font-mono-tabular pointer-events-none absolute bottom-1 left-2 text-[10px]"
          style={{ color: '#94a3b8' }}
        >
          {scene.towerCount} towers · drag to orbit
        </span>
      </div>
    </div>
  );
}

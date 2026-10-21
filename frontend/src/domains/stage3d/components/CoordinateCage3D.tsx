import { useMemo } from 'react';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import { useStage3DStore } from '../store/useStage3DStore';
import { useDisposableThreeResources } from '../hooks/useDisposableThreeResources';
import { GROUND_WORLD_Z, formatTickLabel, worldToThree } from './stage3dSpace';

const AXIS_COLORS = { x: '#f43f5e', y: '#22d3ee', z: '#a3e635' };

function buildAxisLines(): THREE.BufferGeometry {
  // Cube edges [-5,+5]^2 on the ground plane + 3 main axes through the center.
  const g = GROUND_WORLD_Z;
  const pts: number[] = [];
  const push = (ax: number, ay: number, az: number, bx: number, by: number, bz: number) => {
    const [a1, a2, a3] = worldToThree(ax, ay, az);
    const [b1, b2, b3] = worldToThree(bx, by, bz);
    pts.push(a1, a2, a3, b1, b2, b3);
  };
  // Ground square outline.
  push(-5, -5, g, 5, -5, g);
  push(5, -5, g, 5, 5, g);
  push(5, 5, g, -5, 5, g);
  push(-5, 5, g, -5, -5, g);
  // 3 main axes (X red, Y cyan, Z lime) from -5 to +5.
  push(-5, 0, g, 5, 0, g);
  push(0, -5, g, 0, 5, g);
  push(0, 0, -5, 0, 0, 5);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3));
  return geo;
}

/** Normalized [-5,+5] coordinate cage, ground grid, colored axes + raw tick labels. */
export function CoordinateCage3D() {
  const axes = useStage3DStore((s) => s.axes);
  const showTickLabels = useStage3DStore((s) => s.showTickLabels);
  const showDropLines = useStage3DStore((s) => s.showDropLines);

  const cageGeo = useMemo(() => buildAxisLines(), []);
  const cageMat = useMemo(
    () => new THREE.LineBasicMaterial({ color: '#5b6b7f', transparent: true, opacity: 0.55 }),
    [],
  );
  useDisposableThreeResources([cageGeo], [cageMat]);

  const ticks = useMemo(() => {
    if (!showTickLabels || !axes.x || !axes.y || !axes.z) return [];
    const out: Array<{ key: string; pos: [number, number, number]; text: string; color: string }> = [];
    const defs = [
      { meta: axes.x, fixed: { wy: -5.4, wz: GROUND_WORLD_Z }, vary: 'x' as const, color: AXIS_COLORS.x },
      { meta: axes.y, fixed: { wx: 5.6, wz: GROUND_WORLD_Z }, vary: 'y' as const, color: AXIS_COLORS.y },
      { meta: axes.z, fixed: { wx: -5.6, wy: -5.4 }, vary: 'z' as const, color: AXIS_COLORS.z },
    ];
    for (const d of defs) {
      d.meta.tick_world_positions.forEach((w, i) => {
        const raw = d.meta!.tick_raw_labels[i] ?? 0;
        let wx = 0;
        let wy = 0;
        let wz = 0;
        if (d.vary === 'x') {
          wx = w;
          wy = d.fixed.wy;
          wz = d.fixed.wz;
        } else if (d.vary === 'y') {
          wx = d.fixed.wx;
          wy = w;
          wz = d.fixed.wz;
        } else {
          wx = d.fixed.wx;
          wy = d.fixed.wy;
          wz = w;
        }
        out.push({
          key: `${d.vary}-${i}`,
          pos: worldToThree(wx, wy, wz),
          text: formatTickLabel(raw),
          color: d.color,
        });
      });
    }
    return out;
  }, [axes, showTickLabels]);

  const axisName = (m: { axis_name: string } | null, fallback: string) => m?.axis_name ?? fallback;

  return (
    <group>
      {/* Ground grid at the cube floor (three-space y = -5). */}
      {showDropLines && (
        <gridHelper args={[10, 10, '#3b4a5e', '#223040']} position={[0, GROUND_WORLD_Z, 0]} />
      )}
      <lineSegments geometry={cageGeo} material={cageMat} />
      {ticks.map((t) => (
        <Html key={t.key} position={t.pos} center distanceFactor={22} zIndexRange={[10, 0]}>
          <div
            className="font-mono-tabular rounded px-1 text-[10px] leading-4"
            style={{ background: 'rgba(10,15,22,0.72)', color: t.color, whiteSpace: 'nowrap' }}
          >
            {t.text}
          </div>
        </Html>
      ))}
      {showTickLabels && axes.x && axes.y && axes.z && (
        <>
          <Html position={worldToThree(5.6, -5.4, GROUND_WORLD_Z)} center distanceFactor={22} zIndexRange={[10, 0]}>
            <div className="text-[11px] font-semibold" style={{ color: AXIS_COLORS.x }}>
              X · {axisName(axes.x, 'x')}
            </div>
          </Html>
          <Html position={worldToThree(5.6, 5.6, GROUND_WORLD_Z)} center distanceFactor={22} zIndexRange={[10, 0]}>
            <div className="text-[11px] font-semibold" style={{ color: AXIS_COLORS.y }}>
              Y · {axisName(axes.y, 'y')}
            </div>
          </Html>
          <Html position={worldToThree(-5.6, -5.4, 5.8)} center distanceFactor={22} zIndexRange={[10, 0]}>
            <div className="text-[11px] font-semibold" style={{ color: AXIS_COLORS.z }}>
              Z · {axisName(axes.z, 'z')}
            </div>
          </Html>
        </>
      )}
    </group>
  );
}

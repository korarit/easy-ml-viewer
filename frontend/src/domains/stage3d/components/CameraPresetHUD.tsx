import { useEffect } from 'react';
import { useStage3DStore, CAMERA_PRESET_KEYS, type CameraPresetMode } from '../store/useStage3DStore';

const PRESET_LABELS: Array<{ mode: CameraPresetMode; key: string; label: string; title: string }> = [
  { mode: 'isometric', key: '1', label: '1: Iso 3D', title: 'Isometric 3D (key 1)' },
  { mode: 'top_xy', key: '2', label: '2: Top X-Y', title: 'Top-down X-Y map (key 2)' },
  { mode: 'front_xz', key: '3', label: '3: Front X-Z', title: 'Front profile X vs prediction (key 3)' },
  { mode: 'side_yz', key: '4', label: '4: Side Y-Z', title: 'Side profile Y vs prediction (key 4)' },
  { mode: 'follow_ball', key: '5', label: '5: Follow Ball', title: 'Auto-track active ball (key 5)' },
];

/** Floating 1-click camera preset bar (top-right of the 3D viewport). */
export function CameraPresetHUD() {
  const cameraPreset = useStage3DStore((s) => s.cameraPreset);
  const setCameraPreset = useStage3DStore((s) => s.setCameraPreset);
  const resetCamera = useStage3DStore((s) => s.resetCamera);
  const toggleLayer = useStage3DStore((s) => s.toggleLayer);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA');
      if (typing) return;
      const preset = CAMERA_PRESET_KEYS[e.key];
      if (preset) {
        e.preventDefault();
        useStage3DStore.getState().setCameraPreset(preset);
        return;
      }
      if (e.key === 'r' || e.key === 'R') {
        useStage3DStore.getState().resetCamera();
      } else if (e.key === 'g' || e.key === 'G') {
        useStage3DStore.getState().toggleLayer('showDropLines');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const pill = (active: boolean): React.CSSProperties => ({
    background: active ? 'var(--ml-primary)' : 'var(--surface-elevated)',
    color: active ? '#fff' : 'var(--text-primary)',
    border: '1px solid var(--border-subtle)',
  });

  return (
    <div className="pointer-events-auto absolute top-2 right-2 flex max-w-[46%] flex-wrap justify-end gap-1">
      {PRESET_LABELS.map((p) => (
        <button
          key={p.mode}
          title={p.title}
          onClick={() => setCameraPreset(p.mode)}
          className="font-mono-tabular rounded px-2 py-1 text-[11px]"
          style={pill(cameraPreset === p.mode)}
        >
          {p.label}
        </button>
      ))}
      <button
        title="Reset camera zoom & pan (key R)"
        onClick={resetCamera}
        className="font-mono-tabular rounded px-2 py-1 text-[11px]"
        style={pill(false)}
      >
        R: Reset
      </button>
      <button
        title="Toggle drop-lines & grid (key G)"
        onClick={() => toggleLayer('showDropLines')}
        className="font-mono-tabular rounded px-2 py-1 text-[11px]"
        style={pill(useStage3DStore.getState().showDropLines)}
      >
        G: Drops
      </button>
    </div>
  );
}

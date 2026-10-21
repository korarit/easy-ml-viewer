import { useStage3DStore } from '../../stage3d/store/useStage3DStore';

const BUTTONS = [
  { key: 'iso' as const, label: '1:Iso', title: 'Isometric 3D (key 1)', mode: 'isometric' as const },
  { key: 'top' as const, label: '2:Top', title: 'Top-Down X-Y (key 2)', mode: 'top_xy' as const },
  { key: 'front' as const, label: '3:Front', title: 'Front X-Z (key 3)', mode: 'front_xz' as const },
  { key: 'focus' as const, label: '0:Focus', title: 'Follow active ball (key 0)', mode: 'follow_ball' as const },
];

/** Phase 14 — bottom camera preset toolbar (spec §2.4 + DoD `1/2/3/0` keys). */
export function CameraPresetToolbar() {
  const preset = useStage3DStore((s) => s.cameraPreset);
  const setPreset = useStage3DStore((s) => s.setCameraPreset);
  return (
    <div className="flex flex-wrap items-center justify-center gap-1.5" aria-label="Camera presets">
      <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Camera:
      </span>
      {BUTTONS.map((b) => (
        <button
          key={b.key}
          title={b.title}
          onClick={() => setPreset(b.mode)}
          className="font-mono-tabular rounded px-2 py-1 text-[11px]"
          style={{
            background: preset === b.mode ? 'var(--ml-primary)' : 'var(--surface-elevated)',
            color: preset === b.mode ? '#fff' : 'var(--text-primary)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          {b.label}
        </button>
      ))}
    </div>
  );
}


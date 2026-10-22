export interface LassoSparsityBadge2DProps {
  isXZeroed: boolean;
  isYZeroed: boolean;
  bothZeroed: boolean;
  xFeature: string;
  yFeature: string;
  visible?: boolean;
}

/**
 * Phase 42 — L1 sparsity indicators. A zeroed axis means the boundary runs
 * exactly parallel to that axis (the feature has no say); both zeroed means
 * there is no drawable boundary at all (∅ warning).
 */
export function LassoSparsityBadge2D({
  isXZeroed,
  isYZeroed,
  bothZeroed,
  xFeature,
  yFeature,
  visible = true,
}: LassoSparsityBadge2DProps) {
  if (!visible || (!isXZeroed && !isYZeroed)) return null;
  const notes: string[] = [];
  if (bothZeroed) {
    notes.push('∅ Sparsity warning: both visualized features zeroed out by L1 penalty');
  } else {
    if (isXZeroed) notes.push(`Axis X (${xFeature}) zeroed out by L1 penalty — boundary ∥ X axis`);
    if (isYZeroed) notes.push(`Axis Y (${yFeature}) zeroed out by L1 penalty — boundary ∥ Y axis`);
  }
  return (
    <g aria-label="L1 sparsity badges">
      {notes.map((note, i) => (
        <g key={i}>
          <rect
            x={-5}
            y={3.4 - i * 0.62}
            width={note.length * 0.145 + 0.5}
            height={0.5}
            rx={0.08}
            fill="rgba(251,191,36,0.14)"
            stroke="#fbbf24"
            strokeWidth={0.03}
          />
          <text x={-4.78} y={3.76 - i * 0.62} fontSize={0.3} fill="#fbbf24">
            {note}
          </text>
          <title>{note}</title>
        </g>
      ))}
    </g>
  );
}

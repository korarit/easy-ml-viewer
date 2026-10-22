import { wy } from '../DecisionBoundaryContour2D';
import { scaleShapVector } from './boosting2dMath';

export interface XgboostTreeShapVector2DProps {
  originX: number;
  originY: number;
  phiX: number;
  phiY: number;
  /** Backend SHAP source label (xgb_pred_contribs | lgbm_pred_contrib | catboost_shap). */
  method?: string;
  /** Arrow color per family: xgb rose, lgbm amber, catboost violet. */
  tone?: string;
  visible?: boolean;
}

/**
 * Phase 46 — 2D TreeSHAP force push vector (dev-plan §5.2 prototype).
 *
 * The native C++ TreeSHAP slice `(phi_1, phi_2)` fires from the active data
 * point: +phi_x pushes right (toward approve), -phi_y pushes down (toward
 * deny). Length is clamped (unbounded log-odds fit a ±5 stage) while the
 * label always shows the exact raw magnitudes.
 */
export function XgboostTreeShapVector2D({
  originX,
  originY,
  phiX,
  phiY,
  method = 'xgb_pred_contribs',
  tone = '#f43f5e',
  visible = true,
}: XgboostTreeShapVector2DProps) {
  if (!visible) return null;

  const { dx, dy, magnitude } = scaleShapVector(phiX, phiY);
  const targetX = originX + dx;
  const targetY = originY + dy;
  const shortMethod = method.replace('pred_contrib', 'SHAP').replace('pred_', '');

  return (
    <g className="xgboost-shap-vector-2d pointer-events-none" aria-label="2D TreeSHAP force vector">
      {/* 1. เส้นลูกศรแรงผลัก 2D TreeSHAP */}
      <line
        x1={originX}
        y1={wy(originY)}
        x2={targetX}
        y2={wy(targetY)}
        stroke={tone}
        strokeWidth={0.09}
        strokeLinecap="round"
        markerEnd="url(#arrowhead-boost-shap)"
      />
      <defs>
        <marker
          id="arrowhead-boost-shap"
          markerWidth="6"
          markerHeight="6"
          refX="4.5"
          refY="2.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L5,2.5 L0,5 Z" fill={tone} />
        </marker>
      </defs>
      {magnitude >= 1e-12 && (
        <circle cx={originX} cy={wy(originY)} r={0.06} fill={tone} opacity={0.9}>
          <title>{`SHAP origin (phi_x=${phiX}, phi_y=${phiY})`}</title>
        </circle>
      )}
      {/* 2. ป้ายตัวเลขกำกับขนาดแรงผลัก */}
      <text
        x={targetX + 0.12}
        y={wy(targetY) - 0.12}
        fill={tone}
        fontSize={0.32}
        fontFamily="JetBrains Mono, monospace"
      >
        {`SHAP (${phiX >= 0 ? '+' : ''}${phiX.toFixed(2)}, ${phiY >= 0 ? '+' : ''}${phiY.toFixed(2)}) |${magnitude.toFixed(2)}| ${shortMethod}`}
      </text>
    </g>
  );
}

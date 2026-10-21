import {
  formatContribution,
  formatRawValue,
  isOutlierPercentile,
  type FrameExplanation,
} from '../types/explainability.types';

/**
 * Phase 12 — current-frame input inspector + percentile mini-bars.
 *
 * Each row shows the raw value (scientific notation past 4 decimals,
 * edge #6), z-score, and a `0%..100%` percentile bar with a median tick
 * at 50% plus a `P99/P01` outlier badge.
 */
export function FrameFeatureInspector({ explanation }: { explanation: FrameExplanation | null }) {
  if (!explanation) {
    return (
      <div data-testid="feature-inspector-empty">
        No frame selected — pick a row to inspect feature percentiles.
      </div>
    );
  }

  return (
    <table data-testid="feature-inspector-table">
      <thead>
        <tr>
          <th>Feature</th>
          <th>Raw Value</th>
          <th>Pct</th>
          <th>Dist</th>
        </tr>
      </thead>
      <tbody>
        {explanation.attributions.map((attr) => (
          <tr key={attr.feature_index} data-testid={`inspector-row-${attr.feature_name}`}>
            <td>{attr.feature_name}</td>
            <td className="font-mono-tabular">{formatRawValue(attr.raw_value)}</td>
            <td className="font-mono-tabular">
              P{Math.round(attr.percentile)}
              {isOutlierPercentile(attr.percentile) && (
                <span style={{ color: '#f59e0b', marginLeft: 4 }}>⚠ Outlier</span>
              )}
            </td>
            <td>
              <div
                data-testid={`percentile-bar-${attr.feature_name}`}
                style={{
                  position: 'relative',
                  height: 8,
                  minWidth: 80,
                  background: 'rgba(148,163,184,0.15)',
                  borderRadius: 4,
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: 0,
                    bottom: 0,
                    width: 1,
                    background: 'rgba(226,232,240,0.6)',
                  }}
                />
                <div
                  data-testid={`percentile-fill-${attr.feature_name}`}
                  style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${Math.max(0, Math.min(100, attr.percentile)).toFixed(1)}%`,
                    background: isOutlierPercentile(attr.percentile) ? '#f59e0b' : '#38bdf8',
                    borderRadius: 4,
                  }}
                />
                <span className="font-mono-tabular" style={{ marginLeft: 4 }}>
                  z={Number.isFinite(attr.z_score) ? attr.z_score.toFixed(2) : '—'} (
                  {formatContribution(attr.contribution)})
                </span>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

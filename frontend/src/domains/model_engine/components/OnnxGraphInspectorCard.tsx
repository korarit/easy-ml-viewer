import type { OnnxTraceMechanics } from '../api/onnxClassicalApi';

/**
 * Phase 38 — ONNX graph + tensor inspector card: operator mini-graph
 * (`ai.onnx.ml` node chain) and the input/output tensor signature table.
 */
export function OnnxGraphInspectorCard({ mech }: { mech: OnnxTraceMechanics | null }) {
  if (!mech) {
    return (
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        Run a trace to inspect the ONNX graph operators and tensor signatures.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-2 text-xs">
      <div
        className="font-mono-tabular flex flex-wrap items-center gap-1.5 rounded px-2 py-1.5 text-[11px]"
        style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
        title="ONNX graph operator chain"
      >
        {mech.node_chain.map((op, i) => (
          <span key={`${op}-${i}`} className="flex items-center gap-1.5">
            {i > 0 && <span style={{ color: 'var(--text-muted)' }}>➔</span>}
            <span
              className="rounded px-1.5 py-0.5"
              style={{
                border: '1px solid var(--border-strong)',
                background: op.includes('ai.onnx.ml') ? 'rgba(52,211,153,0.12)' : 'transparent',
              }}
            >
              {op}
            </span>
          </span>
        ))}
      </div>
      <div className="overflow-auto">
        <table className="w-full text-left text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <thead>
            <tr style={{ color: 'var(--text-muted)' }}>
              <th className="font-mono-tabular px-2 py-1 font-medium">Tensor</th>
              <th className="font-mono-tabular px-2 py-1 font-medium">Shape</th>
              <th className="font-mono-tabular px-2 py-1 font-medium">Dtype</th>
            </tr>
          </thead>
          <tbody className="font-mono-tabular">
            {mech.input_signature.map((t) => (
              <tr key={t.name} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                <td className="px-2 py-1">{t.name}</td>
                <td className="px-2 py-1">
                  [{t.shape.map((d) => (d == null ? 'batch' : d)).join(', ')}]
                </td>
                <td className="px-2 py-1">{t.dtype}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        ZipMap: {mech.has_zipmap ? (mech.zipmap_bypass_active ? 'bypassed in-memory (raw float32 tensor)' : 'fallback list[dict] unpacker') : 'absent'} · opset{' '}
        {mech.opset_version}
      </p>
    </div>
  );
}

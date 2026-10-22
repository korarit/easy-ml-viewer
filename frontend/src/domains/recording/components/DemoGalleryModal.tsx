import { useMemo, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { STORY_DEMOS, useGuidedDemo } from '../../studio-ux/hooks/useGuidedDemo';

/**
 * Phase 40 — Instant 20-Model Demo Gallery (onboarding).
 * Mirrors `backend/scripts/generate_all_20_demos.py`: one compact binary
 * demo per model family. Demos are generated offline via
 * `python scripts/generate_all_20_demos.py` into `samples/demos/`.
 */
interface DemoCard {
  family: string;
  category: string;
  modelFile: string;
  task: string;
  blurb: string;
}

const DEMO_CATALOG: DemoCard[] = [
  { family: 'linear', category: 'Linear / Generalized Linear', modelFile: 'demo_linear.joblib', task: 'regression', blurb: 'OLS tilted hyperplane + residual drop-lines' },
  { family: 'logistic', category: 'Linear / Generalized Linear', modelFile: 'demo_logistic.joblib', task: 'binary', blurb: 'Sigmoid S-surface + class boundary' },
  { family: 'ridge', category: 'Linear / Generalized Linear', modelFile: 'demo_ridge.joblib', task: 'binary', blurb: 'L2-shrunk weight vectors' },
  { family: 'lasso', category: 'Linear / Generalized Linear', modelFile: 'demo_lasso.joblib', task: 'regression', blurb: 'Sparse plane, zeroed-out axes' },
  { family: 'elasticnet', category: 'Linear / Generalized Linear', modelFile: 'demo_elasticnet.joblib', task: 'regression', blurb: 'L1/L2 penalty-ratio ring' },
  { family: 'sgd', category: 'Linear / Generalized Linear', modelFile: 'demo_sgd.joblib', task: 'binary', blurb: 'Linear margin boundary plane' },
  { family: 'knn', category: 'Distance / Kernel', modelFile: 'demo_knn.joblib', task: 'binary', blurb: 'Expanding radar sphere + spider lines' },
  { family: 'svm', category: 'Distance / Kernel', modelFile: 'demo_svm.joblib', task: 'binary', blurb: 'RBF saddle + support-vector pillars' },
  { family: 'gnb', category: 'Probabilistic', modelFile: 'demo_gnb.joblib', task: 'binary', blurb: 'Gaussian bell hills' },
  { family: 'multinomial_nb', category: 'Probabilistic', modelFile: 'demo_multinomial_nb.joblib', task: 'binary', blurb: 'Log-likelihood towers (counts)' },
  { family: 'bernoulli_nb', category: 'Probabilistic', modelFile: 'demo_bernoulli_nb.joblib', task: 'binary', blurb: 'Binary-feature evidence towers' },
  { family: 'lda', category: 'Probabilistic', modelFile: 'demo_lda.joblib', task: 'binary', blurb: 'Shared-covariance ellipsoids' },
  { family: 'qda', category: 'Probabilistic', modelFile: 'demo_qda.joblib', task: 'binary', blurb: 'Per-class covariance ellipsoids' },
  { family: 'tree', category: 'Trees / Bagging', modelFile: 'demo_tree.joblib', task: 'binary', blurb: 'Voxel stepped terraces + glowing path' },
  { family: 'forest', category: 'Trees / Bagging', modelFile: 'demo_forest.joblib', task: 'binary', blurb: 'Multi-tree voting arena' },
  { family: 'extratrees', category: 'Trees / Bagging', modelFile: 'demo_extratrees.joblib', task: 'binary', blurb: 'Randomized-split variance halo' },
  { family: 'adaboost', category: 'Trees / Bagging', modelFile: 'demo_adaboost.joblib', task: 'binary', blurb: 'Sequential booster cascade' },
  { family: 'sklearn_gbm', category: 'Gradient Boosting', modelFile: 'demo_sklearn_gbm.joblib', task: 'binary', blurb: 'Residual accumulation landscape' },
  { family: 'hist_gbm', category: 'Gradient Boosting', modelFile: 'demo_hist_gbm.joblib', task: 'binary', blurb: 'Quantized histogram-bin terraces' },
  { family: 'xgboost', category: 'Gradient Boosting', modelFile: 'demo_xgboost.joblib', task: 'binary', blurb: 'Boosted leaf flow (+ .ubj native)' },
  { family: 'lightgbm', category: 'Gradient Boosting', modelFile: 'demo_lightgbm.joblib', task: 'binary', blurb: 'Leaf-wise split hierarchy (+ .txt native)' },
  { family: 'catboost', category: 'Gradient Boosting', modelFile: 'demo_catboost.joblib', task: 'binary', blurb: 'Symmetric oblivious grid (+ .cbm native)' },
  { family: 'mlp', category: 'Neural / Pipeline / ONNX', modelFile: 'demo_mlp.joblib', task: 'binary', blurb: 'Floating synapse brain + energy pulses' },
  { family: 'pipeline', category: 'Neural / Pipeline / ONNX', modelFile: 'demo_pipeline.joblib', task: 'binary', blurb: 'Scaler → classifier tunnel' },
  { family: 'onnx_logreg', category: 'Neural / Pipeline / ONNX', modelFile: 'demo_logreg.onnx', task: 'binary', blurb: 'ONNX classical twin (ZipMap off)' },
  { family: 'onnx_rf', category: 'Neural / Pipeline / ONNX', modelFile: 'demo_rf.onnx', task: 'binary', blurb: 'ONNX forest twin (ZipMap off)' },
];

const CATEGORIES = [...new Set(DEMO_CATALOG.map((d) => d.category))];

export function DemoGalleryModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [filter, setFilter] = useState<string>('all');
  const [copied, setCopied] = useState<string | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const { busyKey, stage, loadStory } = useGuidedDemo();

  const cards = useMemo(
    () => (filter === 'all' ? DEMO_CATALOG : DEMO_CATALOG.filter((d) => d.category === filter)),
    [filter],
  );

  if (!open) return null;

  const copyCmd = async (family: string) => {
    const cmd = `python scripts/generate_all_20_demos.py  # -> samples/demos/${family}`;
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(family);
      window.setTimeout(() => setCopied(null), 1500);
    } catch {
      /* clipboard unavailable offline — ignore */
    }
  };

  const openStory = async (key: string) => {
    const story = STORY_DEMOS.find((s) => s.key === key);
    if (!story) return;
    setDemoError(null);
    try {
      await loadStory(story);
      onClose();
    } catch (err) {
      setDemoError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(2,6,16,0.72)' }}
      role="dialog"
      aria-label="Demo gallery"
      onClick={onClose}
    >
      <div
        className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-lg p-5"
        style={{ background: 'var(--surface-panel)', border: '1px solid var(--border-subtle)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-1 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Start with a story — 1 click to 3D</h2>
          <button
            onClick={onClose}
            className="min-h-[32px] rounded px-2 py-1 text-xs"
            style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
          >
            Close
          </button>
        </div>
        <p className="mb-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Each story loads real data + a real model, matches the columns and queues 100 frames for you.
          You land in the 3D studio ready to press play.
        </p>

        <div className="mb-4 grid grid-cols-1 gap-2 md:grid-cols-3">
          {STORY_DEMOS.map((s, i) => {
            const busy = busyKey === s.key;
            return (
              <div
                key={s.key}
                className="flex flex-col gap-2 rounded-md p-3"
                style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
              >
                <span className="font-mono-tabular text-[10px] tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
                  Story {i + 1} · {s.frames}
                </span>
                <p className="text-sm font-semibold leading-snug">{s.question}</p>
                <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>{s.outcome}</p>
                <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{s.lookFor}</p>
                <div className="mt-auto pt-1">
                  <button
                    onClick={() => void openStory(s.key)}
                    disabled={busyKey !== null}
                    className="flex min-h-[36px] w-full items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50"
                    style={{ background: 'var(--ml-primary)', color: '#fff' }}
                  >
                    {busy ? (stage ?? 'Loading…') : <>Open in 3D <ArrowRight size={14} /></>}
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {demoError && (
          <div
            className="mb-3 rounded-md px-3 py-2 text-xs"
            style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}
            role="alert"
          >
            {demoError}
          </div>
        )}

        <details>
          <summary className="cursor-pointer text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
            Browse all {DEMO_CATALOG.length} model families (for developers)
          </summary>
          <p className="mb-3 mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            One compact binary demo per model family across {CATEGORIES.length} categories.
            Generate offline: <span className="font-mono-tabular">python scripts/generate_all_20_demos.py</span>
          </p>

          <div className="mb-4 flex flex-wrap gap-1.5">
            {['all', ...CATEGORIES].map((c) => (
              <button
                key={c}
                onClick={() => setFilter(c)}
                className="min-h-[32px] rounded px-2.5 py-1 text-[11px]"
                style={{
                  border: `1px solid ${filter === c ? 'var(--ml-primary)' : 'var(--border-subtle)'}`,
                  color: filter === c ? 'var(--text-primary)' : 'var(--text-secondary)',
                }}
              >
                {c === 'all' ? `All (${DEMO_CATALOG.length})` : c}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {cards.map((d) => (
              <div
                key={d.family}
                className="rounded-md p-3"
                style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border-subtle)' }}
              >
                <div className="mb-0.5 text-xs font-semibold">{d.family}</div>
                <div className="mb-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {d.category} · {d.task}
                </div>
                <div className="mb-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                  {d.blurb}
                </div>
                <div className="font-mono-tabular mb-2 truncate text-[10px]" style={{ color: 'var(--text-muted)' }} title={d.modelFile}>
                  samples/demos/{d.modelFile}
                </div>
                <button
                  onClick={() => void copyCmd(d.family)}
                  className="min-h-[32px] rounded px-2 py-1 text-[11px]"
                  style={{ border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}
                  title="Copy offline generator command"
                >
                  {copied === d.family ? 'Copied!' : 'Copy generator cmd'}
                </button>
              </div>
            ))}
          </div>
        </details>
      </div>
    </div>
  );
}

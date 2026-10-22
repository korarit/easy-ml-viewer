import { useCallback, useRef, useState } from 'react';
import { Box, Loader2, ShieldCheck } from 'lucide-react';
import { ACCEPTED_MODEL_EXTENSIONS, isAcceptedModelFile, useModelStore } from '../store/useModelStore';

export function ModelUploadDropzone() {
  const { uploadModel, loadSampleModel, isLoadingModel, currentModel } = useModelStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File | undefined) => {
      setLocalError(null);
      if (!file) return;
        if (!isAcceptedModelFile(file.name)) {
          setLocalError(`Unsupported model format. Allowed: .joblib, .pkl, .pickle, .cbm, .json, .ubj, .model, .xgb, .txt, .lgb, .onnx`);
          return;
        }
      try {
        await uploadModel(file);
      } catch {
        /* modelError in store */
      }
    },
    [uploadModel],
  );

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop model file here or browse (Ctrl+M)"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void handleFile(e.dataTransfer.files?.[0]);
        }}
        className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-8 text-center"
        style={{
          borderColor: dragOver ? 'var(--ml-primary)' : 'var(--border-strong)',
          background: dragOver ? 'rgba(59,130,246,0.08)' : 'var(--surface-panel)',
        }}
      >
        {isLoadingModel ? (
          <Loader2 size={22} className="animate-spin" style={{ color: 'var(--ml-primary)' }} />
        ) : (
          <Box size={22} style={{ color: 'var(--text-secondary)' }} />
        )}
        <p className="text-sm font-medium">
          {isLoadingModel ? 'Deserializing model & checking fitted state…' : 'Drop model file here (.joblib / .pkl / .cbm / .txt …)'}
        </p>
        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          or <span className="underline">Browse model artifact (Ctrl+M)</span>
        </p>
        <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Local pickle loading · max 1,024 MB · memory-mapped
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_MODEL_EXTENSIONS}
          className="hidden"
          onChange={(e) => {
            void handleFile(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
      </div>
      {localError && (
        <p className="mt-2 rounded-md px-3 py-2 text-xs" style={{ background: 'rgba(244,63,94,0.12)', color: 'var(--ml-class-1)' }}>
          {localError}
        </p>
      )}
      <div className="mt-3">
        <button
          disabled={isLoadingModel}
          onClick={() => void loadSampleModel('house_price_linreg').catch(() => undefined)}
          className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50"
          style={{ border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }}
        >
          Load Sample Model (House Price Linear Regression)
        </button>
      </div>
      {currentModel && (
        <p className="font-mono-tabular mt-2 flex items-center gap-1.5 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <ShieldCheck size={13} style={{ color: 'var(--ml-class-0)' }} />
          SHA-256: {currentModel.sha256_hash.slice(0, 12)}…{currentModel.sha256_hash.slice(-6)} (verified local file)
        </p>
      )}
    </div>
  );
}

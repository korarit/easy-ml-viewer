import { useCallback, useRef, useState } from 'react';
import { FileUp, Loader2 } from 'lucide-react';
import { ACCEPTED_EXTENSIONS, isAcceptedFile, useDatasetStore } from '../store/useDatasetStore';
import { useSystemStore } from '../../../core/store/useSystemStore';

const SAMPLES = [
  { name: 'house_prices', label: 'House Prices (Regression)' },
  { name: 'iris', label: 'Iris (Classification)' },
];

export function DatasetDropzone() {
  const { uploadDataset, loadSampleDataset, isUploading, uploadProgress } = useDatasetStore();
  const systemFreeMb = useSystemStore((s) => s.health?.memory.system_available_mb ?? null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File | undefined) => {
      setLocalError(null);
      if (!file) return;
      if (!isAcceptedFile(file.name)) {
        setLocalError(`Unsupported file type “${file.name}”. Allowed: .csv, .tsv, .xlsx, .xls, .parquet`);
        return;
      }
      try {
        await uploadDataset(file);
      } catch {
        /* uploadError in store drives the banner */
      }
    },
    [uploadDataset],
  );

  return (
    <div>
      <div
        className="mb-3 flex flex-wrap items-center gap-2 rounded-md px-3 py-2 text-xs"
        style={{ background: 'var(--surface-elevated)', color: 'var(--text-secondary)' }}
      >
        <span style={{ color: 'var(--text-primary)' }} className="font-medium">New here?</span>
        <span>1. Load the sample below → 2. Add a model in Step 2 → 3. Press play in Step 5.</span>
        <span className="font-mono-tabular" style={{ color: 'var(--text-muted)' }}>No signup · files never leave this machine</span>
      </div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop dataset file here or browse (Ctrl+O)"
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
        {isUploading ? (
          <Loader2 size={22} className="animate-spin" style={{ color: 'var(--ml-primary)' }} />
        ) : (
          <FileUp size={22} style={{ color: 'var(--text-secondary)' }} />
        )}
        <p className="text-sm font-medium">
          {isUploading ? (uploadProgress ?? 'Reading your file…') : 'Drop your table file here'}
        </p>
        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          or <span className="underline">browse files (Ctrl+O)</span> · .csv, .tsv, .xlsx, .parquet
        </p>
        <p className="font-mono-tabular text-[11px]" style={{ color: 'var(--text-muted)' }}>
          Max 1,024 MB{systemFreeMb !== null ? ` · ${(systemFreeMb / 1024).toFixed(1)} GB free on this machine` : ''}
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
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
      <div className="mt-3 flex flex-wrap gap-2">
        <span className="w-full text-xs" style={{ color: 'var(--text-secondary)' }}>
          No file handy? Start instantly:
        </span>
        {SAMPLES.map((s, i) => (
          <button
            key={s.name}
            disabled={isUploading}
            onClick={() => void loadSampleDataset(s.name).catch(() => undefined)}
            className="min-h-[36px] rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50"
            style={
              i === 0
                ? { background: 'var(--ml-primary)', color: '#fff' }
                : { border: '1px solid var(--border-strong)', color: 'var(--text-primary)' }
            }
          >
            Try sample: {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}

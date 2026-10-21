import { useRef, useState } from 'react';
import { fetchLoadMlv } from '../api/recordingApi';
import { useRecordingStore } from '../store/useRecordingStore';

/**
 * Phase 15 — instant `.mlv` dropzone (spec §2.2).
 * "⚡ Instant 60FPS Replay — ไม่ต้องใช้ไฟล์โมเดลหรือ CSV"
 */
export function MlvInstantDropzone({ compact = false }: { compact?: boolean }) {
  const enterReplay = useRecordingStore((s) => s.enterReplay);
  const setLoadError = useRecordingStore((s) => s.setLoadError);
  const loadError = useRecordingStore((s) => s.loadError);
  const isLoading = useRecordingStore((s) => s.isLoadingMlv);
  const setLoading = useRecordingStore((s) => s.setLoadingMlv);
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const acceptFile = async (file: File | undefined) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.mlv')) {
      setLoadError('ไฟล์นี้ไม่ใช่ .mlv (Invalid Magic Header) — กรุณาเลือกไฟล์ .mlv ที่ Bake จาก Studio');
      return;
    }
    setLoadError(null);
    setLoading(true);
    try {
      const session = await fetchLoadMlv(file);
      enterReplay(session);
    } catch (err) {
      const code = (err as { code?: string }).code ?? '';
      const msg = err instanceof Error ? err.message : 'Load failed';
      if (code === 'MLV_CORRUPTED') {
        setLoadError(`ไฟล์ .mlv เสียหาย (CRC32 Mismatch) — โปรดดาวน์โหลดไฟล์ใหม่: ${msg}`);
      } else if (code === 'MLV_INVALID_MAGIC') {
        setLoadError('ไฟล์นี้ไม่ใช่ .mlv v1.0 ที่ถูกต้อง (Invalid Magic Header)');
      } else {
        setLoadError(`เปิดไฟล์ไม่สำเร็จ: ${msg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Open .mlv recording"
        onClick={() => fileRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter') fileRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void acceptFile(e.dataTransfer.files?.[0]);
        }}
        className={`rounded-lg border-dashed px-4 text-center ${compact ? 'py-2' : 'py-4'}`}
        style={{
          border: `1px dashed ${dragOver ? 'var(--ml-primary)' : 'var(--border-strong)'}`,
          background: dragOver ? 'rgba(99,102,241,0.08)' : 'transparent',
          cursor: 'pointer',
        }}
      >
        <p className="text-xs font-medium">
          ⚡ {isLoading ? 'Loading .mlv…' : 'Instant 60FPS Replay — ลากไฟล์ .mlv มาวาง'}
        </p>
        {!compact && (
          <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            ไม่ต้องใช้ไฟล์โมเดลหรือ CSV — เปิดดู 3D Replay ทันที &lt; 1 วินาที
          </p>
        )}
        <input
          ref={fileRef}
          type="file"
          accept=".mlv"
          className="hidden"
          onChange={(e) => {
            void acceptFile(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
      </div>
      {loadError && (
        <p className="mt-2 text-xs" style={{ color: 'var(--ml-class-1)' }} role="alert">
          {loadError}
        </p>
      )}
    </div>
  );
}

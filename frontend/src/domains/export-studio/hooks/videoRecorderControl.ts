/**
 * Phase 16 — shared video-recorder control registry.
 *
 * The `ExportStudioModal` owns the single `useWebGLVideoRecorder` instance
 * (it is always mounted inside the 3D studio) and registers its
 * start/stop handlers here, so the floating `VideoRecordingIndicatorBar`
 * and the `Ctrl+Shift+R` shortcut can drive the same recorder without
 * spawning a second `MediaRecorder`.
 */

interface RecorderControls {
  startQuickRecord: () => void;
  stopRecord: () => void;
}

let controls: RecorderControls | null = null;

export function registerRecorderControls(c: RecorderControls | null): void {
  controls = c;
}

export function requestStopRecording(): void {
  controls?.stopRecord();
}

export function requestQuickRecord(): void {
  controls?.startQuickRecord();
}

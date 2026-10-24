/**
 * Phase 57 — deterministic off-screen compositor (spec §2.1, §5 step 2).
 *
 * Owns a single reusable virtual canvas (1920×1080 / 720p / square) and
 * composites one deterministic frame per `renderFrame()` call:
 *
 *   [stage image] + [Why waterfall + What-If dock] + [timeline HUD]
 *
 * The stage bitmap comes from the live 3D WebGL canvas or the live 2D SVG
 * rasterized to an ImageBitmap — the compositor never touches the on-screen
 * theme or camera, so Theme/Angle overrides stay isolated.
 */
import {
  RESOLUTION_TABLE,
  compositorLayout,
  themeTokens,
} from './advancedVideoExportMath';
import type {
  AdvancedVideoExportOptions,
  CompositorLayout,
  FrameWhatIfDatum,
  FrameWhyDatum,
} from '../types/advancedVideoExport.types';

export interface CompositeFrameInput {
  frameIndex: number;
  totalFrames: number;
  /** Stage bitmap already rendered at stage-rect size (or larger; scaled). */
  stageImage: CanvasImageSource | null;
  whyTop: FrameWhyDatum[];
  whatIf: FrameWhatIfDatum | null;
  lossBadge: string | null;
  baseFps: number;
}

function resolveCanvas(): HTMLCanvasElement | null {
  if (typeof document === 'undefined') return null;
  return document.createElement('canvas');
}

export class OfflineCompositor {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private layoutCache: CompositorLayout | null = null;

  constructor(private options: AdvancedVideoExportOptions) {}

  setOptions(next: AdvancedVideoExportOptions): void {
    this.options = next;
    this.layoutCache = null;
  }

  get width(): number {
    return RESOLUTION_TABLE[this.options.resolution].w;
  }

  get height(): number {
    return RESOLUTION_TABLE[this.options.resolution].h;
  }

  layout(): CompositorLayout {
    if (!this.layoutCache) {
      this.layoutCache = compositorLayout(this.width, this.height, this.options.layout);
    }
    return this.layoutCache;
  }

  /** Lazily creates (and reuses) the single virtual-canvas texture. */
  private ensureContext(): CanvasRenderingContext2D | null {
    if (this.ctx) return this.ctx;
    const c = resolveCanvas();
    if (!c) return null;
    c.width = this.width;
    c.height = this.height;
    const ctx = c.getContext('2d');
    if (!ctx) return null;
    this.canvas = c;
    this.ctx = ctx;
    return ctx;
  }

  getCanvas(): HTMLCanvasElement | null {
    this.ensureContext();
    if (this.canvas && (this.canvas.width !== this.width || this.canvas.height !== this.height)) {
      this.canvas.width = this.width;
      this.canvas.height = this.height;
      this.layoutCache = null;
    }
    return this.canvas;
  }

  renderFrame(input: CompositeFrameInput): HTMLCanvasElement | null {
    const ctx = this.ensureContext();
    const canvas = this.getCanvas();
    if (!ctx || !canvas) return null;
    const tokens = themeTokens(this.options.theme);
    const layout = this.layout();

    // Stage background + image.
    ctx.fillStyle = tokens.background;
    ctx.fillRect(0, 0, this.width, this.height);
    if (input.stageImage) {
      try {
        ctx.drawImage(input.stageImage, layout.stage.x, layout.stage.y, layout.stage.w, layout.stage.h);
      } catch {
        /* stage not ready — keep the themed background */
      }
    } else {
      ctx.fillStyle = tokens.grid;
      ctx.font = '20px "JetBrains Mono", monospace';
      ctx.globalAlpha = 0.5;
      ctx.fillText(`STAGE ${this.options.dimension.toUpperCase()} · f${input.frameIndex}`, layout.stage.x + 24, layout.stage.y + 48);
      ctx.globalAlpha = 1.0;
    }

    if (this.options.layout !== 'pure_stage') {
      this.paintTelemetryDock(ctx, input, tokens.hudText, tokens.hudPanel, tokens.accentLoss);
    }
    this.paintHudBar(ctx, input, tokens.hudText, tokens.hudPanel);
    return canvas;
  }

  private paintTelemetryDock(
    ctx: CanvasRenderingContext2D,
    input: CompositeFrameInput,
    text: string,
    panel: string,
    accent: string,
  ): void {
    const { telemetry } = this.layout();
    if (telemetry.w <= 0 || telemetry.h <= 0) return;
    const isPip = this.options.layout === 'pip';
    ctx.save();
    ctx.fillStyle = panel;
    if (isPip) {
      ctx.strokeStyle = 'rgba(148,163,184,0.55)';
      ctx.lineWidth = 2;
      roundRectPath(ctx, telemetry.x, telemetry.y, telemetry.w, telemetry.h, 12);
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.fillRect(telemetry.x, telemetry.y, telemetry.w, telemetry.h);
      ctx.fillStyle = 'rgba(148,163,184,0.35)';
      ctx.fillRect(telemetry.x, telemetry.y, 1, telemetry.h);
    }
    ctx.fillStyle = text;
    const pad = isPip ? 14 : 18;
    let y = telemetry.y + pad + 14;
    const maxW = telemetry.w - pad * 2;
    ctx.font = `600 ${isPip ? 15 : 18}px Inter, system-ui, sans-serif`;
    ctx.fillText('WHY · LOCAL SHAP', telemetry.x + pad, y, maxW);
    y += isPip ? 22 : 28;

    if (this.options.includeWhyPanel) {
      ctx.font = `${isPip ? 12 : 14}px "JetBrains Mono", monospace`;
      const rows = input.whyTop.slice(0, isPip ? 3 : 5);
      if (rows.length === 0) {
        ctx.globalAlpha = 0.6;
        ctx.fillText('no attribution cached', telemetry.x + pad, y, maxW);
        ctx.globalAlpha = 1.0;
        y += 20;
      }
      for (const row of rows) {
        const label = `${row.feature}`.slice(0, isPip ? 14 : 22);
        const v = Number.isFinite(row.impact) ? row.impact : 0;
        const barMax = Math.max(40, maxW - 150);
        const barW = Math.min(barMax, Math.abs(v) * barMax * 2);
        ctx.fillStyle = text;
        ctx.fillText(label, telemetry.x + pad, y, 110);
        ctx.fillStyle = v >= 0 ? '#10b981' : accent;
        ctx.fillRect(telemetry.x + pad + 116, y - 10, Math.max(2, barW), 12);
        ctx.fillStyle = text;
        ctx.fillText(`${v >= 0 ? '+' : ''}${v.toFixed(2)}`, telemetry.x + pad + 122 + barMax, y, 80);
        y += 22;
      }
      y += 6;
    }

    if (this.options.includeWhatIfPanel) {
      ctx.font = `600 ${isPip ? 15 : 18}px Inter, system-ui, sans-serif`;
      ctx.fillStyle = text;
      ctx.fillText('WHAT-IF SANDBOX', telemetry.x + pad, y, maxW);
      y += isPip ? 22 : 28;
      ctx.font = `${isPip ? 12 : 14}px "JetBrains Mono", monospace`;
      const wi = input.whatIf;
      const lines = wi
        ? [
            `Row #${wi.inputRow}  Pred ${wi.prediction.toFixed(3)}`,
            wi.flipped ? `Flip Δ ${(wi.flipDeltaPct ?? 0).toFixed(1)}%` : 'No flip vs reference',
          ]
        : ['Sandbox idle — base row'];
      for (const line of lines) {
        ctx.fillText(line.slice(0, 44), telemetry.x + pad, y, maxW);
        y += 20;
      }
    }
    if (input.lossBadge) {
      ctx.font = `${isPip ? 11 : 13}px "JetBrains Mono", monospace`;
      ctx.globalAlpha = 0.85;
      ctx.fillText(`Loss ${input.lossBadge}`.slice(0, 40), telemetry.x + pad, telemetry.y + telemetry.h - 14, maxW);
      ctx.globalAlpha = 1.0;
    }
    ctx.restore();
  }

  private paintHudBar(
    ctx: CanvasRenderingContext2D,
    input: CompositeFrameInput,
    text: string,
    panel: string,
  ): void {
    const { hudBar } = this.layout();
    ctx.save();
    ctx.fillStyle = panel;
    ctx.fillRect(hudBar.x, hudBar.y, hudBar.w, hudBar.h);
    ctx.fillStyle = text;
    ctx.font = '13px "JetBrains Mono", monospace';
    const line =
      `Frame ${String(input.frameIndex).padStart(4, '0')}/${String(Math.max(input.totalFrames - 1, 0)).padStart(4, '0')}` +
      ` · ${this.options.fps} FPS · ${this.options.speedMultiplier.toFixed(2)}x · ${this.options.cameraPreset}`;
    ctx.fillText(line.slice(0, 120), hudBar.x + 12, hudBar.y + 22, Math.max(50, hudBar.w - 24));
    ctx.restore();
  }

  dispose(): void {
    this.canvas = null;
    this.ctx = null;
    this.layoutCache = null;
  }
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

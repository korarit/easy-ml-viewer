import { useMemo } from 'react';
import { usePlaybackStore } from '../../playback/store/usePlaybackStore';
import { useStage3DStore } from '../../stage3d/store/useStage3DStore';
import { useProjectionStore } from '../../projection3d/store/useProjectionStore';
import { useWhatIfStore } from '../../whatif/store/useWhatIfStore';
import { useStudioLayoutStore } from '../store/useStudioLayoutStore';
import type { StudioCommand } from '../types/studioUx.types';

export interface CommandRegistry {
  commands: StudioCommand[];
  availableFeatures: string[];
}

/**
 * Phase 14 — built-in dynamic command registry.
 * Rebuilt when queue/axis context changes so frame jumps + axis swaps
 * always target live state (no stale closures).
 */
export function useCommandPaletteRegistry(): CommandRegistry {
  const totalFrames = usePlaybackStore((s) => s.totalFrames);
  const featureNames = useStage3DStore((s) => s.featureNames);
  const wireframe = useProjectionStore((s) => s.wireframe);

  return useMemo(() => {
    const commands: StudioCommand[] = [];
    const queue: StudioCommand[] = [];

    // Navigation
    queue.push({
      id: 'goto-first',
      title: 'Go to First Frame',
      hint: 'Jump to frame 0',
      keywords: 'first start home frame 0 row',
      category: 'Navigation',
      run: () => usePlaybackStore.getState().seekToFrame(0, true),
    });
    queue.push({
      id: 'goto-last',
      title: 'Go to Last Frame',
      hint: `Jump to frame ${Math.max(totalFrames - 1, 0)}`,
      keywords: 'last end final frame row',
      category: 'Navigation',
      run: () => usePlaybackStore.getState().seekToFrame(Math.max(totalFrames - 1, 0), true),
    });
    for (const f of [10, 50, 100, 245]) {
      if (f < totalFrames) {
        queue.push({
          id: `goto-frame-${f}`,
          title: `Go to Frame #${f}`,
          hint: `Jump playhead to frame ${f}`,
          keywords: `go goto jump frame row ${f} number`,
          category: 'Navigation',
          run: () => usePlaybackStore.getState().seekToFrame(f, true),
        });
      }
    }
    commands.push(...queue);

    // 3D Axes & Camera
    for (const name of featureNames.slice(0, 12)) {
      commands.push({
        id: `axis-x-${name}`,
        title: `Set X Axis: ${name}`,
        hint: `Rebuild stage with X = ${name}`,
        keywords: `axis x feature ${name} set change`,
        category: '3D Axes & Camera',
        run: async () => {
          const stage = useStage3DStore.getState();
          const idx = stage.featureNames.indexOf(name);
          if (idx >= 0) await stage.setAxes(idx, stage.axisYFeatureIndex);
          else useProjectionStore.getState().setXFeature(name);
        },
      });
      commands.push({
        id: `axis-y-${name}`,
        title: `Set Y Axis: ${name}`,
        hint: `Rebuild stage with Y = ${name}`,
        keywords: `axis y feature ${name} set change`,
        category: '3D Axes & Camera',
        run: async () => {
          const stage = useStage3DStore.getState();
          const idx = stage.featureNames.indexOf(name);
          if (idx >= 0) await stage.setAxes(stage.axisXFeatureIndex, idx);
          else useProjectionStore.getState().setYFeature(name);
        },
      });
    }
    const presets = [
      { id: 'iso', title: 'Camera Preset: Isometric 3D', keywords: 'camera iso isometric 3d view 1', mode: 'isometric' },
      { id: 'top', title: 'Camera Preset: Top-Down 2D', keywords: 'camera top down 2d contour map 2', mode: 'top_xy' },
      { id: 'front', title: 'Camera Preset: Front X-Z', keywords: 'camera front profile xz 3', mode: 'front_xz' },
      { id: 'side', title: 'Camera Preset: Side Y-Z', keywords: 'camera side profile yz', mode: 'side_yz' },
      { id: 'follow', title: 'Camera Preset: Follow Ball', keywords: 'camera follow ball track focus 0 5', mode: 'follow_ball' },
      { id: 'reset', title: 'Camera Preset: Reset View', keywords: 'camera reset view home r', mode: 'isometric' },
    ] as const;
    for (const p of presets) {
      commands.push({
        id: `camera-${p.id}`,
        title: p.title,
        hint: 'Smooth-slerp camera transition',
        keywords: p.keywords,
        category: '3D Axes & Camera',
        run: () => {
          if (p.id === 'reset') useStage3DStore.getState().resetCamera();
          else useStage3DStore.getState().setCameraPreset(p.mode);
        },
      });
    }
    commands.push({
      id: 'toggle-wireframe',
      title: wireframe ? 'Disable Wireframe Overlay' : 'Enable Wireframe Overlay',
      hint: 'Toggle LOD surface wireframe',
      keywords: 'wireframe mesh surface toggle overlay',
      category: '3D Axes & Camera',
      run: () => useProjectionStore.getState().setWireframe(!useProjectionStore.getState().wireframe),
    });
    commands.push({
      id: 'toggle-pca',
      title: 'Toggle 3D PCA Mode',
      hint: 'Switch axis explorer / PCA manifold',
      keywords: 'pca manifold projection toggle 3d global',
      category: '3D Axes & Camera',
      run: () => {
        const proj = useProjectionStore.getState();
        proj.setMode(proj.mode === 'pca_3d' ? 'axis_explorer' : 'pca_3d');
      },
    });

    // Playback
    commands.push({
      id: 'playback-toggle',
      title: 'Play / Pause',
      hint: 'Space',
      keywords: 'play pause space start stop playback vcr',
      category: 'Playback',
      run: () => usePlaybackStore.getState().togglePlayPause(),
    });
    commands.push({
      id: 'playback-step-fwd',
      title: 'Step Forward 1 Frame',
      hint: 'ArrowRight',
      keywords: 'step next forward frame arrow',
      category: 'Playback',
      run: () => usePlaybackStore.getState().stepFrame(1),
    });
    commands.push({
      id: 'playback-step-back',
      title: 'Step Back 1 Frame',
      hint: 'ArrowLeft',
      keywords: 'step back previous frame arrow',
      category: 'Playback',
      run: () => usePlaybackStore.getState().stepFrame(-1),
    });
    commands.push({
      id: 'find-counterfactual',
      title: 'Find Counterfactual from Current Frame',
      hint: 'Runs What-If finder (F)',
      keywords: 'counterfactual flip whatif find minimum change ghost',
      category: 'Playback',
      run: () => void useWhatIfStore.getState().findCounterfactual(),
    });

    // Export & Bake
    commands.push({
      id: 'open-aliases',
      title: 'Edit Class & Feature Aliases…',
      hint: 'Human-readable labels + colors',
      keywords: 'alias label class feature rename color display unit',
      category: 'Export & Bake',
      run: () => useStudioLayoutStore.getState().setAliasModal(true),
    });
    commands.push({
      id: 'toggle-cinema',
      title: 'Toggle Cinema Mode',
      hint: 'Tab — fullscreen 3D',
      keywords: 'cinema fullscreen presentation tab hide panels',
      category: 'Export & Bake',
      run: () => useStudioLayoutStore.getState().toggleCinema(),
    });

    return { commands, availableFeatures: featureNames };
  }, [totalFrames, featureNames, wireframe]);
}


import { create } from 'zustand';
import { API_BASE_URL } from '../../../core/api/client';

export interface FeatureRequirement {
  index: number;
  name: string;
  expected_dtype: string;
  is_auto_generated_name: boolean;
  learned_weight_or_importance: number | null;
}

export interface ModelMetadataResponse {
  model_id: string;
  filename: string;
  sha256_hash: string;
  algorithm_key: string;
  display_name: string;
  family_name: string;
  task_type: string;
  visual_3d_type: string;
  is_pipeline: boolean;
  pipeline_steps: string[];
  n_features_in: number;
  classes: Array<string | number> | null;
  intercept_or_base: number | null;
  features: FeatureRequirement[];
  hyperparameters: Record<string, unknown>;
  warnings: string[];
}

interface ModelStoreState {
  currentModel: ModelMetadataResponse | null;
  lastModelFile: File | null;
  isLoadingModel: boolean;
  modelError: string | null;
  uploadModel: (file: File) => Promise<void>;
  loadSampleModel: (sampleKey: string) => Promise<void>;
  clearModelError: () => void;
}

async function throwIfError(res: Response): Promise<void> {
  if (res.ok) return;
  let message = `Model load failed (${res.status})`;
  try {
    const data = (await res.json()) as { error?: { code?: string; message?: string } };
    if (data?.error?.message) message = `${data.error.code ?? 'ERROR'}: ${data.error.message}`;
  } catch {
    /* keep default */
  }
  throw new Error(message);
}

export const ACCEPTED_MODEL_EXTENSIONS = '.joblib,.pkl,.pickle,.cbm,.json,.ubj,.model,.xgb,.txt,.lgb,.onnx';

export function isAcceptedModelFile(name: string): boolean {
  const lower = name.toLowerCase();
  return ['.joblib', '.pkl', '.pickle', '.cbm', '.json', '.ubj', '.model', '.xgb', '.txt', '.lgb', '.onnx'].some((ext) =>
    lower.endsWith(ext),
  );
}

export const TASK_TYPE_META: Record<string, { label: string; color: string }> = {
  regression: { label: 'Regression', color: '#06B6D4' },
  binary_classification: { label: 'Binary Classification', color: '#818CF8' },
  multiclass_classification: { label: 'Multiclass Classification', color: '#E879F9' },
  clustering: { label: 'Clustering / Unsupervised', color: '#F59E0B' },
  anomaly_detection: { label: 'Anomaly Detection', color: '#F59E0B' },
  dimensionality_reduction: { label: 'Dimensionality Reduction', color: '#94A3B8' },
};

export const VISUAL_3D_LABELS: Record<string, string> = {
  flat_hyperplane: 'Flat Tilted Hyperplane Surface',
  probability_sigmoid_surface: 'Probability Sigmoid Surface',
  stepped_axis_blocks: 'Stepped Axis Blocks',
  ensemble_smooth_terrain: 'Ensemble Smooth Terrain',
  voronoi_knn_proximity: 'Voronoi KNN Proximity',
  svm_margin_tube: 'SVM Margin Tube',
  gaussian_ellipsoid_domes: 'Gaussian Ellipsoid Domes',
  discrete_nb_towers: 'Discrete NB Probability Towers',
  discriminant_ellipsoids: 'Discriminant Covariance Ellipsoids',
  neural_morphing_manifold: 'Neural Morphing Manifold',
  cluster_centroid_voronoi: 'Cluster Centroid Voronoi',
  anomaly_isolation_depth: 'Anomaly Isolation Depth',
};

export const useModelStore = create<ModelStoreState>((set) => ({
  currentModel: null,
  lastModelFile: null,
  isLoadingModel: false,
  modelError: null,

  uploadModel: async (file) => {
    set({ isLoadingModel: true, modelError: null, lastModelFile: file });
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${API_BASE_URL}/api/v1/models/upload`, { method: 'POST', body: form });
      await throwIfError(res);
      set({ currentModel: (await res.json()) as ModelMetadataResponse });
    } catch (err) {
      set({ modelError: err instanceof Error ? err.message : String(err) });
      throw err;
    } finally {
      set({ isLoadingModel: false });
    }
  },

  loadSampleModel: async (sampleKey) => {
    set({ isLoadingModel: true, modelError: null });
    try {
      const res = await fetch(
        `${API_BASE_URL}/api/v1/models/load-sample/${encodeURIComponent(sampleKey)}`,
        { method: 'POST' },
      );
      await throwIfError(res);
      set({ currentModel: (await res.json()) as ModelMetadataResponse });
    } catch (err) {
      set({ modelError: err instanceof Error ? err.message : String(err) });
      throw err;
    } finally {
      set({ isLoadingModel: false });
    }
  },

  clearModelError: () => set({ modelError: null }),
}));

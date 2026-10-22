/** Phase 09 — typed client for POST /api/v1/projection/surface-mesh. */
import { apiFetch } from '../../../core/api/client';
import type {
  BoostFrame2DPayload,
  Contour2DPayload,
  KnnFrame2DPayload,
  LodLevel,
  PinningMode,
  ProjectionMode,
  SurfaceMeshPayload,
  ZMode,
} from '../types/projection.types';

interface SurfaceMeshResponse {
  grid_resolution: number;
  x_feature: string;
  y_feature: string;
  x_range: [number, number];
  y_range: [number, number];
  z_range: [number, number];
  z_heights_world: number[];
  raw_predictions: number[];
  point_positions_world: Array<[number, number, number]>;
  point_row_indices: number[];
  trajectory_points: SurfaceMeshPayload['trajectoryPoints'];
  pca_variance_ratio: [number, number, number] | null;
  pca_loadings: number[][] | null;
  resolved_z_mode: ZMode;
  has_constant_axis_warning: boolean;
  computation_time_ms: number;
  dataset_row_count: number;
}

export interface FetchSurfaceParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  mode: ProjectionMode;
  xFeature: string;
  yFeature: string;
  zMode: ZMode;
  targetClassIndex: number;
  pinningMode: PinningMode;
  lodLevel: LodLevel;
  currentFrameIndex: number | null;
  useRobustRange: boolean;
  selectedRowIndices: number[];
}

interface Contour2DResponse {
  grid_resolution: number;
  x_feature: string;
  y_feature: string;
  x_range_raw: [number, number];
  y_range_raw: [number, number];
  prediction_task: 'classification' | 'regression';
  classes: Array<string | number>;
  heatmap_min: number;
  heatmap_max: number;
  heatmap_quantized: number[];
  contours: Array<{
    threshold: number;
    class_index: number;
    kind: 'decision' | 'confidence' | 'elevation' | 'margin';
    line_segments: Array<[number, number, number, number]>;
  }>;
  response_curve_1d: { x_values_raw: number[]; y_pred: number[] } | null;
  linear_2d: {
    family: string;
    task: 'classification' | 'regression';
    x_feature: string;
    y_feature: string;
    w1: number;
    w2: number;
    b_eff: number;
    decision_threshold: number;
    normal_vector: [number, number];
    corridor_width: number | null;
    margin_level: number | null;
    is_x_zeroed: boolean;
    is_y_zeroed: boolean;
    both_zeroed: boolean;
    sparsity_ratio: number;
    l1_ratio: number | null;
    alpha: number | null;
    penalty: string | null;
    slope_w1: number | null;
    x_sweep_intercept: number | null;
    formula: string;
    is_multiclass: boolean;
    n_classes: number;
    lines: Array<{
      kind: 'decision' | 'margin_pos' | 'margin_neg' | 'confidence' | 'pairwise' | 'slope';
      level: number;
      segment_world: [number, number, number, number] | null;
      segment_raw: [number, number, number, number] | null;
      label: string;
      class_a: number | null;
      class_b: number | null;
    }>;
  } | null;
  svm_2d: {
    model_type: string;
    kernel: string;
    gamma: number | null;
    c: number | null;
    degree: number;
    coef0: number;
    is_regressor: boolean;
    is_binary: boolean;
    n_classes: number;
    n_support_total: number;
    is_linear_no_sv: boolean;
    support_vectors_2d: Array<{
      train_index: number;
      x_world: number;
      y_world: number;
      dual_coef: number;
      halo_scale: number;
      status: 'on_margin' | 'violates_margin' | 'support';
    }>;
    halos_truncated: boolean;
  } | null;
  knn_2d: {
    is_knn_model: boolean;
    model_type: string;
    k: number;
    weights_mode: string;
    metric: string;
    is_radius_model: boolean;
    radius: number | null;
  } | null;
  probabilistic_2d: {
    family: string;
    algorithm: string;
    x_feature: string;
    y_feature: string;
    x_feature_idx: number;
    y_feature_idx: number;
    n_classes: number;
    classes: Array<string | number>;
    ellipses: Array<{
      class_index: number;
      label: string | number | null;
      mean_raw: [number, number];
      std_raw: [number, number];
      rotation_rad: number;
      center_world: [number, number];
      radii_world_1sigma: [number, number];
      prior: number | null;
    }>;
    ellipses_truncated: boolean;
    is_covariance_approximated: boolean;
    discriminant_line: {
      segment_raw: [number, number, number, number] | null;
      segment_world: [number, number, number, number] | null;
      w1: number;
      w2: number;
      b_eff: number;
      label: string;
    } | null;
    conic: {
      kind: string;
      conic_type: string;
      coefficients_raw: { A: number; B: number; C: number; D: number; E: number; F: number };
      class_a: number;
      class_b: number;
      segments_world: Array<[number, number, number, number]>;
      segments_truncated: boolean;
      label: string;
    } | null;
    discrete: {
      variant: string;
      algorithm: string;
      alpha: number | null;
      binarize: number | null;
      axis_log_prob: number[][];
      classes: Array<string | number>;
      formula: string;
    } | null;
    formula: string;
  } | null;
  trees_2d: {
    family: string;
    algorithm: string;
    x_feature: string;
    y_feature: string;
    x_feature_idx: number;
    y_feature_idx: number;
    n_estimators: number;
    trees_exported: number;
    segments: Array<{
      axis: 'x' | 'y';
      threshold: number;
      node_id: number;
      tree_index: number;
      depth: number;
      segment_raw: [number, number, number, number];
      segment_world: [number, number, number, number];
    }>;
    segments_truncated: boolean;
    leaves: Array<{
      leaf_id: number;
      tree_index: number;
      box_raw: [number, number, number, number];
      box_world: [number, number, number, number];
      prediction: string | number | null;
      samples: number;
      impurity: number;
      rule: string;
      is_active: boolean;
    }>;
    leaves_truncated: boolean;
    stages: Array<{
      stage: number;
      feature_index: number;
      axis: 'x' | 'y' | 'off';
      threshold: number | null;
      weight: number;
      tree_depth: number;
      segment_world: [number, number, number, number] | null;
    }>;
    stages_truncated: boolean;
    base_value: number | null;
    no_axis_splits: boolean;
    multi_output: boolean;
    formula: string;
  } | null;
  boosting_2d: {
    family: string;
    algorithm: string;
    x_feature: string;
    y_feature: string;
    x_feature_idx: number;
    y_feature_idx: number;
    n_estimators: number;
    trees_exported: number;
    segments: Array<{
      axis: 'x' | 'y';
      threshold: number;
      node_id: number;
      tree_index: number;
      depth: number;
      segment_raw: [number, number, number, number];
      segment_world: [number, number, number, number];
    }>;
    segments_truncated: boolean;
    leaves: Array<{
      leaf_id: number;
      tree_index: number;
      box_raw: [number, number, number, number];
      box_world: [number, number, number, number];
      leaf_value: number;
      rule: string;
      is_active: boolean;
    }>;
    leaves_truncated: boolean;
    bin_edges_x_raw: number[];
    bin_edges_y_raw: number[];
    bin_lines_x_world: number[];
    bin_lines_y_world: number[];
    active_bin: [number, number] | null;
    oblivious_x_cuts_raw: number[];
    oblivious_y_cuts_raw: number[];
    oblivious_x_lines_world: number[];
    oblivious_y_lines_world: number[];
    oblivious_cells: Array<{
      cell_id: number;
      leaf_index: number;
      bitmask: string;
      box_raw: [number, number, number, number];
      box_world: [number, number, number, number];
      is_active: boolean;
    }>;
    oblivious_truncated: boolean;
    oblivious_rule: string;
    shap_vector: {
      phi_x: number;
      phi_y: number;
      magnitude: number;
      bias: number | null;
      method: string;
    } | null;
    no_axis_splits: boolean;
    formula: string;
  } | null;
  neural_2d: {
    family: string;
    algorithm: string;
    x_feature: string;
    y_feature: string;
    x_feature_idx: number;
    y_feature_idx: number;
    activation: string | null;
    hyperplanes: Array<{
      neuron_idx: number;
      w1: number;
      w2: number;
      b_eff: number;
      segment_raw: [number, number, number, number] | null;
      segment_world: [number, number, number, number] | null;
      is_dead: boolean;
      is_always_active: boolean;
    }>;
    hyperplanes_truncated: boolean;
    dead_count: number;
    active_count: number;
    active_ratio: number | null;
    pipeline: {
      has_toggle: boolean;
      scaler_kind: string;
      inner_family: string;
      inner_algorithm: string;
      mu_x: number | null;
      sigma_x: number | null;
      mu_y: number | null;
      sigma_y: number | null;
      scaled_x_range: [number, number] | null;
      scaled_y_range: [number, number] | null;
      formula: string;
    } | null;
    onnx: {
      operator: string;
      chain: string[];
      n_nodes: number;
      latent_available: boolean;
      latent_points_2d: Array<[number, number]>;
      latent_explained_variance: number[];
      penultimate_width: number;
      formula: string;
    } | null;
    no_axis_splits: boolean;
    formula: string;
  } | null;
  computation_time_ms: number;
}

export interface FetchContourParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  xFeature: string;
  yFeature: string;
  pinningMode: PinningMode;
  currentFrameIndex: number | null;
  useRobustRange: boolean;
  gridResolution: number;
  selectedRowIndices: number[];
}

/** Phase 41 — typed client for POST /api/v1/projection/contour-2d. */
export async function fetchContour2D(
  params: FetchContourParams,
  signal?: AbortSignal,
): Promise<Contour2DPayload> {
  const data = await apiFetch<Contour2DResponse>('/api/v1/projection/contour-2d', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      pinning_mode: params.pinningMode,
      current_frame_index: params.currentFrameIndex,
      use_robust_range: params.useRobustRange,
      grid_resolution: params.gridResolution,
      selected_row_indices: params.selectedRowIndices,
    }),
    signal,
  });
  return {
    gridResolution: data.grid_resolution,
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    xRangeRaw: data.x_range_raw,
    yRangeRaw: data.y_range_raw,
    predictionTask: data.prediction_task,
    classes: data.classes,
    heatmapMin: data.heatmap_min,
    heatmapMax: data.heatmap_max,
    heatmapQuantized: Uint8Array.from(data.heatmap_quantized),
    contours: data.contours.map((c) => ({
      threshold: c.threshold,
      classIndex: c.class_index,
      kind: c.kind,
      lineSegments: c.line_segments,
    })),
    responseCurve1D: data.response_curve_1d
      ? { xValuesRaw: data.response_curve_1d.x_values_raw, yPred: data.response_curve_1d.y_pred }
      : null,
    linear2d: data.linear_2d
      ? {
          family: data.linear_2d.family,
          task: data.linear_2d.task,
          xFeature: data.linear_2d.x_feature,
          yFeature: data.linear_2d.y_feature,
          w1: data.linear_2d.w1,
          w2: data.linear_2d.w2,
          bEff: data.linear_2d.b_eff,
          decisionThreshold: data.linear_2d.decision_threshold,
          normalVector: data.linear_2d.normal_vector,
          corridorWidth: data.linear_2d.corridor_width,
          marginLevel: data.linear_2d.margin_level,
          isXZeroed: data.linear_2d.is_x_zeroed,
          isYZeroed: data.linear_2d.is_y_zeroed,
          bothZeroed: data.linear_2d.both_zeroed,
          sparsityRatio: data.linear_2d.sparsity_ratio,
          l1Ratio: data.linear_2d.l1_ratio,
          alpha: data.linear_2d.alpha,
          penalty: data.linear_2d.penalty,
          slopeW1: data.linear_2d.slope_w1,
          xSweepIntercept: data.linear_2d.x_sweep_intercept,
          formula: data.linear_2d.formula,
          isMulticlass: data.linear_2d.is_multiclass,
          nClasses: data.linear_2d.n_classes,
          lines: data.linear_2d.lines.map((l) => ({
            kind: l.kind,
            level: l.level,
            segmentWorld: l.segment_world,
            segmentRaw: l.segment_raw,
            label: l.label,
            classA: l.class_a,
            classB: l.class_b,
          })),
        }
      : null,
    svm2d: data.svm_2d
      ? {
          modelType: data.svm_2d.model_type,
          kernel: data.svm_2d.kernel,
          gamma: data.svm_2d.gamma,
          c: data.svm_2d.c,
          degree: data.svm_2d.degree,
          coef0: data.svm_2d.coef0,
          isRegressor: data.svm_2d.is_regressor,
          isBinary: data.svm_2d.is_binary,
          nClasses: data.svm_2d.n_classes,
          nSupportTotal: data.svm_2d.n_support_total,
          isLinearNoSv: data.svm_2d.is_linear_no_sv,
          halos: data.svm_2d.support_vectors_2d.map((h) => ({
            trainIndex: h.train_index,
            xWorld: h.x_world,
            yWorld: h.y_world,
            dualCoef: h.dual_coef,
            haloScale: h.halo_scale,
            status: h.status,
          })),
          halosTruncated: data.svm_2d.halos_truncated,
        }
      : null,
    knn2d: data.knn_2d
      ? {
          isKnnModel: data.knn_2d.is_knn_model,
          modelType: data.knn_2d.model_type,
          k: data.knn_2d.k,
          weightsMode: data.knn_2d.weights_mode,
          metric: data.knn_2d.metric,
          isRadiusModel: data.knn_2d.is_radius_model,
          radius: data.knn_2d.radius,
        }
      : null,
    probabilistic2d: data.probabilistic_2d
      ? {
          family: data.probabilistic_2d.family,
          algorithm: data.probabilistic_2d.algorithm,
          xFeature: data.probabilistic_2d.x_feature,
          yFeature: data.probabilistic_2d.y_feature,
          xFeatureIdx: data.probabilistic_2d.x_feature_idx,
          yFeatureIdx: data.probabilistic_2d.y_feature_idx,
          nClasses: data.probabilistic_2d.n_classes,
          classes: data.probabilistic_2d.classes,
          ellipses: data.probabilistic_2d.ellipses.map((e) => ({
            classIndex: e.class_index,
            label: e.label,
            meanRaw: e.mean_raw,
            stdRaw: e.std_raw,
            rotationRad: e.rotation_rad,
            centerWorld: e.center_world,
            radiiWorld1Sigma: e.radii_world_1sigma,
            prior: e.prior,
          })),
          ellipsesTruncated: data.probabilistic_2d.ellipses_truncated,
          isCovarianceApproximated: data.probabilistic_2d.is_covariance_approximated,
          discriminantLine: data.probabilistic_2d.discriminant_line
            ? {
                segmentRaw: data.probabilistic_2d.discriminant_line.segment_raw,
                segmentWorld: data.probabilistic_2d.discriminant_line.segment_world,
                w1: data.probabilistic_2d.discriminant_line.w1,
                w2: data.probabilistic_2d.discriminant_line.w2,
                bEff: data.probabilistic_2d.discriminant_line.b_eff,
                label: data.probabilistic_2d.discriminant_line.label,
              }
            : null,
          conic: data.probabilistic_2d.conic
            ? {
                kind: data.probabilistic_2d.conic.kind,
                conicType: data.probabilistic_2d.conic.conic_type,
                coefficientsRaw: data.probabilistic_2d.conic.coefficients_raw,
                classA: data.probabilistic_2d.conic.class_a,
                classB: data.probabilistic_2d.conic.class_b,
                segmentsWorld: data.probabilistic_2d.conic.segments_world,
                segmentsTruncated: data.probabilistic_2d.conic.segments_truncated,
                label: data.probabilistic_2d.conic.label,
              }
            : null,
          discrete: data.probabilistic_2d.discrete
            ? {
                variant: data.probabilistic_2d.discrete.variant,
                algorithm: data.probabilistic_2d.discrete.algorithm,
                alpha: data.probabilistic_2d.discrete.alpha,
                binarize: data.probabilistic_2d.discrete.binarize,
                axisLogProb: data.probabilistic_2d.discrete.axis_log_prob,
                classes: data.probabilistic_2d.discrete.classes,
                formula: data.probabilistic_2d.discrete.formula,
              }
            : null,
          formula: data.probabilistic_2d.formula,
        }
      : null,
    trees2d: data.trees_2d
      ? {
          family: data.trees_2d.family,
          algorithm: data.trees_2d.algorithm,
          xFeature: data.trees_2d.x_feature,
          yFeature: data.trees_2d.y_feature,
          xFeatureIdx: data.trees_2d.x_feature_idx,
          yFeatureIdx: data.trees_2d.y_feature_idx,
          nEstimators: data.trees_2d.n_estimators,
          treesExported: data.trees_2d.trees_exported,
          segments: data.trees_2d.segments.map((s) => ({
            axis: s.axis,
            threshold: s.threshold,
            nodeId: s.node_id,
            treeIndex: s.tree_index,
            depth: s.depth,
            segmentRaw: s.segment_raw,
            segmentWorld: s.segment_world,
          })),
          segmentsTruncated: data.trees_2d.segments_truncated,
          leaves: data.trees_2d.leaves.map((l) => ({
            leafId: l.leaf_id,
            treeIndex: l.tree_index,
            boxRaw: l.box_raw,
            boxWorld: l.box_world,
            prediction: l.prediction,
            samples: l.samples,
            impurity: l.impurity,
            rule: l.rule,
            isActive: l.is_active,
          })),
          leavesTruncated: data.trees_2d.leaves_truncated,
          stages: data.trees_2d.stages.map((st) => ({
            stage: st.stage,
            featureIndex: st.feature_index,
            axis: st.axis,
            threshold: st.threshold,
            weight: st.weight,
            treeDepth: st.tree_depth,
            segmentWorld: st.segment_world,
          })),
          stagesTruncated: data.trees_2d.stages_truncated,
          baseValue: data.trees_2d.base_value,
          noAxisSplits: data.trees_2d.no_axis_splits,
          multiOutput: data.trees_2d.multi_output,
          formula: data.trees_2d.formula,
        }
      : null,
    boosting2d: data.boosting_2d
      ? {
          family: data.boosting_2d.family,
          algorithm: data.boosting_2d.algorithm,
          xFeature: data.boosting_2d.x_feature,
          yFeature: data.boosting_2d.y_feature,
          xFeatureIdx: data.boosting_2d.x_feature_idx,
          yFeatureIdx: data.boosting_2d.y_feature_idx,
          nEstimators: data.boosting_2d.n_estimators,
          treesExported: data.boosting_2d.trees_exported,
          segments: data.boosting_2d.segments.map((s) => ({
            axis: s.axis,
            threshold: s.threshold,
            nodeId: s.node_id,
            treeIndex: s.tree_index,
            depth: s.depth,
            segmentRaw: s.segment_raw,
            segmentWorld: s.segment_world,
          })),
          segmentsTruncated: data.boosting_2d.segments_truncated,
          leaves: data.boosting_2d.leaves.map((l) => ({
            leafId: l.leaf_id,
            treeIndex: l.tree_index,
            boxRaw: l.box_raw,
            boxWorld: l.box_world,
            leafValue: l.leaf_value,
            rule: l.rule,
            isActive: l.is_active,
          })),
          leavesTruncated: data.boosting_2d.leaves_truncated,
          binEdgesXRaw: data.boosting_2d.bin_edges_x_raw,
          binEdgesYRaw: data.boosting_2d.bin_edges_y_raw,
          binLinesXWorld: data.boosting_2d.bin_lines_x_world,
          binLinesYWorld: data.boosting_2d.bin_lines_y_world,
          activeBin: data.boosting_2d.active_bin,
          obliviousXCutsRaw: data.boosting_2d.oblivious_x_cuts_raw,
          obliviousYCutsRaw: data.boosting_2d.oblivious_y_cuts_raw,
          obliviousXLinesWorld: data.boosting_2d.oblivious_x_lines_world,
          obliviousYLinesWorld: data.boosting_2d.oblivious_y_lines_world,
          obliviousCells: data.boosting_2d.oblivious_cells.map((c) => ({
            cellId: c.cell_id,
            leafIndex: c.leaf_index,
            bitmask: c.bitmask,
            boxRaw: c.box_raw,
            boxWorld: c.box_world,
            isActive: c.is_active,
          })),
          obliviousTruncated: data.boosting_2d.oblivious_truncated,
          obliviousRule: data.boosting_2d.oblivious_rule,
          shapVector: data.boosting_2d.shap_vector
            ? {
                phiX: data.boosting_2d.shap_vector.phi_x,
                phiY: data.boosting_2d.shap_vector.phi_y,
                magnitude: data.boosting_2d.shap_vector.magnitude,
                bias: data.boosting_2d.shap_vector.bias,
                method: data.boosting_2d.shap_vector.method,
              }
            : null,
          noAxisSplits: data.boosting_2d.no_axis_splits,
          formula: data.boosting_2d.formula,
        }
      : null,
    neural2d: data.neural_2d
      ? {
          family: data.neural_2d.family,
          algorithm: data.neural_2d.algorithm,
          xFeature: data.neural_2d.x_feature,
          yFeature: data.neural_2d.y_feature,
          xFeatureIdx: data.neural_2d.x_feature_idx,
          yFeatureIdx: data.neural_2d.y_feature_idx,
          activation: data.neural_2d.activation,
          hyperplanes: data.neural_2d.hyperplanes.map((h) => ({
            neuronIdx: h.neuron_idx,
            w1: h.w1,
            w2: h.w2,
            bEff: h.b_eff,
            segmentRaw: h.segment_raw,
            segmentWorld: h.segment_world,
            isDead: h.is_dead,
            isAlwaysActive: h.is_always_active,
          })),
          hyperplanesTruncated: data.neural_2d.hyperplanes_truncated,
          deadCount: data.neural_2d.dead_count,
          activeCount: data.neural_2d.active_count,
          activeRatio: data.neural_2d.active_ratio,
          pipeline: data.neural_2d.pipeline
            ? {
                hasToggle: data.neural_2d.pipeline.has_toggle,
                scalerKind: data.neural_2d.pipeline.scaler_kind,
                innerFamily: data.neural_2d.pipeline.inner_family,
                innerAlgorithm: data.neural_2d.pipeline.inner_algorithm,
                muX: data.neural_2d.pipeline.mu_x,
                sigmaX: data.neural_2d.pipeline.sigma_x,
                muY: data.neural_2d.pipeline.mu_y,
                sigmaY: data.neural_2d.pipeline.sigma_y,
                scaledXRange: data.neural_2d.pipeline.scaled_x_range,
                scaledYRange: data.neural_2d.pipeline.scaled_y_range,
                formula: data.neural_2d.pipeline.formula,
              }
            : null,
          onnx: data.neural_2d.onnx
            ? {
                operator: data.neural_2d.onnx.operator,
                chain: data.neural_2d.onnx.chain,
                nNodes: data.neural_2d.onnx.n_nodes,
                latentAvailable: data.neural_2d.onnx.latent_available,
                latentPoints2d: data.neural_2d.onnx.latent_points_2d,
                latentExplainedVariance: data.neural_2d.onnx.latent_explained_variance,
                penultimateWidth: data.neural_2d.onnx.penultimate_width,
                formula: data.neural_2d.onnx.formula,
              }
            : null,
          noAxisSplits: data.neural_2d.no_axis_splits,
          formula: data.neural_2d.formula,
        }
      : null,
    computationTimeMs: data.computation_time_ms,
  };
}

export interface FetchKnnFrameParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  xFeature: string;
  yFeature: string;
  rowIndex: number;
}

interface KnnFrame2DResponse {
  x_feature: string;
  y_feature: string;
  row_index: number;
  query_world: [number, number];
  radar_radius_raw: number;
  radar_radius_world: [number, number];
  neighbors: Array<{
    train_index: number;
    x_world: number;
    y_world: number;
    distance_full: number;
    distance_2d: number;
    vote_weight: number;
    rank: number;
    label: string | number | null;
  }>;
  votes: Record<string, number>;
  predicted: string | number | null;
  k: number;
  weights_mode: string;
  metric: string;
  is_radius_model: boolean;
  outlier: boolean;
  display_capped: boolean;
  total_neighbors: number;
  computation_time_ms: number;
}

/** Phase 43 — typed client for POST /api/v1/projection/knn-frame-2d. */
export async function fetchKnnFrame2D(
  params: FetchKnnFrameParams,
  signal?: AbortSignal,
): Promise<KnnFrame2DPayload> {
  const data = await apiFetch<KnnFrame2DResponse>('/api/v1/projection/knn-frame-2d', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      row_index: params.rowIndex,
    }),
    signal,
  });
  return {
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    rowIndex: data.row_index,
    queryWorld: data.query_world,
    radarRadiusRaw: data.radar_radius_raw,
    radarRadiusWorld: data.radar_radius_world,
    neighbors: data.neighbors.map((n) => ({
      trainIndex: n.train_index,
      xWorld: n.x_world,
      yWorld: n.y_world,
      distanceFull: n.distance_full,
      distance2d: n.distance_2d,
      voteWeight: n.vote_weight,
      rank: n.rank,
      label: n.label,
    })),
    votes: data.votes,
    predicted: data.predicted,
    k: data.k,
    weightsMode: data.weights_mode,
    metric: data.metric,
    isRadiusModel: data.is_radius_model,
    outlier: data.outlier,
    displayCapped: data.display_capped,
    totalNeighbors: data.total_neighbors,
    computationTimeMs: data.computation_time_ms,
  };
}

/** Phase 46 — typed client for POST /api/v1/projection/boost-frame-2d. */
export interface FetchBoostFrameParams {
  datasetId: string;
  modelId: string;
  mappingId: string;
  xFeature: string;
  yFeature: string;
  rowIndex: number;
}

interface BoostFrame2DResponse {
  x_feature: string;
  y_feature: string;
  row_index: number;
  query_world: [number, number];
  family: string;
  algorithm: string;
  shap: {
    phi_x: number;
    phi_y: number;
    magnitude: number;
    bias: number | null;
    method: string;
  } | null;
  computation_time_ms: number;
}

export async function fetchBoostFrame2D(
  params: FetchBoostFrameParams,
  signal?: AbortSignal,
): Promise<BoostFrame2DPayload> {
  const data = await apiFetch<BoostFrame2DResponse>('/api/v1/projection/boost-frame-2d', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      row_index: params.rowIndex,
    }),
    signal,
  });
  return {
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    rowIndex: data.row_index,
    queryWorld: data.query_world,
    family: data.family,
    algorithm: data.algorithm,
    shap: data.shap
      ? {
          phiX: data.shap.phi_x,
          phiY: data.shap.phi_y,
          magnitude: data.shap.magnitude,
          bias: data.shap.bias,
          method: data.shap.method,
        }
      : null,
    computationTimeMs: data.computation_time_ms,
  };
}

export async function fetchSurfaceMesh(
  params: FetchSurfaceParams,
  signal?: AbortSignal,
): Promise<SurfaceMeshPayload> {
  const data = await apiFetch<SurfaceMeshResponse>('/api/v1/projection/surface-mesh', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: params.datasetId,
      model_id: params.modelId,
      mapping_id: params.mappingId,
      mode: params.mode,
      x_feature: params.xFeature,
      y_feature: params.yFeature,
      z_mode: params.zMode,
      target_class_index: params.targetClassIndex,
      pinning_mode: params.pinningMode,
      lod_level: params.lodLevel,
      current_frame_index: params.currentFrameIndex,
      use_robust_range: params.useRobustRange,
      selected_row_indices: params.selectedRowIndices,
    }),
    signal,
  });
  return {
    gridResolution: data.grid_resolution,
    xFeature: data.x_feature,
    yFeature: data.y_feature,
    xRange: data.x_range,
    yRange: data.y_range,
    zRange: data.z_range,
    zHeightsWorld: new Float32Array(data.z_heights_world),
    rawPredictions: new Float32Array(data.raw_predictions),
    pointPositionsWorld: data.point_positions_world,
    pointRowIndices: data.point_row_indices,
    trajectoryPoints: data.trajectory_points,
    pcaVarianceRatio: data.pca_variance_ratio,
    pcaLoadings: data.pca_loadings,
    resolvedZMode: data.resolved_z_mode,
    hasConstantAxisWarning: data.has_constant_axis_warning,
    computationTimeMs: data.computation_time_ms,
    datasetRowCount: data.dataset_row_count,
  };
}

export { LinearBoundary2D } from './LinearBoundary2D';
export { SgdMarginCorridor2D } from './SgdMarginCorridor2D';
export { LassoSparsityBadge2D } from './LassoSparsityBadge2D';
export { LinearRegression2D } from './LinearRegression2D';
export { LinearOverlays2D } from './LinearOverlays2D';
export { KnnRadar2D } from './KnnRadar2D';
export { SvmKernelContour2D } from './SvmKernelContour2D';
export { SupportVectorRings2D } from './SupportVectorRings2D';
export { DistanceOverlays2D } from './DistanceOverlays2D';
export { GaussianEllipses2D } from './GaussianEllipses2D';
export { DiscriminantBoundary2D } from './DiscriminantBoundary2D';
export { DiscreteNbSteps2D } from './DiscreteNbSteps2D';
export { ProbabilisticOverlays2D } from './ProbabilisticOverlays2D';
export { TreePartitions2D } from './TreePartitions2D';
export { EnsembleConsensus2D } from './EnsembleConsensus2D';
export { AdaBoostCascade2D } from './AdaBoostCascade2D';
export { TreesOverlays2D } from './TreesOverlays2D';
export { XgboostTreeShapVector2D } from './XgboostTreeShapVector2D';
export { HistGbmQuantizedBins2D } from './HistGbmQuantizedBins2D';
export { CatBoostObliviousGrid2D } from './CatBoostObliviousGrid2D';
export { LightgbmLeafWise2D } from './LightgbmLeafWise2D';
export { BoostingOverlays2D, resolveBoostActiveLeaf } from './BoostingOverlays2D';
export { batchCutPaths, boxToRect, findLeafAt } from './trees2dMath';
export {
  scaleShapVector,
  boostBoxToRect,
  findBoostLeafAt,
  findStrongestLeaf,
  binCellRect,
  cellToRect,
  SHAP_ARROW_MAX_LEN,
  SHAP_ARROW_MIN_LEN,
  SHAP_ARROW_SCALE,
} from './boosting2dMath';
export { CometTrail2D } from './CometTrail2D';
export { DualRingPoint2D } from './DualRingPoint2D';
export { ResidualSpoke2D } from './ResidualSpoke2D';
export { HexbinOverlay2D } from './HexbinOverlay2D';
export type { HexbinMode } from './HexbinOverlay2D';
export { PointIntelligenceLayer2D } from './PointIntelligenceLayer2D';
export { PointDisplayToolbar2D } from './PointDisplayToolbar2D';
export {
  GHOST_ALPHA,
  GHOST_RADIUS,
  RING_INNER_RADIUS,
  RING_OUTER_RADIUS,
  RING_THICKNESS,
  RING_CONTRAST_WIDTH,
  cometSigma,
  cometAlpha,
  cometTrailAlphas,
  cometStrokeWidth,
  contrastStroke,
  errorsOnlyFrames,
  residualSpokeEnd,
  hexVertices,
  hexPath,
  hexbinFill,
} from './pointIntelligence2dMath';
export type { GlyphShape } from './pointIntelligence2dMath';
export { LinearOrthogonalSpoke2D } from './LinearOrthogonalSpoke2D';
export { LinearMarginDualRing2D } from './LinearMarginDualRing2D';
export type { LinearMarginRing } from './LinearMarginDualRing2D';
export { LinearResidualStem2D } from './LinearResidualStem2D';
export type { LinearResidualStem } from './LinearResidualStem2D';
export { LinearHexbinAlignment2D } from './LinearHexbinAlignment2D';
export { LinearPointIntelOverlays2D } from './LinearPointIntelOverlays2D';
export {
  worldLine,
  zScoreRaw,
  expit,
  trailSigns,
  trailSignColor,
  isMarginViolator,
  logisticRingThickness,
  orthogonalSpoke,
  worldDistToLine,
  hexbinBoundaryFlags,
  errorDistanceCorrelation,
  segToSvg,
  TRAIL_UP_COLOR,
  TRAIL_DOWN_COLOR,
  TRAIL_FLAT_COLOR,
  MARGIN_RING_COLOR,
} from './linearPointIntel2dMath';
export type { WorldLine, OrthogonalSpoke } from './linearPointIntel2dMath';
export { SvmSupportVectorHalo2D } from './SvmSupportVectorHalo2D';
export { SvrSlackSpoke2D } from './SvrSlackSpoke2D';
export type { SvrSlackSpokeDatum } from './SvrSlackSpoke2D';
export { KnnVoteArcRing2D } from './KnnVoteArcRing2D';
export { KnnDensityHexbin2D } from './KnnDensityHexbin2D';
export { DistancePointIntelOverlays2D } from './DistancePointIntelOverlays2D';
export {
  SV_HALO_ON_MARGIN_COLOR,
  SV_HALO_VIOLATES_COLOR,
  SV_HALO_SUPPORT_COLOR,
  SVR_SLACK_XI_COLOR,
  SVR_SLACK_XI_STAR_COLOR,
  KNN_DISCORDANCE_COLOR,
  knnCentroid2D,
  detectNeighborSwaps,
  haloColor,
  haloDiamondRadius,
  diamondPath,
  voteArcSegments,
  arcDonutPath,
  svrSlackSpoke,
  knnDensity,
  discordantRanks,
  densityTiers,
  spokeToSvg,
} from './distancePointIntel2dMath';
export type { KnnNeighborPoint, VoteArcSegment, SvrSlackSpoke } from './distancePointIntel2dMath';
export { MlpNeuronHyperplanes2D } from './MlpNeuronHyperplanes2D';
export { PipelineSpaceToggle2D } from './PipelineSpaceToggle2D';
export { OnnxLatentManifold2D } from './OnnxLatentManifold2D';
export { NeuralOverlays2D } from './NeuralOverlays2D';
export { batchHyperplanePaths, fitLatentToWorld, scaledRulerLabel } from './neural2dMath';
export { ellipseColor, sigmaCoverage, SIGMA_LEVELS } from './probabilistic2dMath';
export { clipLineToRectRaw, deriveExactElevationLines, rawToWorld1D } from './linear2dMath';
export type { WorldSeg } from './linear2dMath';

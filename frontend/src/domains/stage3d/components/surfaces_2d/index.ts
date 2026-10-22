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
export { ellipseColor, sigmaCoverage, SIGMA_LEVELS } from './probabilistic2dMath';
export { clipLineToRectRaw, deriveExactElevationLines, rawToWorld1D } from './linear2dMath';
export type { WorldSeg } from './linear2dMath';

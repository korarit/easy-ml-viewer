"""Mapping application use-cases — auto-suggest, dry-run, commit (Phase 05)."""

from __future__ import annotations

import uuid

import pandas as pd
import pyarrow.parquet as pq

from app.application.dataset_use_case import DatasetIngestUseCase  # noqa: F401 (re-export convenience)
from app.application.model_use_case import ModelIngestUseCase
from app.core.exceptions import DomainException
from app.domains.dataset.services import profile_dataframe
from app.domains.dataset.value_objects import ColumnProfile
from app.domains.mapping.casting_service import TypeCastingEngine
from app.domains.mapping.matcher_service import FuzzyColumnMatcherService
from app.domains.mapping.value_objects import (
    DryRunValidationReport,
    FeatureMappingItem,
    ImputationStrategy,
    MappingConfiguration,
)
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.domains.model_engine.value_objects import FeatureRequirement
from app.infrastructure.persistence.in_memory_mapping_repository import (
    InMemoryMappingRepository,
)
from app.infrastructure.storage.parquet_dataset_repository import (
    ParquetDatasetRepository,
)

_DRY_RUN_ROWS = 100


def _resolve_fill(col: ColumnProfile | None, strategy: ImputationStrategy, custom: float) -> float | None:
    if strategy == ImputationStrategy.PASS_NAN:
        return None
    if strategy == ImputationStrategy.ZERO:
        return 0.0
    if strategy == ImputationStrategy.CUSTOM:
        return float(custom)
    if col is None:
        return 0.0
    if strategy == ImputationStrategy.MEAN and col.mean_value is not None:
        return float(col.mean_value)
    if col.median_value is not None:
        return float(col.median_value)
    if col.mean_value is not None:
        return float(col.mean_value)
    return 0.0


class MappingUseCase:
    """Orchestrates dataset parquet + model registry + matcher + dry-run."""

    def __init__(
        self,
        dataset_repo: ParquetDatasetRepository | None = None,
        model_use_case: ModelIngestUseCase | None = None,
        mapping_repo: InMemoryMappingRepository | None = None,
        matcher: FuzzyColumnMatcherService | None = None,
    ) -> None:
        self._datasets = dataset_repo or ParquetDatasetRepository()
        self._models = model_use_case or ModelIngestUseCase()
        self._mappings = mapping_repo or InMemoryMappingRepository()
        self._matcher = matcher or FuzzyColumnMatcherService()

    # -- internals -----------------------------------------------------
    def _model_features(self, model_id: str) -> tuple[list[FeatureRequirement], list[str | int | float] | None]:
        model_obj = self._models.get_model(model_id)
        _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
        schema = UniversalSchemaDetector.extract_schema(final_est)
        return schema.features, schema.classes

    def _dataset_columns(self, dataset_id: str) -> tuple[list[ColumnProfile], pd.DataFrame | None]:
        path = self._datasets._require(dataset_id)
        table = pq.read_table(path, memory_map=True)
        df = table.to_pandas()
        columns, _ = profile_dataframe(df)
        return columns, None

    def _dataset_head(self, dataset_id: str, columns: list[str] | None, limit: int = _DRY_RUN_ROWS) -> pd.DataFrame:
        # Column-selective projection: only mapped cols (+ target) hit RAM (§5.1).
        # Dedupe for the parquet read (duplicate mapping, edge case #6, is still
        # reported by dry_run_validate via the original item list).
        deduped = list(dict.fromkeys(columns)) if columns else None
        return self._datasets.read_columns_mmap(dataset_id, columns=deduped, offset=0, limit=limit)

    def _suggest_target_label_map(
        self,
        dataset_id: str,
        target_column: str | None,
        classes: list[str | int | float] | None,
    ) -> dict[str, int | float] | None:
        if not target_column or not classes:
            return None
        try:
            series = self._datasets.read_columns_mmap(
                dataset_id, columns=[target_column], offset=0, limit=1000
            )[target_column]
        except Exception:
            return None
        uniques = [v for v in pd.unique(series.dropna())][:32]
        if not uniques:
            return None
        # If the target column is already numeric and classes are numeric, no aliasing needed.
        try:
            if pd.api.types.is_numeric_dtype(series.dtype) and all(
                isinstance(c, (int, float)) for c in classes
            ):
                return None
        except Exception:
            pass
        label_map: dict[str, int | float] = {}
        import re as _re

        def _norm(v: object) -> str:
            return _re.sub(r"[^a-z0-9]", "", str(v).lower().strip())

        used: set[int] = set()
        for raw in uniques:
            key = str(raw)
            best_idx: int | None = None
            best_score = -1.0
            for i, cls in enumerate(classes):
                if i in used:
                    continue
                score = 1.0 if _norm(key) == _norm(cls) else 0.0
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None or best_score <= 0.0:
                # Fallback: first unused class in order.
                for i in range(len(classes)):
                    if i not in used:
                        best_idx = i
                        break
            if best_idx is not None:
                used.add(best_idx)
                label_map[key] = classes[best_idx]
        return label_map or None

    def _apply_fill_resolution(
        self, items: list[FeatureMappingItem], columns: list[ColumnProfile]
    ) -> None:
        by_name = {c.name: c for c in columns}
        for item in items:
            col = by_name.get(item.dataset_column_name) if item.dataset_column_name else None
            if item.imputation_strategy == ImputationStrategy.PASS_NAN:
                item.resolved_fill_value = None
            else:
                item.resolved_fill_value = _resolve_fill(col, item.imputation_strategy, item.custom_fill_value)

    # -- public API ----------------------------------------------------
    def auto_suggest(self, dataset_id: str, model_id: str) -> tuple[MappingConfiguration, DryRunValidationReport]:
        features, classes = self._model_features(model_id)
        columns, _ = self._dataset_columns(dataset_id)
        items = self._matcher.auto_match(features, columns)
        self._apply_fill_resolution(items, columns)

        required = [i.dataset_column_name for i in items if i.dataset_column_name]
        head = self._dataset_head(dataset_id, required or None)
        report = TypeCastingEngine.dry_run_validate(head, items)
        config = MappingConfiguration(
            mapping_id=f"map_{uuid.uuid4().hex[:8]}",
            dataset_id=dataset_id,
            model_id=model_id,
            items=items,
            target_column_name=None,
            target_label_map=None,
            is_valid=report.is_valid,
            duplicate_columns=list(report.duplicate_columns_used),
        )
        _ = classes  # reserved for target label-map suggestion on explicit target pick
        return config, report

    def map_by_order(self, dataset_id: str, model_id: str) -> tuple[MappingConfiguration, DryRunValidationReport]:
        features, _ = self._model_features(model_id)
        columns, _ = self._dataset_columns(dataset_id)
        items = self._matcher.map_by_order(features, columns)
        self._apply_fill_resolution(items, columns)
        required = [i.dataset_column_name for i in items if i.dataset_column_name]
        head = self._dataset_head(dataset_id, required or None)
        report = TypeCastingEngine.dry_run_validate(head, items)
        config = MappingConfiguration(
            mapping_id=f"map_{uuid.uuid4().hex[:8]}",
            dataset_id=dataset_id,
            model_id=model_id,
            items=items,
            is_valid=report.is_valid,
            duplicate_columns=list(report.duplicate_columns_used),
        )
        return config, report

    def validate_preview(
        self,
        dataset_id: str,
        model_id: str,
        items: list[FeatureMappingItem],
        target_column_name: str | None = None,
    ) -> DryRunValidationReport:
        columns, _ = self._dataset_columns(dataset_id)
        self._apply_fill_resolution(items, columns)
        required = [i.dataset_column_name for i in items if i.dataset_column_name]
        if target_column_name and target_column_name not in required:
            required.append(target_column_name)
        head = self._dataset_head(dataset_id, required or None)
        return TypeCastingEngine.dry_run_validate(head, items, target_column_name)

    def validate_and_save(
        self,
        dataset_id: str,
        model_id: str,
        items: list[FeatureMappingItem],
        target_column_name: str | None = None,
        target_label_map: dict[str, int | float] | None = None,
    ) -> tuple[MappingConfiguration, DryRunValidationReport]:
        # Guard: unknown target column fails fast with 404-style domain error.
        if target_column_name:
            columns, _ = self._dataset_columns(dataset_id)
            if target_column_name not in {c.name for c in columns}:
                raise DomainException(
                    code="TARGET_COLUMN_NOT_FOUND",
                    message=f"Target column '{target_column_name}' not in dataset",
                    http_status=422,
                )
        report = self.validate_preview(dataset_id, model_id, items, target_column_name)
        _, classes = self._model_features(model_id)
        suggested_map = self._suggest_target_label_map(dataset_id, target_column_name, classes)
        config = MappingConfiguration(
            mapping_id=f"map_{uuid.uuid4().hex[:8]}",
            dataset_id=dataset_id,
            model_id=model_id,
            items=items,
            target_column_name=target_column_name,
            target_label_map=target_label_map or suggested_map,
            is_valid=report.is_valid,
            duplicate_columns=list(report.duplicate_columns_used),
        )
        self._mappings.save(config)
        return config, report

    def get(self, mapping_id: str) -> MappingConfiguration:
        return self._mappings.get(mapping_id)

    def supports_nan(self, model_id: str) -> bool:
        try:
            model_obj = self._models.get_model(model_id)
            _, final_est, _ = UniversalSchemaDetector.unwrap_model_container(model_obj)
            adapter = ModelRegistry.resolve_adapter(final_est, ".joblib")
            return bool(getattr(adapter, "supports_nan_input", False))
        except Exception:
            return False

"""Model application use-cases — upload introspection + sample loading."""

from __future__ import annotations

import gc
import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import ModelLoadError
from app.core.memory_guard import MemoryGuard
from app.domains.model_engine.entities import ModelMetadata
from app.domains.model_engine.registry import ModelRegistry
from app.domains.model_engine.schema_detector import UniversalSchemaDetector
from app.infrastructure.ml_loaders.joblib_model_repository import JoblibModelRepository

# Ensure built-in Phase 03 adapters self-register on import.
from app.domains.model_engine.adapters import linear_regression as _linear_adapter  # noqa: F401

_ALLOWED_MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle"}
_ALLOWED_SAMPLE_SUFFIXES = {".joblib", ".pkl", ".pickle"}


def _sanitize_hyperparameters(raw: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                clean[str(key)] = None
            else:
                clean[str(key)] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (bool, int, float, str)) or v is None for v in value
        ):
            clean[str(key)] = list(value)
        else:
            clean[str(key)] = str(value)
    return clean


class ModelIngestUseCase:
    def __init__(self, repository: JoblibModelRepository | None = None) -> None:
        self._repository = repository or JoblibModelRepository()

    # -- public API ----------------------------------------------------
    def ingest_upload_contents(self, contents: bytes, filename: str) -> ModelMetadata:
        max_bytes = int(settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        if len(contents) > max_bytes:
            raise ModelLoadError(
                f"Model file exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit",
                details={"size_bytes": len(contents)},
                code="UPLOAD_TOO_LARGE",
                http_status=413,
            )
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_MODEL_SUFFIXES:
            raise ModelLoadError(
                f"Unsupported model format '{suffix}' (allowed: .joblib, .pkl, .pickle)",
                details={"suffix": suffix},
                code="UNSUPPORTED_MODEL_FORMAT",
            )
        digest = hashlib.sha256(contents).hexdigest()
        with tempfile.NamedTemporaryFile(suffix=suffix or ".joblib", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        try:
            model_id = f"mdl_{uuid.uuid4().hex[:8]}"
            obj, load_warnings = self._repository.load_and_cache(model_id, tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return self._build_metadata(model_id, filename, digest, obj, load_warnings, suffix)

    def load_sample(self, sample_key: str) -> ModelMetadata:
        safe = Path(sample_key).name
        candidate: Path | None = None
        for suffix in _ALLOWED_SAMPLE_SUFFIXES:
            probe = settings.SAMPLE_DIR / f"{safe}{suffix}"
            if probe.exists():
                candidate = probe
                break
            probe2 = settings.SAMPLE_DIR / safe
            if probe2.exists() and probe2.suffix.lower() in _ALLOWED_SAMPLE_SUFFIXES:
                candidate = probe2
                break
        if candidate is None:
            available = self.available_samples()
            raise ModelLoadError(
                f"Sample model '{sample_key}' not found",
                details={"available_samples": available},
                code="SAMPLE_NOT_FOUND",
                http_status=404,
            )
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        model_id = f"mdl_{uuid.uuid4().hex[:8]}"
        obj, load_warnings = self._repository.load_and_cache(model_id, candidate)
        return self._build_metadata(
            model_id, candidate.name, digest, obj, load_warnings, candidate.suffix.lower()
        )

    def available_samples(self) -> list[str]:
        if not settings.SAMPLE_DIR.exists():
            return []
        return sorted(
            p.stem
            for p in settings.SAMPLE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in _ALLOWED_SAMPLE_SUFFIXES
        )

    # -- internals -----------------------------------------------------
    def _build_metadata(
        self,
        model_id: str,
        filename: str,
        digest: str,
        obj: Any,
        load_warnings: list[str],
        file_ext: str,
    ) -> ModelMetadata:
        if not hasattr(obj, "predict") and not hasattr(obj, "transform"):
            raise ModelLoadError(
                f"File '{filename}' is tabular data, not a model — "
                "upload datasets in Step 1 (Dataset), not here",
                details={"object_type": type(obj).__name__},
                code="NOT_A_VALID_MODEL_ARTIFACT",
            )
        MemoryGuard.assert_can_ingest_file(1024)  # trivially small; schema work is in-RAM light
        predict_obj, final_est, steps = UniversalSchemaDetector.unwrap_model_container(obj)
        UniversalSchemaDetector.validate_is_fitted(final_est)
        adapter = ModelRegistry.resolve_adapter(final_est, file_ext)
        schema = UniversalSchemaDetector.extract_schema(final_est)
        weights = adapter.extract_feature_weights(final_est, schema.n_features_in)
        for feature, weight in zip(schema.features, weights):
            feature.learned_weight_or_importance = weight
        try:
            raw_params = final_est.get_params(deep=False)
        except (AttributeError, TypeError, ValueError):
            raw_params = {}
        intercept: float | None = None
        raw_intercept = getattr(final_est, "intercept_", None)
        if raw_intercept is not None:
            try:
                import numpy as np  # noqa: PLC0415

                arr = np.asarray(raw_intercept, dtype=np.float64).ravel()
                if arr.size == 1 and bool(np.isfinite(arr[0])):
                    intercept = float(arr[0])
            except (TypeError, ValueError):
                intercept = None
        metadata = ModelMetadata(
            model_id=model_id,
            filename=filename,
            sha256_hash=digest,
            algorithm_key=adapter.algorithm_key,
            display_name=adapter.display_name,
            family_name=adapter.family_name,
            task_type=adapter.task_type.value,
            visual_3d_type=adapter.visual_3d_type.value,
            is_pipeline=len(steps) > 1,
            pipeline_steps=steps,
            n_features_in=schema.n_features_in,
            classes=schema.classes,
            intercept_or_base=intercept,
            features=schema.features,
            hyperparameters=_sanitize_hyperparameters(raw_params),
            warnings=load_warnings,
        )
        del predict_obj
        gc.collect()
        return metadata

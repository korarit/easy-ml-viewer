"""Joblib artifact loader — mmap deserialization, warning capture, LRU RAM cap."""

from __future__ import annotations

import gc
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib

from app.core.exceptions import DomainException


def _pin_single_thread(estimator: Any, _seen: set[int] | None = None) -> None:
    """Force ``n_jobs = 1`` throughout nested estimators (Windows loky safety)."""
    seen = _seen if _seen is not None else set()
    if id(estimator) in seen:
        return
    seen.add(id(estimator))
    try:
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
    except (AttributeError, ValueError, TypeError):
        pass
    for child_attr in ("best_estimator_", "estimator_", "base_estimator_"):
        child = getattr(estimator, child_attr, None)
        if child is not None and not isinstance(child, (str, int, float, bool)):
            try:
                _pin_single_thread(child, seen)
            except (AttributeError, TypeError, ValueError):
                pass
    steps = getattr(estimator, "steps", None)
    if isinstance(steps, list):
        for step in steps:
            try:
                _pin_single_thread(step[1] if isinstance(step, tuple) else step, seen)
            except (AttributeError, TypeError, ValueError):
                pass


class JoblibModelRepository:
    """Holds at most ``max_models_in_ram`` deserialized models (Active + Compare)."""

    def __init__(self, cache_dir: str | Path | None = None, max_models_in_ram: int = 2) -> None:
        from app.core.config import settings  # noqa: PLC0415 (lazy: avoids import cycle)

        self._cache_dir = Path(cache_dir) if cache_dir else settings.MODEL_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_models = max(1, max_models_in_ram)
        self._models: OrderedDict[str, Any] = OrderedDict()

    def load_and_cache(self, model_id: str, file_path: str | Path) -> tuple[Any, list[str]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = joblib.load(file_path, mmap_mode="r")
        load_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
        _pin_single_thread(obj)
        self._models[model_id] = obj
        self._models.move_to_end(model_id)
        while len(self._models) > self._max_models:
            _, evicted = self._models.popitem(last=False)
            del evicted
        gc.collect()
        return obj, load_warnings

    def get(self, model_id: str) -> Any:
        try:
            obj = self._models[model_id]
        except KeyError as exc:
            raise DomainException(
                code="MODEL_CACHE_MISS",
                message=f"Cached model '{model_id}' not found (evicted or never loaded)",
                http_status=404,
            ) from exc
        self._models.move_to_end(model_id)
        return obj

    def active_model_ids(self) -> list[str]:
        return list(self._models.keys())

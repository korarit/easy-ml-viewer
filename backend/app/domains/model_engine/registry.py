"""Plugin registry — new model families hook in via decorator, core stays closed."""

from __future__ import annotations

from typing import Any, Type

from app.core.exceptions import ModelLoadError
from app.domains.model_engine.base_adapter import BaseModelAdapter


class ModelRegistry:
    _adapters: dict[str, Type[BaseModelAdapter]] = {}

    @classmethod
    def register(cls, key: str):  # type: ignore[no-untyped-def]
        def decorator(adapter_cls: Type[BaseModelAdapter]) -> Type[BaseModelAdapter]:
            adapter_cls.algorithm_key = key
            cls._adapters[key] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def resolve_adapter(cls, estimator: Any, file_ext: str = ".joblib") -> BaseModelAdapter:
        for adapter_cls in cls._adapters.values():
            try:
                if adapter_cls.can_handle(estimator, file_ext):
                    return adapter_cls()
            except Exception:
                continue
        supported = ", ".join(sorted(cls._adapters.keys())) or "(none registered)"
        raise ModelLoadError(
            f"Unsupported model class '{type(estimator).__name__}' "
            f"(supported: {supported})",
            details={
                "estimator_class": type(estimator).__name__,
                "supported_algorithms": sorted(cls._adapters.keys()),
            },
            code="UNSUPPORTED_MODEL_TYPE",
        )

    @classmethod
    def supported_algorithms(cls) -> list[dict[str, str]]:
        return [
            {
                "algorithm_key": key,
                "display_name": adapter.display_name,
                "family_name": adapter.family_name,
                "task_type": adapter.task_type.value,
                "visual_3d_type": adapter.visual_3d_type.value,
            }
            for key, adapter in sorted(cls._adapters.items())
        ]

    @classmethod
    def clear(cls) -> None:
        """Test-only: reset registrations (adapters re-import to re-register)."""
        cls._adapters.clear()

"""Model repository port (DDD repository interface)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ModelRepository(Protocol):
    def load_and_cache(self, model_id: str, file_path: str | Path) -> tuple[Any, list[str]]:
        """Deserialize artifact, cache in RAM (LRU), return (model_obj, warnings)."""
        ...

    def get(self, model_id: str) -> Any:
        ...

    def active_model_ids(self) -> list[str]:
        ...

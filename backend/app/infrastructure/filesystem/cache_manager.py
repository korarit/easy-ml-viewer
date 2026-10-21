"""Filesystem-backed cache manager (infrastructure layer)."""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.exceptions import CacheOperationError


class CacheManager:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or settings.CACHE_DIR

    def ensure_directories(self) -> Path:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
            (self._cache_dir / "models").mkdir(parents=True, exist_ok=True)
            return self._cache_dir
        except OSError as exc:
            raise CacheOperationError(f"Cannot create cache dir: {exc}") from exc

    def get_total_cache_size_mb(self) -> float:
        if not self._cache_dir.exists():
            return 0.0
        total = 0
        for p in self._cache_dir.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
        return round(total / (1024 * 1024), 3)

    def clear_cache(self) -> tuple[float, int]:
        """Delete cached files. Returns (freed_mb, files_removed)."""
        if not self._cache_dir.exists():
            return 0.0, 0
        freed = 0
        removed = 0
        for p in sorted(self._cache_dir.rglob("*")):
            try:
                if p.is_file():
                    freed += p.stat().st_size
                    p.unlink()
                    removed += 1
            except OSError:
                continue
        freed_mb = round(freed / (1024 * 1024), 3)
        self.ensure_directories()
        return freed_mb, removed

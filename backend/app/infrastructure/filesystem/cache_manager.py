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

    def sweep_stale_cache(
        self, max_age_hours: float = 24.0, quota_gb: float = 1.0
    ) -> tuple[float, int]:
        """Phase 40: auto-clean `.parquet` temp files older than 24h (LRU).

        Eviction order: oldest mtime first. Stops once the total cache
        size is back under ``quota_gb`` (default 1.0 GB DoD ceiling).
        Returns ``(freed_mb, files_removed)``; never raises.
        """
        import time

        if not self._cache_dir.exists():
            return 0.0, 0
        try:
            files = [p for p in self._cache_dir.rglob("*") if p.is_file()]
        except OSError:
            return 0.0, 0
        now = time.time()
        quota_bytes = float(quota_gb) * (1024**3)

        def _size(p) -> int:
            try:
                return p.stat().st_size
            except OSError:
                return 0

        total = sum(_size(p) for p in files)
        # Oldest-first candidate list.
        def _mtime(p) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return now

        candidates = sorted(files, key=_mtime)
        freed = 0
        removed = 0
        for p in candidates:
            age_hours = (now - _mtime(p)) / 3600.0
            over_quota = total > quota_bytes
            if age_hours < max_age_hours and not over_quota:
                continue
            size = _size(p)
            try:
                p.unlink()
            except OSError:
                continue
            freed += size
            total -= size
            removed += 1
        return round(freed / (1024 * 1024), 3), removed

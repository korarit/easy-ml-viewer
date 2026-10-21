"""Shared kernel configuration — cross-platform pathlib resolution + 2-port contract."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_base_dir() -> Path:
    # backend/app/core/config.py -> parents[3] == repo root (easy-ml-viewer/)
    return Path(__file__).resolve().parent.parent.parent.parent


class AppSettings(BaseSettings):
    APP_NAME: str = "Easy ML Viewer Backend"
    APP_VERSION: str = "1.0.0"
    HOST: str = "127.0.0.1"
    PORT: int = 30211
    FRONTEND_ORIGINS: list[str] = [
        "http://localhost:30222",
        "http://127.0.0.1:30222",
    ]

    # NOTE: BASE_DIR is the backend/ folder ( historical ); REPO_ROOT is the
    # monorepo root that owns .cache/ and samples/ per the Phase 02 blueprint.
    BASE_DIR: Path = _default_base_dir() / "backend"
    REPO_ROOT: Path = _default_base_dir()
    CACHE_DIR: Path = _default_base_dir() / ".cache"
    DATASET_CACHE_DIR: Path = _default_base_dir() / ".cache" / "datasets"
    MODEL_CACHE_DIR: Path = _default_base_dir() / ".cache" / "models"
    SAMPLE_DIR: Path = _default_base_dir() / "samples"

    MAX_PROCESS_RAM_MB: int = 4096
    MIN_SYSTEM_FREE_RAM_MB: int = 512
    MAX_UPLOAD_SIZE_MB: int = 1024
    OMP_NUM_THREADS: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("FRONTEND_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


settings = AppSettings()


def ensure_cache_dirs() -> None:
    """Create .cache/ tree. Falls back to system temp dir when read-only."""
    import tempfile

    for path in (settings.CACHE_DIR, settings.DATASET_CACHE_DIR, settings.MODEL_CACHE_DIR):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "easy-ml-viewer-cache"
            for fb in (fallback, fallback / "datasets", fallback / "models"):
                fb.mkdir(parents=True, exist_ok=True)
            # Re-point in-memory settings so the rest of the app keeps working.
            settings.CACHE_DIR = fallback
            settings.DATASET_CACHE_DIR = fallback / "datasets"
            settings.MODEL_CACHE_DIR = fallback / "models"
            break


def apply_thread_pinning() -> None:
    """Pin BLAS/OMP threads BEFORE numpy/sklearn import to avoid oversubscription."""
    n = str(settings.OMP_NUM_THREADS)
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)

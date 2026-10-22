"""FastAPI application entrypoint — thread pinning FIRST, then imports."""

from __future__ import annotations

from app.core.config import apply_thread_pinning, ensure_cache_dirs

apply_thread_pinning()  # must run before numpy/sklearn are ever imported
ensure_cache_dirs()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.presentation.api_router import api_router  # noqa: E402
from app.presentation.exception_handlers import register_exception_handlers  # noqa: E402

log = get_logger("easy_ml_viewer")


def _startup_cache_sweep() -> None:
    """Phase 40: evict `.cache` temp files older than 24h / over 1 GB quota."""
    try:
        from app.infrastructure.filesystem.cache_manager import CacheManager

        freed_mb, removed = CacheManager().sweep_stale_cache()
        if removed:
            log.info("Cache sweep: removed %d stale files (%.1f MB)", removed, freed_mb)
    except Exception as exc:  # never block server startup on cache I/O
        log.warning("Cache sweep skipped: %s", exc)


def create_app() -> FastAPI:
    _startup_cache_sweep()
    app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[*settings.FRONTEND_ORIGINS, "http://localhost:30222", "http://127.0.0.1:30222"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": settings.APP_NAME, "version": settings.APP_VERSION}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    log.info("Starting %s on %s:%s", settings.APP_NAME, settings.HOST, settings.PORT)
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)

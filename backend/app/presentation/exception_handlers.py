"""Uniform JSON error envelope for all DomainExceptions + fallbacks."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import DomainException
from app.core.logging import get_logger

log = get_logger("easy_ml_viewer.errors")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainException)
    async def _domain_error(_: Request, exc: DomainException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Unexpected server error",
                    "details": {},
                }
            },
        )

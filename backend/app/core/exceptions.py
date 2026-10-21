"""Standardised domain exceptions -> uniform JSON error envelope."""

from __future__ import annotations

from typing import Any


class DomainException(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.http_status = http_status


class MemoryLimitExceededError(DomainException):
    def __init__(self, required_mb: float, available_mb: float) -> None:
        super().__init__(
            code="MEMORY_LIMIT_EXCEEDED",
            message=(
                f"Insufficient memory (required {required_mb:.1f} MB, "
                f"available {available_mb:.1f} MB)"
            ),
            details={"required_mb": required_mb, "available_mb": available_mb},
            http_status=413,
        )


class ModelLoadError(DomainException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="MODEL_LOAD_ERROR",
            message=message,
            details=details,
            http_status=422,
        )


class SchemaMismatchError(DomainException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="SCHEMA_MISMATCH",
            message=message,
            details=details,
            http_status=422,
        )


class CacheOperationError(DomainException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="CACHE_OPERATION_ERROR",
            message=message,
            details=details,
            http_status=500,
        )

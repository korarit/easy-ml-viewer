"""Minimal in-memory domain event bus (Shared Kernel)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeVar


@dataclass(frozen=True)
class DomainEvent:
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


TEvent = TypeVar("TEvent", bound=DomainEvent)


class InMemoryEventBus:
    def __init__(self) -> None:
        self._subscribers: dict[type[DomainEvent], list[Callable[[DomainEvent], None]]] = {}

    def subscribe(self, event_type: type[TEvent], handler: Callable[[TEvent], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)  # type: ignore[arg-type]

    def publish(self, event: DomainEvent) -> None:
        for handler in self._subscribers.get(type(event), []):
            handler(event)


event_bus = InMemoryEventBus()


@dataclass(frozen=True, kw_only=True)
class CacheClearedEvent(DomainEvent):
    freed_mb: float


@dataclass(frozen=True, kw_only=True)
class GarbageCollectedEvent(DomainEvent):
    process_rss_mb: float

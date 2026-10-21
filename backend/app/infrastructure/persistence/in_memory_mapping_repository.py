"""In-memory mapping persistence (Phase 05 §4.1)."""

from __future__ import annotations

from collections import OrderedDict

from app.core.exceptions import DomainException
from app.domains.mapping.repository import MappingRepository
from app.domains.mapping.value_objects import MappingConfiguration


class InMemoryMappingRepository(MappingRepository):
    """Keeps at most ``max_entries`` mapping configs (LRU)."""

    _shared: OrderedDict[str, MappingConfiguration] | None = None

    def __init__(self, max_entries: int = 8) -> None:
        if InMemoryMappingRepository._shared is None:
            InMemoryMappingRepository._shared = OrderedDict()
        self._store = InMemoryMappingRepository._shared
        self._max = max(1, max_entries)

    def save(self, config: MappingConfiguration) -> MappingConfiguration:
        self._store[config.mapping_id] = config
        self._store.move_to_end(config.mapping_id)
        while len(self._store) > self._max:
            self._store.popitem(last=False)
        return config

    def get(self, mapping_id: str) -> MappingConfiguration:
        try:
            config = self._store[mapping_id]
        except KeyError as exc:
            raise DomainException(
                code="MAPPING_NOT_FOUND",
                message=f"Mapping '{mapping_id}' not found",
                http_status=404,
            ) from exc
        self._store.move_to_end(mapping_id)
        return config

    def latest(self) -> MappingConfiguration | None:
        if not self._store:
            return None
        return next(reversed(self._store.values()))

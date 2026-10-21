"""Mapping repository port (Phase 05 §4.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domains.mapping.value_objects import MappingConfiguration


class MappingRepository(ABC):
    @abstractmethod
    def save(self, config: MappingConfiguration) -> MappingConfiguration: ...

    @abstractmethod
    def get(self, mapping_id: str) -> MappingConfiguration: ...

    @abstractmethod
    def latest(self) -> MappingConfiguration | None: ...

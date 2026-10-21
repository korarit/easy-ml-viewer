"""Phase 09 — LRU surface cache (≤8 entries, process-wide shared)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from app.domains.spatial_projection.entities import SurfaceMesh3DEntity


class SurfaceMeshCacheRepository:
    """Caches baked :class:`SurfaceMesh3DEntity` results.

    Key: ``(dataset, model, mapping, mode, x, y, z_mode, class, pinning,
    lod, frame, robust, queue_tuple)``. A 60x60 entity holds ~7k floats
    (<100 KB JSON); 8 entries stay well under the 5 MB budget (§5.1).
    Re-selecting a previous axis pair answers in <1 ms.
    """

    _shared: OrderedDict[tuple[Any, ...], SurfaceMesh3DEntity] | None = None

    def __init__(self, max_entries: int = 8) -> None:
        if SurfaceMeshCacheRepository._shared is None:
            SurfaceMeshCacheRepository._shared = OrderedDict()
        self._store = SurfaceMeshCacheRepository._shared
        self._max = max(1, max_entries)

    def get(self, key: tuple[Any, ...]) -> SurfaceMesh3DEntity | None:
        entity = self._store.get(key)
        if entity is not None:
            self._store.move_to_end(key)
        return entity

    def put(self, key: tuple[Any, ...], entity: SurfaceMesh3DEntity) -> None:
        self._store[key] = entity
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    @classmethod
    def clear(cls) -> None:
        """Test-only: reset the process-wide cache."""
        if cls._shared is not None:
            cls._shared.clear()

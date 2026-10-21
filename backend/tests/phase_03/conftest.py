"""Isolation: snapshot the plugin registry so dummy adapters never leak."""

from __future__ import annotations

import pytest

from app.domains.model_engine.registry import ModelRegistry


@pytest.fixture()
def isolated_registry():
    snapshot = dict(ModelRegistry._adapters)
    try:
        yield ModelRegistry
    finally:
        ModelRegistry._adapters.clear()
        ModelRegistry._adapters.update(snapshot)

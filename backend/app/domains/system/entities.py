"""System bounded-context entities (domain state, framework-free)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SystemNode:
    """Represents the host machine as seen by the System context."""

    os_platform: str
    python_version: str
    cpu_logical_cores: int
    omp_threads: int

    def describe(self) -> str:
        return f"{self.os_platform} | py{self.python_version} | {self.cpu_logical_cores} cores"

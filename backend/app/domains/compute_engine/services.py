"""Phase 17 — compute_engine pure domain services (no FastAPI/psutil)."""

from __future__ import annotations

from app.domains.compute_engine.value_objects import MemoryGuardLevel


def memory_guard_level(system_memory_percent: float) -> MemoryGuardLevel:
    """Map RAM pressure to the HardwareStatusPill traffic light.

    <60% normal (green) · 60-80% elevated (amber, strict GC) ·
    >80% guard active (red, block big new jobs).
    """
    if system_memory_percent >= 80.0:
        return "guard_active"
    if system_memory_percent >= 60.0:
        return "elevated"
    return "normal"


def estimate_chunks(total_rows: int, chunk_size: int) -> int:
    if total_rows <= 0 or chunk_size <= 0:
        return 0
    return (total_rows + chunk_size - 1) // chunk_size


def estimate_peak_extra_mb(n_features: int, chunk_size: int) -> float:
    """Conservative 4x float32 working-set estimate per chunk (matches core)."""
    n_features = max(int(n_features), 1)
    chunk_size = max(int(chunk_size), 1)
    return round((chunk_size * n_features * 4 * 4) / (1024 * 1024), 3)


def status_pill_label(
    physical_cores: int,
    logical_cores: int,
    rss_mb: float,
    system_available_mb: float,
    providers: list[str],
) -> str:
    gpu = next(
        (p.replace("ExecutionProvider", "") for p in providers if p != "CPUExecutionProvider"),
        "CPU-only",
    )
    return (
        f"[ CPU: {physical_cores} Cores ({logical_cores} Threads) "
        f"| RAM: {rss_mb:.0f} MB avail {system_available_mb:.0f} MB "
        f"| GPU: {gpu} ]"
    )

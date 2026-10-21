"""Phase 17 — adaptive multi-core CPU pool, GPU probe & chunked execution.

Shared kernel (``app/core/``): owns ``ThreadPoolExecutor`` sizing,
``OMP/MKL`` thread-pinning guard, ONNX provider probing and the flat-memory
chunked generator used by every heavy Phase 18+ simulation.

Windows mandate (§5.2): never use ``ProcessPoolExecutor`` (spawn + pickle
copies the model per core). ``numpy`` / ``scikit-learn`` / GBDT / ONNX all
release the GIL in C/C++/Cython, so ``ThreadPoolExecutor`` with
``n_jobs = optimal_workers`` shares one model in RAM (zero-copy).
"""

from __future__ import annotations

import gc
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, Literal, TypeVar

import numpy as np
import psutil

T = TypeVar("T")

DEFAULT_CHUNK_SIZE = 5_000
CHUNK_SIZE_OPTIONS = (2_500, 5_000, 10_000)


class InsufficientMemoryBudgetError(MemoryError):
    """Raised when free system RAM drops below the safe pre-flight floor."""


class JobCancelledError(RuntimeError):
    """Raised when the user cancels a chunked compute job mid-flight."""


class CancellationToken:
    """Cooperative cancel flag checked between 5k-row chunks (~20-50 ms)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise JobCancelledError("Compute job cancelled by user")


def _physical_cores() -> int:
    return int(psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 2)


def _logical_cores() -> int:
    return int(psutil.cpu_count(logical=True) or _physical_cores())


class AdaptiveComputeEngine:
    """Central manager for CPU threads, GPU providers and memory ceiling."""

    def __init__(self, max_memory_usage_ratio: float = 0.75) -> None:
        self.physical_cores: int = _physical_cores()
        self.logical_cores: int = _logical_cores()
        # Keep ≥1 core free for the FastAPI event loop + OS UI.
        self.optimal_workers: int = max(1, self.physical_cores - 1)
        self.max_memory_usage_ratio: float = max_memory_usage_ratio

        # Prevent Thread oversubscription between OpenMP/MKL and the pool.
        os.environ.setdefault("OMP_NUM_THREADS", str(self.optimal_workers))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.optimal_workers))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.optimal_workers))

        self.executor = ThreadPoolExecutor(
            max_workers=self.optimal_workers,
            thread_name_prefix="emlv_compute_worker",
        )
        self.available_onnx_providers: list[str] = self._probe_onnx_providers()

    @staticmethod
    def _probe_onnx_providers() -> list[str]:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            providers = list(ort.get_available_providers())
            return providers or ["CPUExecutionProvider"]
        except Exception:
            return ["CPUExecutionProvider"]

    def get_preferred_onnx_providers(self, prefer_gpu: bool = True) -> list[str]:
        if not prefer_gpu:
            return ["CPUExecutionProvider"]
        priority_order = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        ranked = [p for p in priority_order if p in self.available_onnx_providers]
        return ranked or ["CPUExecutionProvider"]

    def route_workload(
        self,
        n_rows: int,
        is_deep_model: bool = False,
        device_mode: Literal["auto", "cpu_only", "prefer_gpu"] = "auto",
    ) -> Literal["cpu_vectorized", "gpu_accelerated"]:
        """Smart CPU vs GPU router (§4.1 + edge-case #6).

        Small/medium tabular batches stay on vectorized multi-core CPU to
        avoid Host-to-Device copy overhead; large deep/3D-surface tensors go
        to GPU when a non-CPU provider exists.
        """
        if device_mode == "cpu_only":
            return "cpu_vectorized"
        has_gpu = any(p != "CPUExecutionProvider" for p in self.available_onnx_providers)
        if device_mode == "prefer_gpu" and has_gpu and (is_deep_model or n_rows >= 10_000):
            return "gpu_accelerated"
        if is_deep_model and has_gpu and n_rows >= 10_000:
            return "gpu_accelerated"
        return "cpu_vectorized"

    def check_memory_budget(self, required_extra_bytes: int = 64 * 1024 * 1024) -> dict[str, float]:
        """Pre-flight RAM gate run before every chunk (HTTP 507 on failure)."""
        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        rss_bytes = proc.memory_info().rss

        if vm.percent >= (self.max_memory_usage_ratio * 100.0) or vm.available < required_extra_bytes:
            gc.collect()
            vm = psutil.virtual_memory()
            if vm.available < required_extra_bytes:
                avail_mb = vm.available / (1024 * 1024)
                req_mb = required_extra_bytes / (1024 * 1024)
                raise InsufficientMemoryBudgetError(
                    f"Insufficient system memory (need >= {req_mb:.0f} MB, "
                    f"only {avail_mb:.0f} MB available)"
                )

        return {
            "process_rss_mb": round(rss_bytes / (1024 * 1024), 2),
            "system_available_mb": round(vm.available / (1024 * 1024), 2),
            "system_percent": float(vm.percent),
        }

    def process_in_chunks(
        self,
        X: np.ndarray,
        chunk_fn: Callable[[np.ndarray, int, int], T],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        cancel_token: CancellationToken | None = None,
        on_progress: Callable[[int, int, float], None] | None = None,
    ) -> Iterator[T]:
        """Yield ``chunk_fn`` results one ``chunk_size``-row block at a time.

        Flat memory profile: ``del chunk_view`` after each block + RAM +
        cancellation check before every block (~50 ms responsiveness).
        """
        total_rows = int(X.shape[0]) if X.ndim else 0
        if total_rows == 0:
            return
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        n_cols = int(X.shape[1]) if X.ndim > 1 else 1
        bytes_per_chunk = max(32 * 1024 * 1024, chunk_size * max(n_cols, 1) * 4 * 4)
        for start_idx in range(0, total_rows, chunk_size):
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            self.check_memory_budget(required_extra_bytes=bytes_per_chunk)

            end_idx = min(total_rows, start_idx + chunk_size)
            chunk_view = np.ascontiguousarray(X[start_idx:end_idx], dtype=np.float32)
            result = chunk_fn(chunk_view, start_idx, end_idx)
            del chunk_view

            if on_progress is not None:
                pct = round((end_idx / total_rows) * 100.0, 1)
                on_progress(end_idx, total_rows, pct)

            yield result


_compute_engine: AdaptiveComputeEngine | None = None
_engine_lock = threading.Lock()


def get_compute_engine() -> AdaptiveComputeEngine:
    """Process-wide singleton (pool must not be rebuilt per request)."""
    global _compute_engine
    if _compute_engine is None:
        with _engine_lock:
            if _compute_engine is None:
                _compute_engine = AdaptiveComputeEngine()
    return _compute_engine

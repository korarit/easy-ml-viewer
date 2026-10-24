"""Phase 15 — `.mlv` binary codec (EMLV v1.0: MsgPack + Zstandard).

File layout: ``MAGIC(8) + CRC32-LE(4) + zstd(msgpack(payload))``.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

import msgpack
import numpy as np
import zstandard as zstd

MLV_MAGIC_HEADER: bytes = b"EMLV\x01\x00\x00\x00"  # 8 bytes: Magic 'EMLV' + Version 1.0.0
HEADER_PREFIX_SIZE: int = 12  # 8 bytes Magic + 4 bytes uint32 CRC32
MAX_DECOMPRESSED_BYTES: int = 250 * 1024 * 1024  # 250 MB Zip-Bomb guard
MLV_SCHEMA_VERSION: str = "1.0"


class MlvFormatError(ValueError):
    """Raised when a `.mlv` file is malformed or corrupted."""


#: Phase 56 — dtypes the codec preserves verbatim (v1 float32 path untouched).
_PRESERVED_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "float16", "float32")


def _msgpack_default_encoder(obj: Any) -> Any:
    """Map NumPy scalars / arrays to MessagePack-compatible types."""
    if isinstance(obj, np.ndarray):
        # Phase 56: preserve compact quantized dtypes (int16/uint8/...) so
        # point-intelligence tracks stay under the 150 KB spec budget.
        # All other dtypes keep the legacy float32 path (backward compatible).
        dtype_name = str(obj.dtype.name) if obj.dtype.name in _PRESERVED_DTYPES else "float32"
        arr = np.ascontiguousarray(obj, dtype=np.dtype(dtype_name))
        return {
            "__nd__": True,
            "dtype": dtype_name,
            "shape": list(arr.shape),
            "bytes": arr.tobytes(),
        }
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not Mlv-serializable")


def _msgpack_object_hook(obj: dict[Any, Any]) -> Any:
    """Restore binary buffers back to read-write NumPy arrays."""
    if obj.get("__nd__") is True:
        raw_bytes: bytes = obj["bytes"]
        shape: tuple[int, ...] = tuple(obj["shape"])
        dtype_name = str(obj.get("dtype", "float32"))
        if dtype_name not in _PRESERVED_DTYPES:
            dtype_name = "float32"  # legacy / unknown → honest float32 fallback
        return np.frombuffer(raw_bytes, dtype=np.dtype(dtype_name)).reshape(shape).copy()
    return obj


def encode_mlv_package(payload_dict: dict[str, Any], compression_level: int = 3) -> bytes:
    """Pack a payload dict into `.mlv` binary bytes with Magic + CRC32."""
    packed_msg = msgpack.packb(payload_dict, default=_msgpack_default_encoder, use_bin_type=True)
    compressor = zstd.ZstdCompressor(level=compression_level, threads=-1)
    compressed_payload = compressor.compress(packed_msg)
    del packed_msg  # free the pre-compression buffer immediately

    crc_val = zlib.crc32(compressed_payload) & 0xFFFFFFFF
    crc_bytes = struct.pack("<I", crc_val)
    return MLV_MAGIC_HEADER + crc_bytes + compressed_payload


def decode_mlv_package(file_bytes: bytes) -> dict[str, Any]:
    """Validate Magic + CRC32 and unpack `.mlv` bytes back to a payload dict."""
    if len(file_bytes) < HEADER_PREFIX_SIZE:
        raise MlvFormatError("File too small for the .mlv v1.0 standard (Truncated File)")

    magic = file_bytes[:8]
    if magic != MLV_MAGIC_HEADER:
        raise MlvFormatError("Header is not valid .mlv v1.0 format (Invalid Magic Header)")

    expected_crc = struct.unpack("<I", file_bytes[8:12])[0]
    compressed_payload = file_bytes[12:]
    actual_crc = zlib.crc32(compressed_payload) & 0xFFFFFFFF

    if actual_crc != expected_crc:
        raise MlvFormatError(
            f".mlv file corrupted during storage or download "
            f"(CRC32 Mismatch: {actual_crc:#x} != {expected_crc:#x})"
        )

    decompressor = zstd.ZstdDecompressor()
    uncompressed_msg = decompressor.decompress(compressed_payload, max_output_size=MAX_DECOMPRESSED_BYTES)
    return msgpack.unpackb(uncompressed_msg, object_hook=_msgpack_object_hook, raw=False)

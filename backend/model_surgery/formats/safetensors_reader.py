"""Safetensors header reader.

Reads only the 8-byte length prefix and the JSON header. Tensor contents are
never read. Header JSON is untrusted: duplicate keys, unknown dtypes, bad
shapes, inconsistent byte ranges, overlaps and out-of-file offsets are all
detected.

Format: https://github.com/huggingface/safetensors#format
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import FormatError

# Same cap as the reference implementation (100 MB header).
MAX_HEADER_BYTES = 100 * 1024 * 1024
MAX_TENSORS = 1_000_000
MAX_DIMS = 16

# dtype -> bits per element. Sub-byte types (F4/F6) pack along the last axis.
DTYPE_BITS: dict[str, int] = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
}


@dataclass
class STTensor:
    name: str
    dtype: str
    shape: list[int]
    begin: int  # relative to the data section
    end: int
    n_elements: int

    @property
    def n_bytes(self) -> int:
        return self.end - self.begin


@dataclass
class SafetensorsFile:
    path: Path
    file_size: int
    header_size: int
    metadata: dict[str, str]
    tensors: list[STTensor]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def data_offset(self) -> int:
        return 8 + self.header_size

    @property
    def valid(self) -> bool:
        return not self.errors


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise FormatError(f"duplicate key in safetensors header: '{k}'")
        out[k] = v
    return out


def _is_uint(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def read_safetensors(path: str | os.PathLike) -> SafetensorsFile:
    p = Path(path)
    if not p.is_file():
        raise FormatError(f"not a regular file: {p}")
    file_size = p.stat().st_size
    if file_size < 8:
        raise FormatError("file too small to be safetensors")

    with open(p, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n > MAX_HEADER_BYTES:
            raise FormatError(f"header length {n} exceeds {MAX_HEADER_BYTES} byte limit")
        if 8 + n > file_size:
            raise FormatError(f"header length {n} exceeds file size {file_size}")
        raw = fh.read(n)
    if len(raw) != n:
        raise FormatError("short read of safetensors header")
    if not raw.lstrip().startswith(b"{"):
        raise FormatError("safetensors header is not a JSON object")

    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except UnicodeDecodeError as e:
        raise FormatError(f"header is not valid UTF-8: {e}") from None
    except json.JSONDecodeError as e:
        raise FormatError(f"header is not valid JSON: {e.msg} at {e.pos}") from None
    if not isinstance(header, dict):
        raise FormatError("safetensors header is not a JSON object")

    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, str] = {}
    meta = header.pop("__metadata__", None)
    if meta is not None:
        if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
            errors.append("__metadata__ must map strings to strings")
        else:
            metadata = {k: (v if len(v) <= 16_384 else v[:16_384]) for k, v in meta.items()}

    if len(header) > MAX_TENSORS:
        raise FormatError(f"{len(header)} tensors exceeds limit {MAX_TENSORS}")

    data_size = file_size - 8 - n
    tensors: list[STTensor] = []
    for name, info in header.items():
        if not isinstance(info, dict):
            errors.append(f"tensor '{name[:120]}': descriptor is not an object")
            continue
        dtype, shape, offs = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in DTYPE_BITS:
            errors.append(f"tensor '{name}': unknown dtype {dtype!r}")
            continue
        if not isinstance(shape, list) or len(shape) > MAX_DIMS or not all(_is_uint(d) for d in shape):
            errors.append(f"tensor '{name}': invalid shape {shape!r}")
            continue
        if not (isinstance(offs, list) and len(offs) == 2 and all(_is_uint(o) for o in offs) and offs[0] <= offs[1]):
            errors.append(f"tensor '{name}': invalid data_offsets {offs!r}")
            continue
        n_el = 1
        for d in shape:
            n_el *= d
        if n_el > 1 << 50:
            errors.append(f"tensor '{name}': shape {shape} is implausibly large")
            continue
        bits = n_el * DTYPE_BITS[dtype]
        expected = (bits + 7) // 8
        if offs[1] - offs[0] != expected:
            errors.append(f"tensor '{name}': byte range {offs[1] - offs[0]} != {expected} expected for {dtype}{shape}")
        if offs[1] > data_size:
            errors.append(f"tensor '{name}': data ends at {offs[1]} beyond data section of {data_size} bytes (truncated file?)")
        tensors.append(STTensor(name, dtype, list(shape), offs[0], offs[1], n_el))

    ordered = sorted(tensors, key=lambda t: (t.begin, t.end))
    cursor = 0
    for t in ordered:
        if t.begin < cursor:
            errors.append(f"tensor '{t.name}' overlaps a preceding tensor")
        elif t.begin > cursor:
            warnings.append(f"{t.begin - cursor} byte gap before tensor '{t.name}'")
        cursor = max(cursor, t.end)
    if cursor < data_size:
        warnings.append(f"{data_size - cursor} trailing bytes not referenced by any tensor")

    if len(errors) > 200:
        errors = errors[:200] + [f"... {len(errors) - 200} more integrity errors"]
    if len(warnings) > 50:
        warnings = warnings[:50] + [f"... {len(warnings) - 50} more warnings"]

    return SafetensorsFile(p, file_size, n, metadata, tensors, errors, warnings)

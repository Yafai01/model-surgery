"""Pure-Python GGUF header reader.

Reads the GGUF header, metadata key/value pairs and tensor descriptors without
reading tensor data. The file is treated as untrusted: every length, count,
shape, type id and offset is bounds-checked against the real file size and a
set of hard limits before it is used.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from . import FormatError
from .ggml_types import GGML_TYPES, GGMLType, tensor_nbytes

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
GGML_MAX_DIMS = 4
GGML_MAX_NAME = 64

# Value type ids from the GGUF spec.
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL = range(8)
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(8, 13)

_SCALAR_FMT = {
    T_UINT8: ("B", "uint8"),
    T_INT8: ("b", "int8"),
    T_UINT16: ("H", "uint16"),
    T_INT16: ("h", "int16"),
    T_UINT32: ("I", "uint32"),
    T_INT32: ("i", "int32"),
    T_FLOAT32: ("f", "float32"),
    T_BOOL: ("?", "bool"),
    T_UINT64: ("Q", "uint64"),
    T_INT64: ("q", "int64"),
    T_FLOAT64: ("d", "float64"),
}
_TYPE_NAMES = {k: v[1] for k, v in _SCALAR_FMT.items()} | {T_STRING: "string", T_ARRAY: "array"}


@dataclass(frozen=True)
class GGUFLimits:
    max_tensors: int = 200_000
    max_kv: int = 100_000
    max_string_bytes: int = 64 * 1024 * 1024
    max_array_len: int = 50_000_000
    max_array_depth: int = 3
    max_alignment: int = 1024 * 1024
    max_name_bytes: int = 1024
    # What we keep in memory for display. Anything longer is truncated and flagged.
    keep_string_chars: int = 16_384
    keep_array_items: int = 32
    keep_numeric_array_items: int = 1024
    keep_array_item_chars: int = 256


@dataclass
class GGUFValue:
    type: str  # "uint32", "string", "array[string]", ...
    value: Any  # full value for scalars, possibly-truncated preview for strings/arrays
    length: int | None = None  # element count for arrays, byte length for strings
    truncated: bool = False


@dataclass
class GGUFTensor:
    name: str
    shape: list[int]  # GGML order: shape[0] is the fastest-varying dimension (ne0)
    type_id: int
    ggml_type: GGMLType | None
    offset: int  # relative to the start of the tensor data section
    n_elements: int
    n_bytes: int | None


@dataclass
class GGUFFile:
    path: Path
    file_size: int
    version: int
    byte_order: str  # "little" | "big"
    alignment: int
    data_offset: int
    metadata: dict[str, GGUFValue]
    tensors: list[GGUFTensor]
    errors: list[str] = field(default_factory=list)  # integrity failures
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def get(self, key: str, default: Any = None) -> Any:
        v = self.metadata.get(key)
        return default if v is None else v.value

    def get_array_len(self, key: str) -> int | None:
        v = self.metadata.get(key)
        return v.length if v is not None and v.type.startswith("array") else None


class _Reader:
    def __init__(self, fh: BinaryIO, file_size: int, limits: GGUFLimits):
        self.fh = fh
        self.size = file_size
        self.limits = limits
        self.bo = "<"

    @property
    def pos(self) -> int:
        return self.fh.tell()

    def remaining(self) -> int:
        return self.size - self.pos

    def read(self, n: int, what: str) -> bytes:
        if n < 0 or n > self.remaining():
            raise FormatError(f"truncated file: need {n} bytes for {what} at offset {self.pos}, {self.remaining()} left")
        data = self.fh.read(n)
        if len(data) != n:
            raise FormatError(f"short read for {what} at offset {self.pos}")
        return data

    def skip(self, n: int, what: str) -> None:
        if n < 0 or n > self.remaining():
            raise FormatError(f"truncated file: cannot skip {n} bytes for {what}")
        self.fh.seek(n, os.SEEK_CUR)

    def scalar(self, t: int, what: str) -> Any:
        fmt = self.bo + _SCALAR_FMT[t][0]
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size, what))[0]

    def u32(self, what: str) -> int:
        return self.scalar(T_UINT32, what)

    def u64(self, what: str) -> int:
        return self.scalar(T_UINT64, what)

    def raw_string(self, what: str, max_bytes: int) -> bytes:
        n = self.u64(f"{what} length")
        if n > max_bytes:
            raise FormatError(f"{what} length {n} exceeds limit {max_bytes}")
        return self.read(n, what)

    def string(self, what: str, max_bytes: int | None = None) -> str:
        return self.raw_string(what, max_bytes or self.limits.max_string_bytes).decode("utf-8", errors="replace")


def _truncate(s: str, n: int) -> tuple[str, bool]:
    return (s, False) if len(s) <= n else (s[:n], True)


def _read_value(r: _Reader, t: int, key: str, depth: int = 0) -> GGUFValue:
    lim = r.limits
    if t in _SCALAR_FMT:
        v = r.scalar(t, key)
        if t == T_FLOAT32 or t == T_FLOAT64:
            v = float(v)
        return GGUFValue(_TYPE_NAMES[t], v)
    if t == T_STRING:
        raw = r.raw_string(key, lim.max_string_bytes)
        s, trunc = _truncate(raw.decode("utf-8", errors="replace"), lim.keep_string_chars)
        return GGUFValue("string", s, length=len(raw), truncated=trunc)
    if t == T_ARRAY:
        if depth >= lim.max_array_depth:
            raise FormatError(f"metadata '{key}': array nesting deeper than {lim.max_array_depth}")
        et = r.u32(f"{key} array type")
        n = r.u64(f"{key} array length")
        if et not in _TYPE_NAMES:
            raise FormatError(f"metadata '{key}': unknown array element type {et}")
        if n > lim.max_array_len:
            raise FormatError(f"metadata '{key}': array length {n} exceeds limit {lim.max_array_len}")
        preview: list[Any] = []
        truncated = False
        if et in _SCALAR_FMT:
            esize = struct.calcsize(_SCALAR_FMT[et][0])
            if n * esize > r.remaining():
                raise FormatError(f"truncated file: metadata '{key}' array needs {n * esize} bytes")
            # Short numeric arrays (e.g. per-layer head counts) are kept whole.
            keep = n if n <= lim.keep_numeric_array_items else lim.keep_array_items
            for _ in range(keep):
                v = r.scalar(et, key)
                preview.append(float(v) if et in (T_FLOAT32, T_FLOAT64) else v)
            r.skip((n - keep) * esize, key)
            truncated = n > keep
        else:
            # Strings and nested arrays have variable length; walk every element.
            if n * 8 > r.remaining():
                raise FormatError(f"truncated file: metadata '{key}' array of {n} elements cannot fit")
            for i in range(n):
                item = _read_value(r, et, key, depth + 1)
                if i < lim.keep_array_items:
                    v = item.value
                    if isinstance(v, str):
                        v, t2 = _truncate(v, lim.keep_array_item_chars)
                        truncated |= t2
                    preview.append(v)
            truncated |= n > lim.keep_array_items
        return GGUFValue(f"array[{_TYPE_NAMES[et]}]", preview, length=n, truncated=truncated)
    raise FormatError(f"metadata '{key}': unknown value type {t}")


def _detect_byte_order(fh: BinaryIO) -> tuple[int, str]:
    raw = fh.read(4)
    if len(raw) != 4:
        raise FormatError("truncated file: missing GGUF version")
    le = struct.unpack("<I", raw)[0]
    be = struct.unpack(">I", raw)[0]
    if 1 <= le <= 0xFFFF:
        return le, "little"
    if 1 <= be <= 0xFFFF:
        return be, "big"
    raise FormatError(f"implausible GGUF version field {le:#x}")


def is_gguf(path: str | os.PathLike) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def read_gguf(path: str | os.PathLike, limits: GGUFLimits | None = None) -> GGUFFile:
    """Parse a GGUF header. Raises FormatError for structurally unreadable files.

    Per-tensor integrity problems (out-of-bounds data, overlaps, misalignment)
    are collected in ``GGUFFile.errors`` so a partially valid file can still be
    described; callers must check ``.valid`` before trusting the tensor data.
    """
    limits = limits or GGUFLimits()
    p = Path(path)
    if not p.is_file():
        raise FormatError(f"not a regular file: {p}")
    file_size = p.stat().st_size

    with open(p, "rb", buffering=1024 * 1024) as fh:
        if fh.read(4) != GGUF_MAGIC:
            raise FormatError("missing GGUF magic bytes")
        version, byte_order = _detect_byte_order(fh)
        if version == 1:
            raise FormatError("GGUF v1 (pre-August-2023) is not supported; re-convert the model with current llama.cpp")
        if version > 3:
            raise FormatError(f"GGUF version {version} is newer than this reader supports (2-3)")

        r = _Reader(fh, file_size, limits)
        r.bo = "<" if byte_order == "little" else ">"
        warnings: list[str] = []
        errors: list[str] = []
        if byte_order == "big":
            warnings.append("big-endian GGUF: most llama.cpp builds only load little-endian files")

        n_tensors = r.u64("tensor count")
        n_kv = r.u64("metadata count")
        if n_tensors > limits.max_tensors:
            raise FormatError(f"tensor count {n_tensors} exceeds limit {limits.max_tensors}")
        if n_kv > limits.max_kv:
            raise FormatError(f"metadata count {n_kv} exceeds limit {limits.max_kv}")

        metadata: dict[str, GGUFValue] = {}
        for i in range(n_kv):
            key = r.string(f"metadata key #{i}", limits.max_name_bytes)
            if key in metadata:
                raise FormatError(f"duplicate metadata key '{key}'")
            vt = r.u32(f"type of '{key}'")
            metadata[key] = _read_value(r, vt, key)

        alignment = GGUF_DEFAULT_ALIGNMENT
        if "general.alignment" in metadata:
            av = metadata["general.alignment"]
            a = av.value
            if av.type != "uint32" or not isinstance(a, int) or a <= 0 or a & (a - 1) or a > limits.max_alignment:
                raise FormatError(f"invalid general.alignment {a!r}")
            alignment = a

        tensors: list[GGUFTensor] = []
        seen: set[str] = set()
        for i in range(n_tensors):
            name = r.string(f"tensor name #{i}", limits.max_name_bytes)
            if name in seen:
                raise FormatError(f"duplicate tensor name '{name}'")
            seen.add(name)
            if len(name.encode()) >= GGML_MAX_NAME:
                warnings.append(f"tensor name longer than ggml's {GGML_MAX_NAME}-byte limit: '{name[:80]}'")
            n_dims = r.u32(f"'{name}' n_dims")
            if not 1 <= n_dims <= GGML_MAX_DIMS:
                raise FormatError(f"tensor '{name}' has {n_dims} dimensions (allowed 1..{GGML_MAX_DIMS})")
            shape = [r.u64(f"'{name}' dim") for _ in range(n_dims)]
            type_id = r.u32(f"'{name}' type")
            offset = r.u64(f"'{name}' offset")

            n_el = 1
            for d in shape:
                n_el *= d
            if n_el > 1 << 50:
                raise FormatError(f"tensor '{name}' shape {shape} is implausibly large")
            gt = GGML_TYPES.get(type_id)
            n_bytes: int | None = None
            if gt is None:
                errors.append(f"tensor '{name}' uses unknown or removed ggml type id {type_id}")
            elif shape[0] % gt.block_size:
                errors.append(f"tensor '{name}': row length {shape[0]} not divisible by {gt.name} block size {gt.block_size}")
            else:
                n_bytes = tensor_nbytes(gt, shape)
            if 0 in shape:
                warnings.append(f"tensor '{name}' has a zero-length dimension")
            if offset % alignment:
                errors.append(f"tensor '{name}' offset {offset} is not aligned to {alignment}")
            tensors.append(GGUFTensor(name, shape, type_id, gt, offset, n_el, n_bytes))

        header_end = r.pos
        data_offset = header_end + (-header_end % alignment)
        if data_offset > file_size:
            if n_tensors:
                errors.append("file ends before the tensor data section begins")
            data_size = 0
        else:
            data_size = file_size - data_offset

    # Bounds and overlap checks on the tensor data section.
    sized = sorted((t for t in tensors if t.n_bytes is not None), key=lambda t: t.offset)
    prev: GGUFTensor | None = None
    for t in sized:
        assert t.n_bytes is not None
        if t.offset + t.n_bytes > data_size:
            errors.append(
                f"tensor '{t.name}' data [{t.offset}, {t.offset + t.n_bytes}) exceeds data section of {data_size} bytes (truncated file?)"
            )
        if prev is not None and prev.n_bytes is not None and prev.offset + prev.n_bytes > t.offset:
            errors.append(f"tensors '{prev.name}' and '{t.name}' overlap")
        prev = t
    if sized:
        last = sized[-1]
        end = last.offset + (last.n_bytes or 0)
        slack = data_size - end
        if slack > 64 * 1024 * 1024:
            warnings.append(f"{slack} unreferenced bytes after the last tensor")

    if len(errors) > 200:
        errors = errors[:200] + [f"... {len(errors) - 200} more integrity errors"]

    return GGUFFile(
        path=p,
        file_size=file_size,
        version=version,
        byte_order=byte_order,
        alignment=alignment,
        data_offset=data_offset,
        metadata=metadata,
        tensors=tensors,
        errors=errors,
        warnings=warnings,
    )

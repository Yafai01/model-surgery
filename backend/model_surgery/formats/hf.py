"""Hugging Face model directory reader (config.json + safetensors shards).

Only JSON and safetensors headers are read. Pickle-based weights
(pytorch_model.bin, *.pt, *.ckpt) are reported but never opened, because
unpickling can execute arbitrary code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import FormatError, UnsupportedFormatError
from .safetensors_reader import SafetensorsFile, read_safetensors

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_SHARDS = 4096

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
OTHER_WEIGHT_SUFFIXES = {".h5": "TensorFlow/Keras HDF5", ".msgpack": "Flax msgpack", ".onnx": "ONNX", ".gguf": "GGUF"}
SIDE_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
}


@dataclass
class UnparsedFile:
    path: Path
    size: int
    reason: str


@dataclass
class HFModelDir:
    path: Path
    config: dict[str, Any]
    config_path: Path
    index_path: Path | None
    shards: list[SafetensorsFile]
    side_files: list[Path] = field(default_factory=list)
    unparsed: list[UnparsedFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise FormatError(f"duplicate JSON key '{k}'")
        out[k] = v
    return out


def read_json_limited(path: Path, max_bytes: int | None = None) -> Any:
    max_bytes = max_bytes or MAX_JSON_BYTES
    size = path.stat().st_size
    if size > max_bytes:
        raise FormatError(f"{path.name} is {size} bytes; refusing to parse JSON larger than {max_bytes}")
    try:
        return json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=_no_duplicates)
    except UnicodeDecodeError:
        raise FormatError(f"{path.name} is not valid UTF-8") from None
    except json.JSONDecodeError as e:
        raise FormatError(f"{path.name} is not valid JSON: {e.msg} (line {e.lineno})") from None


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _safe_child(directory: Path, name: Any) -> Path:
    """Resolve a filename taken from untrusted JSON, refusing anything but a plain sibling file."""
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        raise FormatError(f"refusing unsafe shard filename {name!r} in index")
    p = directory / name
    if not _inside(p, directory):
        raise FormatError(f"shard {name!r} resolves outside the model directory")
    return p


def looks_like_hf_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def read_hf_dir(path: str | os.PathLike) -> HFModelDir:
    d = Path(path)
    if not d.is_dir():
        raise FormatError(f"not a directory: {d}")
    config_path = d / "config.json"
    if not config_path.is_file():
        raise FormatError("directory has no config.json; not a Hugging Face model directory")
    if not _inside(config_path, d):
        raise FormatError("config.json resolves outside the model directory")
    config = read_json_limited(config_path)
    if not isinstance(config, dict):
        raise FormatError("config.json is not a JSON object")

    errors: list[str] = []
    warnings: list[str] = []
    unparsed: list[UnparsedFile] = []
    side_files: list[Path] = []
    for entry in sorted(d.iterdir()):
        if not entry.is_file():
            continue
        suf = entry.suffix.lower()
        if suf in PICKLE_SUFFIXES and entry.name != "training_args.bin":
            unparsed.append(UnparsedFile(entry, entry.stat().st_size, "pickle-based PyTorch weights are never opened (unpickling can execute code)"))
        elif suf in OTHER_WEIGHT_SUFFIXES:
            unparsed.append(UnparsedFile(entry, entry.stat().st_size, f"{OTHER_WEIGHT_SUFFIXES[suf]} files are not parsed in a Hugging Face directory scan"))
        elif entry.name in SIDE_FILES:
            side_files.append(entry)

    index_path = d / "model.safetensors.index.json"
    shard_paths: list[Path]
    weight_map: dict[str, str] | None = None
    if index_path.is_file():
        index = read_json_limited(index_path, 64 * 1024 * 1024)
        wm = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(wm, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in wm.items()):
            raise FormatError("model.safetensors.index.json has no valid weight_map")
        weight_map = wm
        names = sorted(set(wm.values()))
        if len(names) > MAX_SHARDS:
            raise FormatError(f"index references {len(names)} shards (limit {MAX_SHARDS})")
        shard_paths = [_safe_child(d, n) for n in names]
        missing = [p.name for p in shard_paths if not p.is_file()]
        if missing:
            errors.append(f"index references missing shard(s): {', '.join(missing[:10])}")
            shard_paths = [p for p in shard_paths if p.is_file()]
    else:
        index_path = None
        shard_paths = sorted(p for p in d.glob("*.safetensors") if p.is_file() and _inside(p, d))
        if len(shard_paths) > MAX_SHARDS:
            raise FormatError(f"{len(shard_paths)} safetensors files (limit {MAX_SHARDS})")

    if not shard_paths:
        if unparsed:
            raise UnsupportedFormatError(
                "no safetensors weights found; this directory only contains formats Model Surgery will not open: "
                + ", ".join(sorted({u.path.suffix for u in unparsed}))
            )
        raise FormatError("no safetensors weight files found in the directory")

    shards: list[SafetensorsFile] = []
    seen: dict[str, str] = {}
    for sp in shard_paths:
        try:
            sf = read_safetensors(sp)
        except FormatError as e:
            errors.append(f"{sp.name}: {e}")
            continue
        errors.extend(f"{sp.name}: {m}" for m in sf.errors)
        warnings.extend(f"{sp.name}: {m}" for m in sf.warnings[:10])
        for t in sf.tensors:
            if t.name in seen:
                errors.append(f"tensor '{t.name}' appears in both {seen[t.name]} and {sp.name}")
            seen[t.name] = sp.name
        shards.append(sf)

    if weight_map is not None:
        for tname, fname in weight_map.items():
            actual = seen.get(tname)
            if actual is None:
                errors.append(f"index lists tensor '{tname}' but it is not in any shard")
            elif actual != fname:
                errors.append(f"index maps '{tname}' to {fname} but it is stored in {actual}")
        extra = set(seen) - set(weight_map)
        if extra:
            warnings.append(f"{len(extra)} tensor(s) present in shards but missing from the index")

    if len(errors) > 200:
        errors = errors[:200] + [f"... {len(errors) - 200} more"]
    return HFModelDir(d, config, config_path, index_path, shards, side_files, unparsed, errors, warnings)

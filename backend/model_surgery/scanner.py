"""Turn a local path into a ScanResult.

Detection is by content (magic bytes) first and file name second. Nothing in
this module executes, imports or deserialises model content.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .architecture import build_architecture, classify
from .config import Settings
from .formats import FormatError, UnsupportedFormatError
from .formats.ggml_types import GGML_TYPES_BY_NAME, UNQUANTIZED_FTYPES, ftype_label
from .formats.gguf import GGUFFile, is_gguf, read_gguf
from .formats.hf import PICKLE_SUFFIXES, HFModelDir, looks_like_hf_dir, read_hf_dir
from .formats.safetensors_reader import read_safetensors
from .schemas import (
    DTypeRow,
    Fact,
    FormatInfo,
    Integrity,
    MetadataEntry,
    QuantizationSummary,
    ScanResult,
    SourceFile,
    TensorRecord,
)

_SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
_PACKED_QUANT_METHODS = {"gptq", "awq", "bitsandbytes", "hqq", "aqlm", "eetq", "exl2", "quanto", "vptq", "higgs", "spqr"}
_UNSUPPORTED_SUFFIXES = {
    ".onnx": "ONNX models are not supported yet",
    ".tflite": "TensorFlow Lite models are not supported",
    ".h5": "Keras/HDF5 weights are not supported",
    ".msgpack": "Flax msgpack weights are not supported",
    ".engine": "TensorRT engines are not supported",
    ".mlmodel": "Core ML models are not supported",
}


class ScanError(ValueError):
    """User-facing scan failure (bad path, disallowed location, unsupported input)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(v: Any) -> Any:
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return str(v)
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return v


# --------------------------------------------------------------------------- facts


class _Facts:
    def __init__(self) -> None:
        self.items: dict[str, Fact] = {}

    def add(self, key: str, label: str, value: Any, prov: str, source: str, override: bool = False) -> None:
        if value is None or (key in self.items and not override):
            return
        if isinstance(value, float) and value.is_integer() and key not in ("rope_theta", "norm_eps"):
            value = int(value)
        self.items[key] = Fact(key=key, label=label, value=value, provenance=prov, source=source)  # type: ignore[arg-type]

    def get(self, key: str) -> Any:
        f = self.items.get(key)
        return None if f is None else f.value

    def ordered(self) -> list[Fact]:
        order = [
            "architecture", "model_class", "n_params", "n_params_logical", "n_layers", "hidden_size", "n_heads",
            "n_kv_heads", "attention_type", "head_dim", "head_dim_v", "ffn_size", "vocab_size", "context_length",
            "n_experts", "n_experts_used", "sliding_window", "rope_theta", "norm_eps", "tied_embeddings",
            "ssm_state_size", "torch_dtype",
        ]
        return sorted(self.items.values(), key=lambda f: order.index(f.key) if f.key in order else len(order))


def _scalarize(value: Any, key: str, notes: list[str]) -> Any:
    """Per-layer arrays (e.g. head counts in OpenELM/Gemma3n) are summarised by their maximum."""
    if isinstance(value, list):
        nums = [v for v in value if isinstance(v, (int, float))]
        if not nums:
            return None
        if len(set(nums)) > 1:
            notes.append(f"{key} varies per layer ({min(nums)}..{max(nums)}); the maximum is used for estimates")
        return max(nums)
    return value


def _facts_from_gguf(g: GGUFFile, facts: _Facts, notes: list[str]) -> str | None:
    arch = g.get("general.architecture")
    if not isinstance(arch, str):
        notes.append("GGUF has no general.architecture key")
        return None
    facts.add("architecture", "Architecture", arch, "verified", "general.architecture")

    def kv(label: str, key: str, suffix: str) -> None:
        full = f"{arch}.{suffix}"
        if full in g.metadata:
            facts.add(key, label, _scalarize(g.get(full), full, notes), "verified", full)

    kv("Layers", "n_layers", "block_count")
    kv("Hidden size", "hidden_size", "embedding_length")
    kv("Attention heads", "n_heads", "attention.head_count")
    kv("KV heads", "n_kv_heads", "attention.head_count_kv")
    kv("Head dim (K)", "head_dim", "attention.key_length")
    kv("Head dim (V)", "head_dim_v", "attention.value_length")
    kv("FFN size", "ffn_size", "feed_forward_length")
    kv("Training context length", "context_length", "context_length")
    kv("Vocabulary size", "vocab_size", "vocab_size")
    kv("Experts", "n_experts", "expert_count")
    kv("Experts used per token", "n_experts_used", "expert_used_count")
    kv("Sliding window", "sliding_window", "attention.sliding_window")
    kv("RoPE base", "rope_theta", "rope.freq_base")
    kv("Norm epsilon", "norm_eps", "attention.layer_norm_rms_epsilon")
    kv("Norm epsilon", "norm_eps", "attention.layer_norm_epsilon")
    kv("SSM state size", "ssm_state_size", "ssm.state_size")
    n_tok = g.get_array_len("tokenizer.ggml.tokens")
    if n_tok is not None:
        facts.add("vocab_size", "Vocabulary size", n_tok, "verified", "length of tokenizer.ggml.tokens")
    return arch


_HF_KEYS: list[tuple[str, str, tuple[str, ...]]] = [
    ("n_layers", "Layers", ("num_hidden_layers", "n_layer", "num_layers", "n_layers")),
    ("hidden_size", "Hidden size", ("hidden_size", "n_embd", "d_model", "dim")),
    ("n_heads", "Attention heads", ("num_attention_heads", "n_head", "num_heads", "n_heads")),
    ("n_kv_heads", "KV heads", ("num_key_value_heads", "num_kv_heads", "n_head_kv", "n_kv_heads", "multi_query_group_num")),
    ("head_dim", "Head dim (K)", ("head_dim", "kv_channels")),
    ("ffn_size", "FFN size", ("intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size", "hidden_dim")),
    ("vocab_size", "Vocabulary size", ("vocab_size", "padded_vocab_size")),
    ("context_length", "Training context length", ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length")),
    ("n_experts", "Experts", ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")),
    ("n_experts_used", "Experts used per token", ("num_experts_per_tok", "moe_top_k", "top_k_experts")),
    ("sliding_window", "Sliding window", ("sliding_window",)),
    ("rope_theta", "RoPE base", ("rope_theta", "rotary_emb_base")),
    ("norm_eps", "Norm epsilon", ("rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_epsilon")),
    ("ssm_state_size", "SSM state size", ("state_size", "d_state", "ssm_state_size")),
]


def _facts_from_hf(config: dict[str, Any], facts: _Facts, notes: list[str]) -> str | None:
    lm = config
    lm_src = "config.json"
    for sub in ("text_config", "llm_config", "language_config"):
        if isinstance(config.get(sub), dict):
            lm = {**config, **config[sub]}
            lm_src = f"config.json:{sub}"
            notes.append(f"language-model hyperparameters taken from '{sub}' (multimodal config)")
            break

    model_type = config.get("model_type")
    if isinstance(model_type, str):
        facts.add("architecture", "Architecture", model_type, "verified", "config.json:model_type")
    archs = config.get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        facts.add("model_class", "Model class", archs[0], "verified", "config.json:architectures[0]")

    for key, label, names in _HF_KEYS:
        for n in names:
            v = lm.get(n)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                facts.add(key, label, v, "verified", f"{lm_src}:{n}")
                break
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                facts.add(key, label, _scalarize(v, n, notes), "verified", f"{lm_src}:{n}")
                break
    if facts.get("n_kv_heads") is None and lm.get("multi_query") is True:
        facts.add("n_kv_heads", "KV heads", 1, "verified", f"{lm_src}:multi_query")
    tie = lm.get("tie_word_embeddings")
    if isinstance(tie, bool):
        facts.add("tied_embeddings", "Tied embeddings", tie, "verified", f"{lm_src}:tie_word_embeddings")
    dt = lm.get("torch_dtype") or lm.get("dtype")
    if isinstance(dt, str):
        facts.add("torch_dtype", "Declared dtype", dt, "verified", f"{lm_src}:torch_dtype")
    return model_type if isinstance(model_type, str) else None


def _shape_dims(t: TensorRecord) -> tuple[int, int] | None:
    """(in_features, out_features) for a 2-D weight, regardless of storage order."""
    if len(t.shape) < 2:
        return None
    if t.shape_order == "ggml":
        return t.shape[0], t.shape[1]
    return t.shape[-1], t.shape[-2]


def _complete_facts(facts: _Facts, tensors: list[TensorRecord], notes: list[str], packed: bool) -> None:
    n_params = sum(t.n_elements for t in tensors)
    facts.add("n_params", "Parameters (tensor elements)", n_params, "derived", "sum of tensor element counts")

    by_role: dict[tuple[str, str | None], list[TensorRecord]] = defaultdict(list)
    for t in tensors:
        by_role[(t.component, t.role)].append(t)

    emb = next((t for t in by_role.get(("embedding", "token"), []) if len(t.shape) == 2), None)
    if emb and not packed:
        d = _shape_dims(emb)
        assert d
        hidden, vocab = d
        if facts.get("hidden_size") is None:
            facts.add("hidden_size", "Hidden size", hidden, "inferred", f"shape of {emb.name}")
        elif facts.get("hidden_size") != hidden:
            notes.append(f"metadata hidden size {facts.get('hidden_size')} differs from {emb.name} width {hidden}")
        if facts.get("vocab_size") is None:
            facts.add("vocab_size", "Vocabulary size", vocab, "inferred", f"shape of {emb.name}")

    layer_idx = {t.layer for t in tensors if t.layer is not None and t.stack is not None}
    stacks = Counter(t.stack for t in tensors if t.stack is not None)
    if stacks:
        primary = stacks.most_common(1)[0][0]
        n_l = len({t.layer for t in tensors if t.stack == primary})
        if facts.get("n_layers") is None:
            facts.add("n_layers", "Layers", n_l, "inferred", f"distinct layer indices under '{primary}'")
        elif facts.get("n_layers") != n_l:
            notes.append(f"metadata declares {facts.get('n_layers')} layers but tensors under '{primary}' cover {n_l}")
    del layer_idx

    n_heads, hidden = facts.get("n_heads"), facts.get("hidden_size")
    if facts.get("head_dim") is None and isinstance(n_heads, int) and isinstance(hidden, int) and n_heads > 0:
        facts.add("head_dim", "Head dim (K)", hidden // n_heads, "derived", "hidden_size / n_heads")
    head_dim = facts.get("head_dim")
    if facts.get("head_dim_v") is None and head_dim is not None:
        facts.add("head_dim_v", "Head dim (V)", head_dim, "derived", "assumed equal to K head dim")

    if facts.get("n_kv_heads") is None and isinstance(head_dim, int) and head_dim > 0 and not packed:
        k = next((t for t in by_role.get(("attention", "k"), []) if len(t.shape) == 2 and t.layer == 0), None)
        if k:
            d = _shape_dims(k)
            assert d
            if d[1] % head_dim == 0:
                facts.add("n_kv_heads", "KV heads", d[1] // head_dim, "inferred", f"{k.name} output width / head_dim")
        elif isinstance(n_heads, int):
            facts.add("n_kv_heads", "KV heads", n_heads, "inferred", "no K projection found; assumed equal to attention heads")

    if facts.get("ffn_size") is None and not packed:
        up = next((t for t in by_role.get(("ffn", "up"), []) + by_role.get(("ffn", "gate"), []) if len(t.shape) == 2), None)
        if up:
            facts.add("ffn_size", "FFN size", _shape_dims(up)[1], "inferred", f"shape of {up.name}")  # type: ignore[index]

    nk, nh = facts.get("n_kv_heads"), facts.get("n_heads")
    if isinstance(nk, int) and isinstance(nh, int) and nk > 0:
        kind = "MHA" if nk == nh else ("MQA" if nk == 1 else f"GQA ({nh // nk if nh % nk == 0 else nh / nk:g} query heads per KV head)")
        facts.add("attention_type", "Attention type", kind, "derived", "n_heads vs n_kv_heads")

    if facts.get("tied_embeddings") is None:
        has_out = bool(by_role.get(("output", "lm_head")))
        if emb is not None:
            facts.add(
                "tied_embeddings",
                "Tied embeddings",
                not has_out,
                "inferred",
                "no separate output/lm_head tensor" if not has_out else "separate output/lm_head tensor present",
            )

    if packed:
        est = estimate_params_from_dims(facts)
        if est:
            facts.add("n_params_logical", "Parameters (logical, estimated)", est, "inferred", "computed from config dimensions")
            notes.append("checkpoint stores packed quantized weights: tensor element count is not the logical parameter count")


def estimate_params_from_dims(facts: _Facts | dict[str, Any]) -> int | None:
    g = facts.get if isinstance(facts, _Facts) else facts.get
    L, H, V = g("n_layers"), g("hidden_size"), g("vocab_size")
    if not all(isinstance(x, int) and x > 0 for x in (L, H, V)):
        return None
    nh = g("n_heads") or 1
    nkv = g("n_kv_heads") or nh
    hd = g("head_dim") or H // nh
    F = g("ffn_size") or 4 * H
    E = g("n_experts") or 1
    attn = H * nh * hd * 2 + H * nkv * hd * 2
    ffn = 3 * H * F * E
    emb = V * H * (1 if g("tied_embeddings") is not False else 2)
    return L * (attn + ffn + 2 * H) + emb + H


# --------------------------------------------------------------------------- summaries


def _dtype_breakdown(tensors: list[TensorRecord]) -> list[DTypeRow]:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for t in tensors:
        a = agg[t.dtype]
        a[0] += 1
        a[1] += t.n_elements
        a[2] += t.n_bytes or 0
    rows = [DTypeRow(dtype=k, n_tensors=v[0], n_elements=v[1], n_bytes=v[2]) for k, v in agg.items()]
    return sorted(rows, key=lambda r: r.n_bytes, reverse=True)


def _effective_bpw(tensors: list[TensorRecord]) -> float | None:
    n = sum(t.n_elements for t in tensors if t.n_bytes is not None)
    b = sum(t.n_bytes for t in tensors if t.n_bytes is not None)
    return round(b * 8 / n, 3) if n else None


def _dominant_matrix_dtype(tensors: list[TensorRecord]) -> str | None:
    c: Counter[str] = Counter()
    for t in tensors:
        if len(t.shape) >= 2:
            c[t.dtype] += t.n_elements
    return c.most_common(1)[0][0] if c else None


def _make_tensor(name: str, shape: list[int], order: str, dtype: str, n_el: int, n_bytes: int | None, offset: int | None, fi: int) -> TensorRecord:
    c = classify(name)
    row = (shape[0] if order == "ggml" else shape[-1]) if shape else 1
    return TensorRecord(
        name=name,
        shape=shape,
        shape_order=order,  # type: ignore[arg-type]
        row_length=row,
        dtype=dtype,
        n_elements=n_el,
        n_bytes=n_bytes,
        offset=offset,
        file_index=fi,
        stack=c.stack,
        layer=c.layer,
        component=c.component,
        role=c.role,
    )


# --------------------------------------------------------------------------- GGUF


def _gguf_shards(first: Path, g: GGUFFile, errors: list[str]) -> list[Path]:
    count = g.get("split.count")
    if not isinstance(count, int) or count <= 1:
        return [first]
    m = _SPLIT_RE.match(first.name)
    if not m:
        errors.append(f"split.count={count} but the file name does not follow the '-NNNNN-of-NNNNN.gguf' pattern")
        return [first]
    if int(m.group("no")) != 1:
        errors.append("this is not the first shard of a split GGUF; select the -00001-of-NNNNN file")
    paths = []
    for i in range(1, count + 1):
        p = first.with_name(f"{m.group('base')}-{i:05d}-of-{count:05d}.gguf")
        if p.is_file():
            paths.append(p)
        else:
            errors.append(f"missing split shard {p.name}")
    return paths or [first]


def _scan_gguf(path: Path) -> dict[str, Any]:
    g0 = read_gguf(path)
    errors = list(g0.errors)
    warnings = list(g0.warnings)
    notes: list[str] = []

    shard_paths = _gguf_shards(path, g0, errors)
    parsed: list[GGUFFile] = []
    for p in shard_paths:
        if p == path:
            parsed.append(g0)
            continue
        try:
            gs = read_gguf(p)
        except FormatError as e:
            errors.append(f"{p.name}: {e}")
            continue
        errors.extend(f"{p.name}: {m}" for m in gs.errors)
        warnings.extend(f"{p.name}: {m}" for m in gs.warnings)
        parsed.append(gs)

    tensors: list[TensorRecord] = []
    seen: set[str] = set()
    for fi, g in enumerate(parsed):
        for t in g.tensors:
            if t.name in seen:
                errors.append(f"tensor '{t.name}' appears in more than one shard")
            seen.add(t.name)
            dtype = t.ggml_type.name if t.ggml_type else f"unknown({t.type_id})"
            tensors.append(_make_tensor(t.name, t.shape, "ggml", dtype, t.n_elements, t.n_bytes, g.data_offset + t.offset, fi))
    expected = g0.get("split.tensors.count")
    if isinstance(expected, int) and expected != len(tensors):
        errors.append(f"split.tensors.count={expected} but {len(tensors)} tensors were found across shards")

    facts = _Facts()
    arch = _facts_from_gguf(g0, facts, notes)

    ft = g0.get("general.file_type")
    is_q = any(GGML_TYPES_BY_NAME.get(t.dtype, None) is not None and GGML_TYPES_BY_NAME[t.dtype].quantized for t in tensors)
    if isinstance(ft, int):
        label, label_prov = ftype_label(ft), "verified"
    else:
        label, label_prov = _dominant_matrix_dtype(tensors), "inferred"
        if label:
            notes.append("general.file_type missing; quantization label inferred from the dominant matrix dtype")
    qnotes = []
    qv = g0.get("general.quantization_version")
    if qv is not None:
        qnotes.append(f"general.quantization_version = {qv}")
    if label and label.split(" ")[0] not in UNQUANTIZED_FTYPES and is_q:
        qnotes.append("llama.cpp quantization presets mix tensor types; see the dtype breakdown for the actual per-tensor types")

    files = [SourceFile(path=str(g.path), size_bytes=g.file_size, role="gguf" if len(parsed) == 1 else f"gguf shard {i + 1}/{len(parsed)}", parsed=True) for i, g in enumerate(parsed)]
    metadata = [MetadataEntry(key=k, type=v.type, value=_json_safe(v.value), length=v.length, truncated=v.truncated) for k, v in g0.metadata.items()]
    name = g0.get("general.name") or path.stem
    return dict(
        kind="gguf",
        name=str(name),
        files=files,
        format=FormatInfo(
            name="GGUF",
            version=f"v{g0.version}",
            details={
                "byte_order": g0.byte_order,
                "alignment": g0.alignment,
                "tensor_count": len(tensors),
                "metadata_count": len(g0.metadata),
                "data_offset": g0.data_offset,
                "shards": len(parsed),
            },
        ),
        errors=errors,
        warnings=warnings,
        notes=notes,
        facts=facts,
        family=arch,
        tensors=tensors,
        metadata=metadata,
        quant=QuantizationSummary(
            label=label,
            label_provenance=label_prov if label else None,  # type: ignore[arg-type]
            is_quantized=is_q,
            effective_bits_per_weight=_effective_bpw(tensors),
            packed_checkpoint=False,
            notes=qnotes,
        ),
        packed=False,
    )


# --------------------------------------------------------------------------- safetensors / HF


def _st_tensors(shards, offset_base: bool = True) -> list[TensorRecord]:
    out = []
    for fi, sf in enumerate(shards):
        for t in sf.tensors:
            out.append(_make_tensor(t.name, t.shape, "row_major", t.dtype, t.n_elements, t.n_bytes, sf.data_offset + t.begin, fi))
    return out


def _scan_hf(d: Path) -> dict[str, Any]:
    hf: HFModelDir = read_hf_dir(d)
    notes: list[str] = []
    facts = _Facts()
    family = _facts_from_hf(hf.config, facts, notes)

    qc = hf.config.get("quantization_config")
    if not isinstance(qc, dict):
        tc = hf.config.get("text_config")
        qc = tc.get("quantization_config") if isinstance(tc, dict) and isinstance(tc.get("quantization_config"), dict) else None
    method = str(qc.get("quant_method", "unknown")).lower() if qc else None
    packed = method in _PACKED_QUANT_METHODS

    tensors = _st_tensors(hf.shards)
    _complete_facts(facts, tensors, notes, packed)

    qnotes: list[str] = []
    if qc:
        bits = qc.get("bits") or qc.get("w_bit") or ("4" if qc.get("load_in_4bit") else "8" if qc.get("load_in_8bit") else None)
        label = f"{method} {bits}-bit" if bits else method
        label_prov = "verified"
        qnotes.append(f"quantization_config.quant_method = {method}")
        if packed:
            qnotes.append("weights are stored packed; per-tensor dtypes describe storage containers, not logical precision")
    else:
        label, label_prov = _dominant_matrix_dtype(tensors), "derived"
    is_q = bool(qc) or any(t.dtype in ("I8", "U8", "F8_E4M3", "F8_E5M2", "F4") and len(t.shape) >= 2 for t in tensors)

    files: list[SourceFile] = [SourceFile(path=str(hf.config_path), size_bytes=hf.config_path.stat().st_size, role="config", parsed=True)]
    if hf.index_path:
        files.append(SourceFile(path=str(hf.index_path), size_bytes=hf.index_path.stat().st_size, role="safetensors index", parsed=True))
    shard_base = len(files)
    for t in tensors:  # make file_index point into `files`
        t.file_index += shard_base
    for sf in hf.shards:
        files.append(SourceFile(path=str(sf.path), size_bytes=sf.file_size, role="safetensors shard", parsed=True))
    for sp in hf.side_files:
        files.append(SourceFile(path=str(sp), size_bytes=sp.stat().st_size, role="tokenizer/config", parsed=False, note="present; not needed for inspection"))
    for u in hf.unparsed:
        files.append(SourceFile(path=str(u.path), size_bytes=u.size, role="unparsed weights", parsed=False, note=u.reason))

    metadata: list[MetadataEntry] = []
    for k, v in hf.config.items():
        metadata.append(MetadataEntry(key=f"config.{k}", type=type(v).__name__, value=_json_safe(v)))
    for sf in hf.shards[:1]:
        for k, v in sf.metadata.items():
            metadata.append(MetadataEntry(key=f"safetensors.__metadata__.{k}", type="str", value=v))

    warnings = list(hf.warnings)
    if hf.unparsed:
        warnings.append(f"{len(hf.unparsed)} file(s) in non-safetensors formats were listed but not opened")
    name = hf.config.get("_name_or_path") if isinstance(hf.config.get("_name_or_path"), str) and hf.config.get("_name_or_path") else d.name
    return dict(
        kind="hf_safetensors",
        name=str(name),
        files=files,
        format=FormatInfo(
            name="Hugging Face safetensors",
            version=None,
            details={
                "shards": len(hf.shards),
                "index_file": hf.index_path is not None,
                "tensor_count": len(tensors),
                "side_files": [p.name for p in hf.side_files],
            },
        ),
        errors=list(hf.errors),
        warnings=warnings,
        notes=notes,
        facts=facts,
        family=family,
        tensors=tensors,
        metadata=metadata,
        quant=QuantizationSummary(
            label=label,
            label_provenance=label_prov if label else None,  # type: ignore[arg-type]
            is_quantized=is_q,
            effective_bits_per_weight=None if packed else _effective_bpw(tensors),
            packed_checkpoint=packed,
            config=_json_safe(qc) if qc else None,
            notes=qnotes,
        ),
        packed=packed,
    )


def _scan_single_safetensors(p: Path) -> dict[str, Any]:
    sf = read_safetensors(p)
    tensors = _st_tensors([sf])
    facts = _Facts()
    notes = ["single safetensors file without config.json: architecture hyperparameters are inferred from tensor shapes only"]
    return dict(
        kind="safetensors",
        name=p.stem,
        files=[SourceFile(path=str(p), size_bytes=sf.file_size, role="safetensors", parsed=True)],
        format=FormatInfo(name="safetensors", version=None, details={"header_bytes": sf.header_size, "tensor_count": len(tensors)}),
        errors=list(sf.errors),
        warnings=list(sf.warnings),
        notes=notes,
        facts=facts,
        family=None,
        tensors=tensors,
        metadata=[MetadataEntry(key=f"__metadata__.{k}", type="str", value=v) for k, v in sf.metadata.items()],
        quant=QuantizationSummary(
            label=_dominant_matrix_dtype(tensors),
            label_provenance="derived",
            is_quantized=False,
            effective_bits_per_weight=_effective_bpw(tensors),
            packed_checkpoint=False,
        ),
        packed=False,
    )


# --------------------------------------------------------------------------- entry point


def resolve_input(raw: str, settings: Settings) -> Path:
    if not raw or "\x00" in raw or len(raw) > 4096:
        raise ScanError("invalid path")
    p = Path(raw.strip().strip('"')).expanduser()
    if not p.is_absolute():
        raise ScanError("please provide an absolute path")
    if not p.exists():
        raise ScanError(f"path does not exist: {p}")
    p = p.resolve()
    if not settings.is_allowed(p):
        raise ScanError("path is outside the configured allowed roots")
    if not (p.is_file() or p.is_dir()):
        raise ScanError("path is not a regular file or directory")
    return p


def scan_path(raw: str, settings: Settings) -> ScanResult:
    p = resolve_input(raw, settings)
    redirect_note: str | None = None

    if p.is_file() and p.name == "config.json":
        p, redirect_note = p.parent, "config.json selected: scanned the containing Hugging Face model directory"
    elif p.is_file() and p.suffix.lower() == ".safetensors" and looks_like_hf_dir(p.parent):
        p, redirect_note = p.parent, "safetensors shard selected inside a Hugging Face model directory: scanned the whole directory"

    try:
        if p.is_dir():
            if looks_like_hf_dir(p):
                data = _scan_hf(p)
            elif any(p.glob("*.gguf")):
                raise ScanError("this directory contains GGUF files; select a specific .gguf file")
            elif any(p.glob("*.safetensors")):
                raise ScanError("directory has safetensors files but no config.json; select a .safetensors file directly")
            else:
                raise ScanError("directory is not a recognised model directory (no config.json + safetensors)")
        elif is_gguf(p):
            data = _scan_gguf(p)
        else:
            suf = p.suffix.lower()
            if suf == ".safetensors":
                data = _scan_single_safetensors(p)
            elif suf in PICKLE_SUFFIXES:
                raise UnsupportedFormatError(
                    f"{suf} files are Python pickles. Model Surgery never opens pickle files because loading them can execute "
                    "arbitrary code. Convert the model to safetensors with trusted tooling first."
                )
            elif suf in _UNSUPPORTED_SUFFIXES:
                raise UnsupportedFormatError(_UNSUPPORTED_SUFFIXES[suf])
            elif suf == ".gguf":
                raise FormatError("file has a .gguf extension but does not start with the GGUF magic bytes")
            else:
                raise UnsupportedFormatError("unrecognised file type (supported: .gguf, .safetensors, Hugging Face model directory)")
    except (FormatError, UnsupportedFormatError, OSError) as e:
        raise ScanError(str(e)) from e

    facts: _Facts = data["facts"]
    tensors: list[TensorRecord] = data["tensors"]
    notes: list[str] = data["notes"]
    if data["kind"] == "gguf" or data["kind"] == "safetensors":
        _complete_facts(facts, tensors, notes, data["packed"])
    if redirect_note:
        notes.insert(0, redirect_note)

    family = data["family"]
    arch = build_architecture(tensors, family, "verified" if family else None)
    arch.notes = notes + arch.notes

    files: list[SourceFile] = data["files"]
    return ScanResult(
        id=uuid.uuid4().hex,
        scanned_at=_now(),
        app_version=__version__,
        kind=data["kind"],
        name=data["name"],
        path=str(p),
        files=files,
        total_size_bytes=sum(f.size_bytes for f in files),
        format=data["format"],
        integrity=Integrity(valid=not data["errors"], errors=data["errors"], warnings=data["warnings"]),
        hyperparameters=facts.ordered(),
        quantization=data["quant"],
        dtype_breakdown=_dtype_breakdown(tensors),
        architecture=arch,
        metadata=data["metadata"],
        tensors=tensors,
    )

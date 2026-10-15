"""Writers for small synthetic GGUF / safetensors files, including corrupt variants."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from model_surgery.formats.ggml_types import GGML_TYPES, GGML_TYPES_BY_NAME

T = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "str": 8, "arr": 9, "u64": 10, "i64": 11, "f64": 12}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _s(bo: str, s: str) -> bytes:
    b = s.encode()
    return struct.pack(bo + "Q", len(b)) + b


def _val(bo: str, t: int, v: Any) -> bytes:
    if t == 8:
        return _s(bo, v)
    if t == 9:
        et, items = v
        out = struct.pack(bo + "IQ", et, len(items))
        for it in items:
            out += _val(bo, et, it)
        return out
    return struct.pack(bo + FMT[t], v)


def tensor_size(type_name: str, shape: list[int]) -> int:
    g = GGML_TYPES_BY_NAME[type_name]
    n = 1
    for d in shape:
        n *= d
    return n // g.block_size * g.type_size


def gguf_bytes(
    kv: list[tuple[str, int, Any]],
    tensors: list[tuple[str, list[int], str]],
    *,
    version: int = 3,
    big_endian: bool = False,
    alignment: int = 32,
    offsets: dict[str, int] | None = None,
    truncate_data: int = 0,
    type_ids: dict[str, int] | None = None,
) -> bytes:
    bo = ">" if big_endian else "<"
    head = b"GGUF" + struct.pack(bo + "I", version) + struct.pack(bo + "QQ", len(tensors), len(kv))
    for key, t, v in kv:
        head += _s(bo, key) + struct.pack(bo + "I", t) + _val(bo, t, v)
    cursor = 0
    infos = b""
    sizes = []
    for name, shape, tname in tensors:
        size = tensor_size(tname, shape)
        off = (offsets or {}).get(name, cursor)
        tid = (type_ids or {}).get(name, GGML_TYPES_BY_NAME[tname].id)
        infos += _s(bo, name) + struct.pack(bo + "I", len(shape)) + b"".join(struct.pack(bo + "Q", d) for d in shape)
        infos += struct.pack(bo + "IQ", tid, off)
        sizes.append(size)
        cursor += size + (-size % alignment)
    data = head + infos
    data += b"\0" * (-len(data) % alignment)
    data += b"\0" * cursor
    return data[: len(data) - truncate_data] if truncate_data else data


def llama_like_gguf(
    path: Path,
    n_layers: int = 2,
    hidden: int = 64,
    n_heads: int = 4,
    n_kv: int = 2,
    ffn: int = 128,
    vocab: int = 256,
    wtype: str = "F16",
    ftype: int = 1,
    with_metadata: bool = True,
    tied: bool = False,
    **kw: Any,
) -> Path:
    head_dim = hidden // n_heads
    kv = [("general.architecture", T["str"], "llama"), ("general.name", T["str"], "tiny-llama")]
    if with_metadata:
        kv += [
            ("llama.block_count", T["u32"], n_layers),
            ("llama.embedding_length", T["u32"], hidden),
            ("llama.feed_forward_length", T["u32"], ffn),
            ("llama.attention.head_count", T["u32"], n_heads),
            ("llama.attention.head_count_kv", T["u32"], n_kv),
            ("llama.context_length", T["u32"], 2048),
            ("llama.rope.freq_base", T["f32"], 10000.0),
            ("general.file_type", T["u32"], ftype),
            ("tokenizer.ggml.tokens", T["arr"], (T["str"], [f"t{i}" for i in range(vocab)])),
        ]
    tensors: list[tuple[str, list[int], str]] = [("token_embd.weight", [hidden, vocab], wtype)]
    for i in range(n_layers):
        tensors += [
            (f"blk.{i}.attn_norm.weight", [hidden], "F32"),
            (f"blk.{i}.attn_q.weight", [hidden, hidden], wtype),
            (f"blk.{i}.attn_k.weight", [hidden, n_kv * head_dim], wtype),
            (f"blk.{i}.attn_v.weight", [hidden, n_kv * head_dim], wtype),
            (f"blk.{i}.attn_output.weight", [hidden, hidden], wtype),
            (f"blk.{i}.ffn_norm.weight", [hidden], "F32"),
            (f"blk.{i}.ffn_gate.weight", [hidden, ffn], wtype),
            (f"blk.{i}.ffn_up.weight", [hidden, ffn], wtype),
            (f"blk.{i}.ffn_down.weight", [ffn, hidden], wtype),
        ]
    tensors.append(("output_norm.weight", [hidden], "F32"))
    if not tied:
        tensors.append(("output.weight", [hidden, vocab], wtype))
    path.write_bytes(gguf_bytes(kv, tensors, **kw))
    return path


def safetensors_bytes(tensors: dict[str, tuple[str, list[int]]], metadata: dict[str, str] | None = None, *, header_override: Any = None, raw_header: bytes | None = None, data_pad: int = 0) -> bytes:
    from model_surgery.formats.safetensors_reader import DTYPE_BITS

    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    cursor = 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        size = (n * DTYPE_BITS[dtype] + 7) // 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    if header_override is not None:
        header = header_override
    hb = raw_header if raw_header is not None else json.dumps(header).encode()
    return struct.pack("<Q", len(hb)) + hb + b"\0" * (cursor + data_pad)


def hf_llama_dir(d: Path, n_layers: int = 2, hidden: int = 64, n_heads: int = 4, n_kv: int = 2, ffn: int = 128, vocab: int = 256, shards: int = 2, config_extra: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    head_dim = hidden // n_heads
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "vocab_size": vocab,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    cfg.update(config_extra or {})
    (d / "config.json").write_text(json.dumps(cfg))
    tensors: dict[str, tuple[str, list[int]]] = {"model.embed_tokens.weight": ("BF16", [vocab, hidden])}
    for i in range(n_layers):
        p = f"model.layers.{i}"
        tensors.update(
            {
                f"{p}.input_layernorm.weight": ("BF16", [hidden]),
                f"{p}.self_attn.q_proj.weight": ("BF16", [hidden, hidden]),
                f"{p}.self_attn.k_proj.weight": ("BF16", [n_kv * head_dim, hidden]),
                f"{p}.self_attn.v_proj.weight": ("BF16", [n_kv * head_dim, hidden]),
                f"{p}.self_attn.o_proj.weight": ("BF16", [hidden, hidden]),
                f"{p}.post_attention_layernorm.weight": ("BF16", [hidden]),
                f"{p}.mlp.gate_proj.weight": ("BF16", [ffn, hidden]),
                f"{p}.mlp.up_proj.weight": ("BF16", [ffn, hidden]),
                f"{p}.mlp.down_proj.weight": ("BF16", [hidden, ffn]),
            }
        )
    tensors["model.norm.weight"] = ("BF16", [hidden])
    tensors["lm_head.weight"] = ("BF16", [vocab, hidden])
    names = list(tensors)
    per = -(-len(names) // shards)
    weight_map = {}
    for s in range(shards):
        chunk = names[s * per : (s + 1) * per]
        fname = f"model-{s + 1:05d}-of-{shards:05d}.safetensors" if shards > 1 else "model.safetensors"
        (d / fname).write_bytes(safetensors_bytes({n: tensors[n] for n in chunk}, {"format": "pt"}))
        weight_map.update({n: fname for n in chunk})
    if shards > 1:
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (d / "tokenizer.json").write_text("{}")
    return d


__all__ = ["T", "gguf_bytes", "llama_like_gguf", "safetensors_bytes", "hf_llama_dir", "tensor_size", "GGML_TYPES"]

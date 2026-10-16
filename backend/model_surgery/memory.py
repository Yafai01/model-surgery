"""Memory estimates. Nothing here is a measurement.

Weight-storage estimates replay an approximation of llama.cpp's per-tensor
type selection (llama-quant.cpp) over the model's real tensor list, so block
scale overhead, fallback types for awkward row lengths, and higher-precision
output/attn_v/ffn_down tensors are accounted for. Inference memory adds a KV
cache computed from verified hyperparameters plus heuristic compute-buffer and
runtime terms; each term carries its formula and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from .devices import DeviceProfile, get_profile
from .formats.ggml_types import GGML_TYPES_BY_NAME
from .schemas import Provenance, ScanResult, TensorRecord

KV_BYTES_PER_ELEMENT = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

# llama.cpp file-type preset -> dominant ggml tensor type.
FTYPE_MAIN_TYPE = {
    "F32": "F32", "F16": "F16", "BF16": "BF16", "Q8_0": "Q8_0",
    "Q4_0": "Q4_0", "Q4_1": "Q4_1", "Q5_0": "Q5_0", "Q5_1": "Q5_1",
    "Q2_K": "Q2_K", "Q2_K_S": "Q2_K", "Q3_K": "Q3_K", "Q3_K_S": "Q3_K", "Q3_K_M": "Q3_K", "Q3_K_L": "Q3_K",
    "Q4_K": "Q4_K", "Q4_K_S": "Q4_K", "Q4_K_M": "Q4_K", "Q5_K": "Q5_K", "Q5_K_S": "Q5_K", "Q5_K_M": "Q5_K", "Q6_K": "Q6_K",
    "IQ1_S": "IQ1_S", "IQ1_M": "IQ1_M", "IQ2_XXS": "IQ2_XXS", "IQ2_XS": "IQ2_XS", "IQ2_S": "IQ2_S", "IQ2_M": "IQ2_S",
    "IQ3_XXS": "IQ3_XXS", "IQ3_XS": "IQ3_S", "IQ3_S": "IQ3_S", "IQ3_M": "IQ3_S", "IQ4_NL": "IQ4_NL", "IQ4_XS": "IQ4_XS",
    "TQ1_0": "TQ1_0", "TQ2_0": "TQ2_0", "MXFP4_MOE": "MXFP4",
}
_FLOAT_FTYPES = {"F32", "F16", "BF16"}
_FALLBACK = {
    "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0",
    "Q2_K": "IQ4_NL", "Q3_K": "IQ4_NL", "IQ2_XXS": "IQ4_NL", "IQ2_XS": "IQ4_NL", "IQ2_S": "IQ4_NL",
    "IQ3_XXS": "IQ4_NL", "IQ3_S": "IQ4_NL", "IQ1_S": "IQ4_NL", "IQ1_M": "IQ4_NL", "IQ4_XS": "IQ4_NL",
    "TQ1_0": "IQ4_NL", "TQ2_0": "IQ4_NL",
}


@dataclass(frozen=True)
class Scheme:
    id: str
    label: str
    ftype: str
    family: str  # FP32 | FP16 | INT8 | 4-bit


SCHEMES = [
    Scheme("fp32", "FP32", "F32", "FP32"),
    Scheme("fp16", "FP16 / BF16", "F16", "FP16"),
    Scheme("q8_0", "INT8 · Q8_0", "Q8_0", "INT8"),
    Scheme("q4_k_m", "4-bit · Q4_K_M", "Q4_K_M", "4-bit"),
    Scheme("q4_0", "4-bit · Q4_0", "Q4_0", "4-bit"),
]
SCHEME_BY_ID = {s.id: s for s in SCHEMES}


def _use_more_bits(i: int, n: int) -> bool:
    # llama-quant.cpp: first/last eighth of layers plus every third layer in between.
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


def _fits(type_name: str, row: int) -> bool:
    return row % GGML_TYPES_BY_NAME[type_name].block_size == 0


def _with_fallback(type_name: str, row: int) -> str:
    t = type_name
    for _ in range(3):
        if _fits(t, row):
            return t
        t = _FALLBACK.get(t, "Q8_0" if t not in ("Q8_0", "F16") else "F16")
    return "F16"


def target_type_for(t: TensorRecord, ftype: str, seq: tuple[int, int], has_output: bool) -> str:
    """Approximate llama.cpp's choice of ggml type for one tensor under a preset.

    ``seq`` is (i, n): this tensor's position among same-role tensors (attn_v or
    ffn_down) in file order, which is what llama.cpp's counters track.
    """
    if len(t.shape) < 2 or not t.name.endswith("weight"):
        return "F32"
    if t.component == "router":  # llama.cpp never quantizes ffn_gate_inp
        return "F32"
    if t.role == "position" or "conv1d" in t.name:
        return "F32" if ftype == "F32" else "F16"
    main = FTYPE_MAIN_TYPE.get(ftype, ftype)
    if ftype in _FLOAT_FTYPES:
        return main
    is_output = t.component == "output" or (not has_output and t.component == "embedding" and t.role == "token")
    if ftype == "MXFP4_MOE":
        # Only expert tensors become MXFP4; everything else is kept at Q8_0.
        return "MXFP4" if t.component == "experts" and _fits("MXFP4", t.row_length) else _with_fallback("Q8_0", t.row_length)
    if is_output:
        want = "Q8_0" if ftype == "Q8_0" else "Q6_K"
        return want if _fits(want, t.row_length) else "Q8_0"
    want = main
    i, n = seq
    if ftype in ("Q4_K_M", "Q5_K_M") and t.role in ("v", "down") and _use_more_bits(i, max(n, 1)):
        want = "Q6_K"
    elif ftype == "Q3_K_M" and t.role == "v" and i < 2:
        want = "Q5_K"
    elif ftype == "Q3_K_L" and t.role in ("v", "down"):
        want = "Q5_K"
    elif ftype == "Q4_K_S" and t.role == "v" and i < 4:
        want = "Q5_K"
    return _with_fallback(want, t.row_length)


def estimate_tensor_bytes(scan: ScanResult, ftype: str) -> dict[str, tuple[str, int]]:
    has_output = any(t.component == "output" for t in scan.tensors)

    def counted(t: TensorRecord) -> str | None:
        return t.role if t.role in ("v", "down") and t.name.endswith("weight") and len(t.shape) >= 2 else None

    totals: dict[str, int] = {}
    for t in scan.tensors:
        if (r := counted(t)) is not None:
            totals[r] = totals.get(r, 0) + 1
    seen: dict[str, int] = {}
    out: dict[str, tuple[str, int]] = {}
    for t in scan.tensors:
        r = counted(t)
        seq = (seen.get(r, 0), totals.get(r, 0)) if r else (0, 0)
        if r:
            seen[r] = seq[0] + 1
        tt = target_type_for(t, ftype, seq, has_output)
        g = GGML_TYPES_BY_NAME[tt]
        out[t.name] = (tt, -(-t.n_elements // g.block_size) * g.type_size)
    return out


# --------------------------------------------------------------------------- API models


class WeightEstimate(BaseModel):
    id: str
    label: str
    family: str
    bytes: int
    bits_per_weight: float
    provenance: Provenance
    method: str
    type_mix: dict[str, int] = Field(default_factory=dict)  # ggml type -> tensor count


class MemoryComponent(BaseModel):
    key: str
    label: str
    bytes: int
    provenance: Provenance
    formula: str


class FitResult(BaseModel):
    device_id: str
    device_name: str
    memory_kind: str
    capacity_bytes: int
    usable_bytes: int
    required_bytes: int
    headroom_bytes: int
    verdict: Literal["fits", "tight", "partial_offload", "does_not_fit"]
    gpu_layers: int | None = None
    total_layers: int | None = None
    decode_ceiling_tokens_per_s: float | None = None
    notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class MemoryRequest(BaseModel):
    weight_format: str = "current"
    context_length: int = Field(4096, ge=1, le=10_000_000)
    batch_size: int = Field(1, ge=1, le=4096)
    kv_cache_type: Literal["f32", "f16", "bf16", "q8_0", "q4_0"] = "f16"
    flash_attention: bool = True
    n_ubatch: int = Field(512, ge=1, le=65536)
    device_id: str | None = "laptop-cpu-16gb"
    custom_device: DeviceProfile | None = None


class MemoryEstimate(BaseModel):
    is_measurement: Literal[False] = False
    disclaimer: str
    weights: list[WeightEstimate]
    selected_weights: WeightEstimate
    inference: list[MemoryComponent]
    total_bytes: int
    assumptions: list[str]
    warnings: list[str]
    fit: FitResult | None


# --------------------------------------------------------------------------- estimation


def weight_estimates(scan: ScanResult) -> list[WeightEstimate]:
    n_params = scan.fact_value("n_params") or 0
    logical = scan.fact_value("n_params_logical")
    out: list[WeightEstimate] = []
    stored = sum(t.n_bytes or 0 for t in scan.tensors)
    if n_params and not scan.quantization.packed_checkpoint:
        mix: dict[str, int] = {}
        for t in scan.tensors:
            mix[t.dtype] = mix.get(t.dtype, 0) + 1
        out.append(
            WeightEstimate(
                id="current",
                label=f"As stored ({scan.quantization.label or 'current'})",
                family="current",
                bytes=stored,
                bits_per_weight=round(stored * 8 / n_params, 3),
                provenance="derived",
                method="sum of per-tensor byte sizes from the file header",
                type_mix=mix,
            )
        )
    for s in SCHEMES:
        if scan.quantization.packed_checkpoint and logical:
            bpw = {"fp32": 32, "fp16": 16, "q8_0": 8.5, "q4_k_m": 4.85, "q4_0": 4.5}[s.id]
            b = int(logical * bpw / 8)
            out.append(
                WeightEstimate(
                    id=s.id, label=s.label, family=s.family, bytes=b, bits_per_weight=bpw, provenance="inferred",
                    method="logical parameter estimate × nominal bits per weight (packed checkpoint; per-tensor rules unavailable)",
                )
            )
            continue
        per = estimate_tensor_bytes(scan, s.ftype)
        b = sum(v[1] for v in per.values())
        mix = {}
        for tt, _ in per.values():
            mix[tt] = mix.get(tt, 0) + 1
        out.append(
            WeightEstimate(
                id=s.id, label=s.label, family=s.family, bytes=b,
                bits_per_weight=round(b * 8 / n_params, 3) if n_params else 0.0,
                provenance="inferred",
                method="per-tensor replay of llama.cpp type selection incl. block scales, fallbacks and 1-D tensors kept F32",
                type_mix=mix,
            )
        )
    return out


def _attention_layer_count(scan: ScanResult) -> int | None:
    primary = next((s for s in scan.architecture.stacks if s.primary), None)
    if primary is None:
        return None
    return sum(1 for l in primary.layers if any(c.component == "attention" for c in l.components))


def _per_tensor_bytes(scan: ScanResult, fmt: str) -> dict[str, int]:
    if fmt == "current":
        return {t.name: t.n_bytes or 0 for t in scan.tensors}
    s = SCHEME_BY_ID[fmt]
    return {k: v[1] for k, v in estimate_tensor_bytes(scan, s.ftype).items()}


def estimate_memory(scan: ScanResult, req: MemoryRequest) -> MemoryEstimate:
    warnings: list[str] = []
    assumptions: list[str] = []
    weights = weight_estimates(scan)
    sel = next((w for w in weights if w.id == req.weight_format), None)
    if sel is None:
        sel = weights[0]
        if req.weight_format != sel.id:
            warnings.append(f"weight format '{req.weight_format}' unavailable for this model; using '{sel.id}'")

    f = scan.fact_value
    n_layers = f("n_layers")
    hidden = f("hidden_size")
    n_heads = f("n_heads")
    n_kv = f("n_kv_heads")
    hd_k = f("head_dim")
    hd_v = f("head_dim_v") or hd_k
    vocab = f("vocab_size") or 0
    ffn = f("ffn_size") or (4 * hidden if hidden else 0)
    ctx, batch = req.context_length, req.batch_size
    train_ctx = f("context_length")

    comps: list[MemoryComponent] = [
        MemoryComponent(key="weights", label=f"Weights ({sel.label})", bytes=sel.bytes, provenance=sel.provenance, formula=sel.method)
    ]

    # KV cache -------------------------------------------------------------
    attn_layers = _attention_layer_count(scan)
    kv_layers = attn_layers if attn_layers else n_layers
    kv_bytes = 0
    if all(isinstance(x, int) and x > 0 for x in (kv_layers, n_kv, hd_k, hd_v)):
        bpe = KV_BYTES_PER_ELEMENT[req.kv_cache_type]
        tokens = ctx * batch
        kv_bytes = int(kv_layers * tokens * n_kv * (hd_k + hd_v) * bpe)
        inputs = [scan.fact(k) for k in ("n_kv_heads", "head_dim")]
        prov: Provenance = "derived" if all(x is not None and x.provenance != "inferred" for x in inputs) else "inferred"
        comps.append(
            MemoryComponent(
                key="kv_cache",
                label=f"KV cache ({req.kv_cache_type.upper()})",
                bytes=kv_bytes,
                provenance=prov,
                formula=f"{kv_layers} layers × {tokens:,} tokens × {n_kv} KV heads × ({hd_k}+{hd_v}) dims × {bpe:g} B",
            )
        )
        assumptions.append(f"context length {ctx:,} tokens per sequence × {batch} parallel sequence(s); llama.cpp's -c equals the total ({tokens:,})")
        if attn_layers and n_layers and attn_layers < n_layers:
            assumptions.append(f"only {attn_layers} of {n_layers} layers have attention tensors; recurrent-state memory is not modelled")
    else:
        warnings.append("KV cache could not be estimated: missing layer count, KV heads or head dimension")

    if req.kv_cache_type in ("q8_0", "q4_0") and not req.flash_attention:
        warnings.append("llama.cpp requires flash attention for a quantized V cache")
    sw = f("sliding_window")
    if isinstance(sw, int) and 0 < sw < ctx:
        assumptions.append(f"sliding-window attention ({sw} tokens) is ignored; KV is sized as if every layer attends to the full context (upper bound)")
    if isinstance(train_ctx, int) and ctx > train_ctx:
        warnings.append(f"requested context {ctx:,} exceeds the model's training context {train_ctx:,}")
    fam = (scan.architecture.family or "").lower()
    if fam.startswith("deepseek2") or fam in ("deepseek_v2", "deepseek_v3", "kimi_k2"):
        warnings.append("multi-head latent attention compresses the KV cache; this figure is an upper bound")

    # Compute buffer (heuristic) ------------------------------------------------
    if isinstance(hidden, int) and hidden > 0:
        ub = min(req.n_ubatch, ctx * batch)
        logits = vocab * batch * 4
        acts = ub * (hidden * 4 + ffn * 2) * 4
        scores = 0 if req.flash_attention or not n_heads else n_heads * ub * ctx * 4
        comps.append(
            MemoryComponent(
                key="compute",
                label="Compute buffers (heuristic)",
                bytes=logits + acts + scores,
                provenance="inferred",
                formula=(
                    f"logits {vocab:,}×{batch}×4 B + activations {ub}×(4·{hidden}+2·{ffn})×4 B"
                    + ("" if not scores else f" + attention scores {n_heads}×{ub}×{ctx:,}×4 B (no flash attention)")
                ),
            )
        )
        assumptions.append(f"micro-batch (n_ubatch) of {ub} tokens for prompt processing; FP32 activations")
        if not req.flash_attention:
            assumptions.append("without flash attention the KQ score matrix is materialised per micro-batch")

    profile: DeviceProfile | None = req.custom_device or (get_profile(req.device_id) if req.device_id else None)
    overhead = profile.runtime_overhead_bytes if profile else 200 * 1024**2
    comps.append(
        MemoryComponent(
            key="runtime",
            label="Runtime overhead",
            bytes=overhead,
            provenance="inferred",
            formula="device-profile assumption" if profile else "default 200 MB assumption",
        )
    )
    total = sum(c.bytes for c in comps)

    fit = _fit(scan, profile, comps, sel, req, kv_bytes) if profile else None
    return MemoryEstimate(
        disclaimer="Estimates derived from file metadata and stated assumptions. They are not measurements; real usage depends on runtime, build flags and drivers.",
        weights=weights,
        selected_weights=sel,
        inference=comps,
        total_bytes=total,
        assumptions=assumptions,
        warnings=warnings,
        fit=fit,
    )


def _fit(scan: ScanResult, p: DeviceProfile, comps: list[MemoryComponent], sel: WeightEstimate, req: MemoryRequest, kv_bytes: int) -> FitResult:
    usable = p.usable_bytes
    total = sum(c.bytes for c in comps)
    notes: list[str] = []
    per_tensor = _per_tensor_bytes(scan, sel.id)

    def verdict_for(required: int, avail: int) -> str:
        return "fits" if required <= 0.9 * avail else ("tight" if required <= avail else "does_not_fit")

    verdict = verdict_for(total, usable)
    gpu_layers = None
    total_layers = None
    ceiling = None

    # bytes read per generated token: all weights except the embedding table (one row is gathered),
    # with only the routed fraction of expert weights, plus the whole KV cache (upper bound).
    n_exp, n_used = scan.fact_value("n_experts"), scan.fact_value("n_experts_used")
    frac = (n_used / n_exp) if isinstance(n_exp, int) and isinstance(n_used, int) and n_exp > 0 else 1.0
    read = 0
    for t in scan.tensors:
        b = per_tensor.get(t.name, 0)
        if t.component == "embedding" and any(x.component == "output" for x in scan.tensors):
            continue
        read += int(b * frac) if t.component == "experts" else b
    read += kv_bytes // max(req.batch_size, 1)

    if p.memory_kind == "discrete_gpu" and verdict == "does_not_fit":
        primary = next((s for s in scan.architecture.stacks if s.primary), None)
        if primary and primary.layers:
            total_layers = primary.n_layers
            layer_bytes = [sum(per_tensor.get(n, 0) for c in l.components for n in c.tensors) for l in primary.layers]
            per_layer = max(layer_bytes) + kv_bytes // max(len(layer_bytes), 1)
            fixed = sum(c.bytes for c in comps if c.key in ("compute", "runtime"))
            out_bytes = sum(per_tensor.get(t.name, 0) for t in scan.tensors if t.component == "output")
            budget = usable - fixed - out_bytes
            gpu_layers = max(0, min(total_layers, budget // per_layer if per_layer else 0))
            host_need = total - (gpu_layers * per_layer) - fixed
            if gpu_layers >= total_layers:
                verdict = "tight"
                notes.append(
                    f"all {total_layers} layers fit in VRAM only because llama.cpp keeps the token-embedding table in host RAM; little headroom"
                )
            elif gpu_layers > 0 and (p.host_ram_bytes is None or host_need <= p.host_ram_bytes):
                verdict = "partial_offload"
                notes.append(f"about {gpu_layers} of {total_layers} layers fit in VRAM; the rest run from system RAM (much slower)")
            else:
                notes.append("model does not fit in VRAM and the remainder exceeds the assumed host RAM")
    elif p.bandwidth_gbps and read:
        ceiling = round(p.bandwidth_gbps * 1e9 / read, 1)
        notes.append("decode ceiling = memory bandwidth ÷ bytes read per token; a theoretical upper bound, not a benchmark")

    if p.memory_kind == "unified":
        notes.append("unified memory: CPU and GPU allocations share the same pool")
    if p.memory_kind == "discrete_gpu":
        notes.append("llama.cpp keeps the token-embedding table in host RAM even with full GPU offload")
    return FitResult(
        device_id=p.id,
        device_name=p.name,
        memory_kind=p.memory_kind,
        capacity_bytes=p.capacity_bytes,
        usable_bytes=usable,
        required_bytes=total,
        headroom_bytes=usable - total,
        verdict=verdict,  # type: ignore[arg-type]
        gpu_layers=gpu_layers,
        total_layers=total_layers,
        decode_ceiling_tokens_per_s=ceiling,
        notes=notes,
        assumptions=p.assumptions,
    )

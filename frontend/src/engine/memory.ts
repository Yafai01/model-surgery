// Port of backend/model_surgery/memory.py and devices.py. Estimates only, never measurements.

import type { DeviceProfile, FitResult, MemoryComponent, MemoryEstimate, MemoryRequest, Provenance, ScanResult, TensorRecord, WeightEstimate } from "../types";
import { GGML_BY_NAME } from "./ggml";
import { factValue } from "./scanner";

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

export const DEFAULT_PROFILES: DeviceProfile[] = [
  {
    id: "laptop-cpu-16gb", name: "Laptop · CPU inference · 16 GB RAM", memory_kind: "system_ram", capacity_bytes: 16 * GiB, reserved_bytes: 5 * GiB,
    runtime_overhead_bytes: 200 * MiB, host_ram_bytes: null, bandwidth_gbps: 60,
    assumptions: [
      "16 GB system RAM; ~5 GB reserved for Windows 11, browser and background apps",
      "~200 MB llama.cpp CPU runtime overhead",
      "~60 GB/s dual-channel DDR5 bandwidth (laptop-typical, varies widely)",
    ],
  },
  {
    id: "laptop-gpu-8gb", name: "Laptop · 8 GB VRAM GPU + 16 GB RAM", memory_kind: "discrete_gpu", capacity_bytes: 8 * GiB, reserved_bytes: 512 * MiB,
    runtime_overhead_bytes: 450 * MiB, host_ram_bytes: 11 * GiB, bandwidth_gbps: 256,
    assumptions: [
      "8 GB dedicated VRAM; ~0.5 GB reserved for the display/compositor",
      "~450 MB CUDA context and cuBLAS workspace",
      "layers that do not fit are assumed to stay in system RAM (partial offload)",
      "~256 GB/s VRAM bandwidth (mid-range laptop GPU)",
    ],
  },
  {
    id: "jetson-orin-nano-8gb", name: "Jetson Orin Nano 8GB · desktop", memory_kind: "unified", capacity_bytes: Math.trunc(7.4 * GiB), reserved_bytes: Math.trunc(1.8 * GiB),
    runtime_overhead_bytes: 350 * MiB, host_ram_bytes: null, bandwidth_gbps: 68,
    assumptions: [
      "8 GB LPDDR5 shared by CPU and GPU; ~7.4 GiB visible to Linux after carve-outs",
      "~1.8 GiB reserved for JetPack Ubuntu desktop (GNOME) and system services",
      "~350 MB CUDA context; GPU allocations come from the same pool as the CPU",
      "68 GB/s memory bandwidth (102 GB/s in JetPack 6.2 'Super' MAXN mode)",
    ],
  },
  {
    id: "jetson-orin-nano-8gb-headless", name: "Jetson Orin Nano 8GB · headless", memory_kind: "unified", capacity_bytes: Math.trunc(7.4 * GiB), reserved_bytes: Math.trunc(0.8 * GiB),
    runtime_overhead_bytes: 350 * MiB, host_ram_bytes: null, bandwidth_gbps: 68,
    assumptions: [
      "same hardware as the desktop profile with the GUI disabled (multi-user.target)",
      "~0.8 GiB reserved for the kernel and system services",
      "68 GB/s memory bandwidth (102 GB/s in JetPack 6.2 'Super' MAXN mode)",
    ],
  },
];

const KV_BPE: Record<string, number> = { f32: 4, f16: 2, bf16: 2, q8_0: 34 / 32, q4_0: 18 / 32 };

const FTYPE_MAIN: Record<string, string> = {
  F32: "F32", F16: "F16", BF16: "BF16", Q8_0: "Q8_0", Q4_0: "Q4_0", Q4_1: "Q4_1", Q5_0: "Q5_0", Q5_1: "Q5_1",
  Q2_K: "Q2_K", Q2_K_S: "Q2_K", Q3_K: "Q3_K", Q3_K_S: "Q3_K", Q3_K_M: "Q3_K", Q3_K_L: "Q3_K",
  Q4_K: "Q4_K", Q4_K_S: "Q4_K", Q4_K_M: "Q4_K", Q5_K: "Q5_K", Q5_K_S: "Q5_K", Q5_K_M: "Q5_K", Q6_K: "Q6_K",
  IQ1_S: "IQ1_S", IQ1_M: "IQ1_M", IQ2_XXS: "IQ2_XXS", IQ2_XS: "IQ2_XS", IQ2_S: "IQ2_S", IQ2_M: "IQ2_S",
  IQ3_XXS: "IQ3_XXS", IQ3_XS: "IQ3_S", IQ3_S: "IQ3_S", IQ3_M: "IQ3_S", IQ4_NL: "IQ4_NL", IQ4_XS: "IQ4_XS",
  TQ1_0: "TQ1_0", TQ2_0: "TQ2_0", MXFP4_MOE: "MXFP4",
};
const FLOAT_FTYPES = new Set(["F32", "F16", "BF16"]);
const FALLBACK: Record<string, string> = {
  Q4_K: "Q5_0", Q5_K: "Q5_1", Q6_K: "Q8_0", Q2_K: "IQ4_NL", Q3_K: "IQ4_NL", IQ2_XXS: "IQ4_NL", IQ2_XS: "IQ4_NL", IQ2_S: "IQ4_NL",
  IQ3_XXS: "IQ4_NL", IQ3_S: "IQ4_NL", IQ1_S: "IQ4_NL", IQ1_M: "IQ4_NL", IQ4_XS: "IQ4_NL", TQ1_0: "IQ4_NL", TQ2_0: "IQ4_NL",
};

export const SCHEMES = [
  { id: "fp32", label: "FP32", ftype: "F32", family: "FP32", nominalBpw: 32 },
  { id: "fp16", label: "FP16 / BF16", ftype: "F16", family: "FP16", nominalBpw: 16 },
  { id: "q8_0", label: "INT8 · Q8_0", ftype: "Q8_0", family: "INT8", nominalBpw: 8.5 },
  { id: "q4_k_m", label: "4-bit · Q4_K_M", ftype: "Q4_K_M", family: "4-bit", nominalBpw: 4.85 },
  { id: "q4_0", label: "4-bit · Q4_0", ftype: "Q4_0", family: "4-bit", nominalBpw: 4.5 },
];

// Python-style modulo (result has the sign of the divisor).
const pmod = (a: number, b: number) => ((a % b) + b) % b;
export const useMoreBits = (i: number, n: number) => i < Math.floor(n / 8) || i >= Math.floor((7 * n) / 8) || pmod(i - Math.floor(n / 8), 3) === 2;
const fits = (t: string, row: number) => row % GGML_BY_NAME.get(t)!.blockSize === 0;

function withFallback(type: string, row: number): string {
  let t = type;
  for (let k = 0; k < 3; k++) {
    if (fits(t, row)) return t;
    t = FALLBACK[t] ?? (t !== "Q8_0" && t !== "F16" ? "Q8_0" : "F16");
  }
  return "F16";
}

export function targetTypeFor(t: TensorRecord, ftype: string, seq: [number, number], hasOutput: boolean): string {
  if (t.shape.length < 2 || !t.name.endsWith("weight")) return "F32";
  if (t.component === "router") return "F32";
  if (t.role === "position" || t.name.includes("conv1d")) return ftype === "F32" ? "F32" : "F16";
  const main = FTYPE_MAIN[ftype] ?? ftype;
  if (FLOAT_FTYPES.has(ftype)) return main;
  const isOutput = t.component === "output" || (!hasOutput && t.component === "embedding" && t.role === "token");
  if (ftype === "MXFP4_MOE") return t.component === "experts" && fits("MXFP4", t.row_length) ? "MXFP4" : withFallback("Q8_0", t.row_length);
  if (isOutput) {
    const want = ftype === "Q8_0" ? "Q8_0" : "Q6_K";
    return fits(want, t.row_length) ? want : "Q8_0";
  }
  let want = main;
  const [i, n] = seq;
  if ((ftype === "Q4_K_M" || ftype === "Q5_K_M") && (t.role === "v" || t.role === "down") && useMoreBits(i, Math.max(n, 1))) want = "Q6_K";
  else if (ftype === "Q3_K_M" && t.role === "v" && i < 2) want = "Q5_K";
  else if (ftype === "Q3_K_L" && (t.role === "v" || t.role === "down")) want = "Q5_K";
  else if (ftype === "Q4_K_S" && t.role === "v" && i < 4) want = "Q5_K";
  return withFallback(want, t.row_length);
}

/** Per-tensor [ggml type, bytes] under a llama.cpp preset, replaying its type-selection rules in file order. */
export function estimateTensorBytes(scan: ScanResult, ftype: string): Map<string, [string, number]> {
  const hasOutput = scan.tensors.some((t) => t.component === "output");
  const counted = (t: TensorRecord) => ((t.role === "v" || t.role === "down") && t.name.endsWith("weight") && t.shape.length >= 2 ? t.role : null);
  const totals: Record<string, number> = {};
  for (const t of scan.tensors) {
    const r = counted(t);
    if (r) totals[r] = (totals[r] ?? 0) + 1;
  }
  const seen: Record<string, number> = {};
  const out = new Map<string, [string, number]>();
  for (const t of scan.tensors) {
    const r = counted(t);
    const seq: [number, number] = r ? [seen[r] ?? 0, totals[r] ?? 0] : [0, 0];
    if (r) seen[r] = seq[0] + 1;
    const tt = targetTypeFor(t, ftype, seq, hasOutput);
    const g = GGML_BY_NAME.get(tt)!;
    out.set(t.name, [tt, Math.ceil(t.n_elements / g.blockSize) * g.typeSize]);
  }
  return out;
}

const round3 = (x: number) => Math.round(x * 1000) / 1000;

export function weightEstimates(scan: ScanResult): WeightEstimate[] {
  const nParams = (factValue(scan, "n_params") as number) || 0;
  const logical = factValue(scan, "n_params_logical") as number | null;
  const out: WeightEstimate[] = [];
  const stored = scan.tensors.reduce((a, t) => a + (t.n_bytes ?? 0), 0);
  if (nParams && !scan.quantization.packed_checkpoint) {
    const mix: Record<string, number> = {};
    for (const t of scan.tensors) mix[t.dtype] = (mix[t.dtype] ?? 0) + 1;
    out.push({
      id: "current", label: `As stored (${scan.quantization.label ?? "current"})`, family: "current", bytes: stored,
      bits_per_weight: round3((stored * 8) / nParams), provenance: "derived", method: "sum of per-tensor byte sizes from the file header", type_mix: mix,
    });
  }
  for (const s of SCHEMES) {
    if (scan.quantization.packed_checkpoint && logical) {
      out.push({
        id: s.id, label: s.label, family: s.family, bytes: Math.trunc((logical * s.nominalBpw) / 8), bits_per_weight: s.nominalBpw, provenance: "inferred",
        method: "logical parameter estimate × nominal bits per weight (packed checkpoint; per-tensor rules unavailable)", type_mix: {},
      });
      continue;
    }
    const per = estimateTensorBytes(scan, s.ftype);
    let b = 0;
    const mix: Record<string, number> = {};
    for (const [tt, n] of per.values()) {
      b += n;
      mix[tt] = (mix[tt] ?? 0) + 1;
    }
    out.push({
      id: s.id, label: s.label, family: s.family, bytes: b, bits_per_weight: nParams ? round3((b * 8) / nParams) : 0, provenance: "inferred",
      method: "per-tensor replay of llama.cpp type selection incl. block scales, fallbacks and 1-D tensors kept F32", type_mix: mix,
    });
  }
  return out;
}

function perTensorBytes(scan: ScanResult, fmt: string): Map<string, number> {
  if (fmt === "current") return new Map(scan.tensors.map((t) => [t.name, t.n_bytes ?? 0]));
  const s = SCHEMES.find((x) => x.id === fmt)!;
  return new Map([...estimateTensorBytes(scan, s.ftype)].map(([k, v]) => [k, v[1]]));
}

const isPosInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && v > 0;

export function estimateMemory(scan: ScanResult, req: MemoryRequest): MemoryEstimate {
  const warnings: string[] = [];
  const assumptions: string[] = [];
  const weights = weightEstimates(scan);
  let sel = weights.find((w) => w.id === req.weight_format);
  if (!sel) {
    sel = weights[0];
    if (req.weight_format !== sel.id) warnings.push(`weight format '${req.weight_format}' unavailable for this model; using '${sel.id}'`);
  }
  const f = (k: string) => factValue(scan, k);
  const nLayers = f("n_layers") as number | null;
  const hidden = f("hidden_size") as number | null;
  const nHeads = f("n_heads") as number | null;
  const nKv = f("n_kv_heads") as number | null;
  const hdK = f("head_dim") as number | null;
  const hdV = (f("head_dim_v") as number | null) || hdK;
  const vocab = (f("vocab_size") as number | null) || 0;
  const ffn = (f("ffn_size") as number | null) || (hidden ? 4 * hidden : 0);
  const { context_length: ctx, batch_size: batch } = req;
  const trainCtx = f("context_length");

  const comps: MemoryComponent[] = [{ key: "weights", label: `Weights (${sel.label})`, bytes: sel.bytes, provenance: sel.provenance, formula: sel.method }];

  const primary = scan.architecture.stacks.find((s) => s.primary);
  const attnLayers = primary ? primary.layers.filter((l) => l.components.some((c) => c.component === "attention")).length : null;
  const kvLayers = attnLayers || nLayers;
  let kvBytes = 0;
  if ([kvLayers, nKv, hdK, hdV].every(isPosInt)) {
    const bpe = KV_BPE[req.kv_cache_type];
    const tokens = ctx * batch;
    kvBytes = Math.trunc(kvLayers! * tokens * nKv! * (hdK! + hdV!) * bpe);
    const inputs = ["n_kv_heads", "head_dim"].map((k) => scan.hyperparameters.find((x) => x.key === k));
    const prov: Provenance = inputs.every((x) => x && x.provenance !== "inferred") ? "derived" : "inferred";
    comps.push({
      key: "kv_cache", label: `KV cache (${req.kv_cache_type.toUpperCase()})`, bytes: kvBytes, provenance: prov,
      formula: `${kvLayers} layers × ${tokens.toLocaleString("en-US")} tokens × ${nKv} KV heads × (${hdK}+${hdV}) dims × ${+bpe.toPrecision(6)} B`,
    });
    assumptions.push(`context length ${ctx.toLocaleString("en-US")} tokens per sequence × ${batch} parallel sequence(s); llama.cpp's -c equals the total (${tokens.toLocaleString("en-US")})`);
    if (attnLayers && nLayers && attnLayers < nLayers) assumptions.push(`only ${attnLayers} of ${nLayers} layers have attention tensors; recurrent-state memory is not modelled`);
  } else warnings.push("KV cache could not be estimated: missing layer count, KV heads or head dimension");

  if ((req.kv_cache_type === "q8_0" || req.kv_cache_type === "q4_0") && !req.flash_attention) warnings.push("llama.cpp requires flash attention for a quantized V cache");
  const sw = f("sliding_window");
  if (isPosInt(sw) && sw < ctx) assumptions.push(`sliding-window attention (${sw} tokens) is ignored; KV is sized as if every layer attends to the full context (upper bound)`);
  if (isPosInt(trainCtx) && ctx > trainCtx) warnings.push(`requested context ${ctx.toLocaleString("en-US")} exceeds the model's training context ${trainCtx.toLocaleString("en-US")}`);
  const fam = (scan.architecture.family ?? "").toLowerCase();
  if (fam.startsWith("deepseek2") || ["deepseek_v2", "deepseek_v3", "kimi_k2"].includes(fam)) warnings.push("multi-head latent attention compresses the KV cache; this figure is an upper bound");

  if (isPosInt(hidden)) {
    const ub = Math.min(req.n_ubatch, ctx * batch);
    const logits = vocab * batch * 4;
    const acts = ub * (hidden * 4 + ffn * 2) * 4;
    const scores = req.flash_attention || !nHeads ? 0 : nHeads * ub * ctx * 4;
    comps.push({
      key: "compute", label: "Compute buffers (heuristic)", bytes: logits + acts + scores, provenance: "inferred",
      formula: `logits ${vocab.toLocaleString("en-US")}×${batch}×4 B + activations ${ub}×(4·${hidden}+2·${ffn})×4 B` + (scores ? ` + attention scores ${nHeads}×${ub}×${ctx.toLocaleString("en-US")}×4 B (no flash attention)` : ""),
    });
    assumptions.push(`micro-batch (n_ubatch) of ${ub} tokens for prompt processing; FP32 activations`);
    if (!req.flash_attention) assumptions.push("without flash attention the KQ score matrix is materialised per micro-batch");
  }

  const profile = req.custom_device ?? (req.device_id ? DEFAULT_PROFILES.find((p) => p.id === req.device_id) ?? null : null);
  comps.push({
    key: "runtime", label: "Runtime overhead", bytes: profile ? profile.runtime_overhead_bytes : 200 * MiB, provenance: "inferred",
    formula: profile ? "device-profile assumption" : "default 200 MB assumption",
  });
  const total = comps.reduce((a, c) => a + c.bytes, 0);
  return {
    is_measurement: false,
    disclaimer: "Estimates derived from file metadata and stated assumptions. They are not measurements; real usage depends on runtime, build flags and drivers.",
    weights,
    selected_weights: sel,
    inference: comps,
    total_bytes: total,
    assumptions,
    warnings,
    fit: profile ? fit(scan, profile, comps, sel, req, kvBytes) : null,
  };
}

function fit(scan: ScanResult, p: DeviceProfile, comps: MemoryComponent[], sel: WeightEstimate, req: MemoryRequest, kvBytes: number): FitResult {
  const usable = Math.max(0, p.capacity_bytes - p.reserved_bytes);
  const total = comps.reduce((a, c) => a + c.bytes, 0);
  const notes: string[] = [];
  const per = perTensorBytes(scan, sel.id);
  const verdictFor = (req_: number, avail: number): FitResult["verdict"] => (req_ <= 0.9 * avail ? "fits" : req_ <= avail ? "tight" : "does_not_fit");
  let verdict = verdictFor(total, usable);
  let gpuLayers: number | null = null;
  let totalLayers: number | null = null;
  let ceiling: number | null = null;

  const nExp = factValue(scan, "n_experts");
  const nUsed = factValue(scan, "n_experts_used");
  const frac = isPosInt(nExp) && isPosInt(nUsed) ? nUsed / nExp : 1;
  const hasOutput = scan.tensors.some((x) => x.component === "output");
  let read = 0;
  for (const t of scan.tensors) {
    const b = per.get(t.name) ?? 0;
    if (t.component === "embedding" && hasOutput) continue;
    read += t.component === "experts" ? Math.trunc(b * frac) : b;
  }
  read += Math.floor(kvBytes / Math.max(req.batch_size, 1));

  if (p.memory_kind === "discrete_gpu" && verdict === "does_not_fit") {
    const primary = scan.architecture.stacks.find((s) => s.primary);
    if (primary && primary.layers.length) {
      totalLayers = primary.n_layers;
      const layerBytes = primary.layers.map((l) => l.components.reduce((a, c) => a + c.tensors.reduce((x, n) => x + (per.get(n) ?? 0), 0), 0));
      const perLayer = Math.max(...layerBytes) + Math.floor(kvBytes / Math.max(layerBytes.length, 1));
      const fixed = comps.filter((c) => c.key === "compute" || c.key === "runtime").reduce((a, c) => a + c.bytes, 0);
      const outBytes = scan.tensors.filter((t) => t.component === "output").reduce((a, t) => a + (per.get(t.name) ?? 0), 0);
      const budget = usable - fixed - outBytes;
      gpuLayers = Math.max(0, Math.min(totalLayers, perLayer ? Math.floor(budget / perLayer) : 0));
      const hostNeed = total - gpuLayers * perLayer - fixed;
      if (gpuLayers >= totalLayers) {
        verdict = "tight";
        notes.push(`all ${totalLayers} layers fit in VRAM only because llama.cpp keeps the token-embedding table in host RAM; little headroom`);
      } else if (gpuLayers > 0 && (p.host_ram_bytes === null || hostNeed <= p.host_ram_bytes)) {
        verdict = "partial_offload";
        notes.push(`about ${gpuLayers} of ${totalLayers} layers fit in VRAM; the rest run from system RAM (much slower)`);
      } else notes.push("model does not fit in VRAM and the remainder exceeds the assumed host RAM");
    }
  } else if (p.bandwidth_gbps && read) {
    ceiling = Math.round(((p.bandwidth_gbps * 1e9) / read) * 10) / 10;
    notes.push("decode ceiling = memory bandwidth ÷ bytes read per token; a theoretical upper bound, not a benchmark");
  }
  if (p.memory_kind === "unified") notes.push("unified memory: CPU and GPU allocations share the same pool");
  if (p.memory_kind === "discrete_gpu") notes.push("llama.cpp keeps the token-embedding table in host RAM even with full GPU offload");
  return {
    device_id: p.id, device_name: p.name, memory_kind: p.memory_kind, capacity_bytes: p.capacity_bytes, usable_bytes: usable,
    required_bytes: total, headroom_bytes: usable - total, verdict, gpu_layers: gpuLayers, total_layers: totalLayers,
    decode_ceiling_tokens_per_s: ceiling, notes, assumptions: p.assumptions,
  };
}

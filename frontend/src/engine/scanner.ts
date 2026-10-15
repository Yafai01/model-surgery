// Port of backend/model_surgery/scanner.py for files picked in the browser.
// Nothing is uploaded: only header bytes are read from the local File objects.

import type { DTypeRow, Fact, FormatInfo, MetadataEntry, Provenance, QuantizationSummary, ScanResult, SourceFile, TensorRecord } from "../types";
import { buildArchitecture, classify } from "./architecture";
import { GGML_BY_NAME, UNQUANTIZED_FTYPES, ftypeLabel } from "./ggml";
import { FormatError, type GGUFFile, ggufGet, isGGUF, readGGUF } from "./gguf";
import { PICKLE_SUFFIXES, UnsupportedFormatError, findConfig, readHFFolder, relPath, suffix } from "./hf";
import { readSafetensors, type SafetensorsFile } from "./safetensors";

export const APP_VERSION = "0.1.0-web";
const SPLIT_RE = /^(.+)-(\d{5})-of-(\d{5})\.gguf$/i;
const PACKED = new Set(["gptq", "awq", "bitsandbytes", "hqq", "aqlm", "eetq", "exl2", "quanto", "vptq", "higgs", "spqr"]);
const UNSUPPORTED: Record<string, string> = {
  ".onnx": "ONNX models are not supported yet",
  ".tflite": "TensorFlow Lite models are not supported",
  ".h5": "Keras/HDF5 weights are not supported",
  ".msgpack": "Flax msgpack weights are not supported",
  ".engine": "TensorRT engines are not supported",
  ".mlmodel": "Core ML models are not supported",
};

export class ScanError extends Error {}

// ------------------------------------------------------------------ facts

const FACT_ORDER = [
  "architecture", "model_class", "n_params", "n_params_logical", "n_layers", "hidden_size", "n_heads", "n_kv_heads",
  "attention_type", "head_dim", "head_dim_v", "ffn_size", "vocab_size", "context_length", "n_experts", "n_experts_used",
  "sliding_window", "rope_theta", "norm_eps", "tied_embeddings", "ssm_state_size", "torch_dtype",
];

class Facts {
  items = new Map<string, Fact>();
  add(key: string, label: string, value: unknown, prov: Provenance, source: string, override = false) {
    if (value === null || value === undefined || (this.items.has(key) && !override)) return;
    this.items.set(key, { key, label, value: value as Fact["value"], provenance: prov, source });
  }
  get(key: string): unknown {
    return this.items.get(key)?.value ?? null;
  }
  ordered(): Fact[] {
    const idx = (k: string) => (FACT_ORDER.includes(k) ? FACT_ORDER.indexOf(k) : FACT_ORDER.length);
    return [...this.items.values()].sort((a, b) => idx(a.key) - idx(b.key));
  }
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

function scalarize(value: unknown, key: string, notes: string[]): unknown {
  if (Array.isArray(value)) {
    const nums = value.filter(isNum);
    if (!nums.length) return null;
    if (new Set(nums).size > 1) notes.push(`${key} varies per layer (${Math.min(...nums)}..${Math.max(...nums)}); the maximum is used for estimates`);
    return Math.max(...nums);
  }
  return value;
}

function factsFromGGUF(g: GGUFFile, facts: Facts, notes: string[]): string | null {
  const arch = ggufGet(g, "general.architecture");
  if (typeof arch !== "string") {
    notes.push("GGUF has no general.architecture key");
    return null;
  }
  facts.add("architecture", "Architecture", arch, "verified", "general.architecture");
  const kv = (label: string, key: string, sfx: string) => {
    const full = `${arch}.${sfx}`;
    if (g.metadata.has(full)) facts.add(key, label, scalarize(ggufGet(g, full), full, notes), "verified", full);
  };
  kv("Layers", "n_layers", "block_count");
  kv("Hidden size", "hidden_size", "embedding_length");
  kv("Attention heads", "n_heads", "attention.head_count");
  kv("KV heads", "n_kv_heads", "attention.head_count_kv");
  kv("Head dim (K)", "head_dim", "attention.key_length");
  kv("Head dim (V)", "head_dim_v", "attention.value_length");
  kv("FFN size", "ffn_size", "feed_forward_length");
  kv("Training context length", "context_length", "context_length");
  kv("Vocabulary size", "vocab_size", "vocab_size");
  kv("Experts", "n_experts", "expert_count");
  kv("Experts used per token", "n_experts_used", "expert_used_count");
  kv("Sliding window", "sliding_window", "attention.sliding_window");
  kv("RoPE base", "rope_theta", "rope.freq_base");
  kv("Norm epsilon", "norm_eps", "attention.layer_norm_rms_epsilon");
  kv("Norm epsilon", "norm_eps", "attention.layer_norm_epsilon");
  kv("SSM state size", "ssm_state_size", "ssm.state_size");
  const tok = g.metadata.get("tokenizer.ggml.tokens");
  if (tok && tok.type.startsWith("array")) facts.add("vocab_size", "Vocabulary size", tok.length, "verified", "length of tokenizer.ggml.tokens");
  return arch;
}

const HF_KEYS: [string, string, string[]][] = [
  ["n_layers", "Layers", ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]],
  ["hidden_size", "Hidden size", ["hidden_size", "n_embd", "d_model", "dim"]],
  ["n_heads", "Attention heads", ["num_attention_heads", "n_head", "num_heads", "n_heads"]],
  ["n_kv_heads", "KV heads", ["num_key_value_heads", "num_kv_heads", "n_head_kv", "n_kv_heads", "multi_query_group_num"]],
  ["head_dim", "Head dim (K)", ["head_dim", "kv_channels"]],
  ["ffn_size", "FFN size", ["intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size", "hidden_dim"]],
  ["vocab_size", "Vocabulary size", ["vocab_size", "padded_vocab_size"]],
  ["context_length", "Training context length", ["max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length"]],
  ["n_experts", "Experts", ["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"]],
  ["n_experts_used", "Experts used per token", ["num_experts_per_tok", "moe_top_k", "top_k_experts"]],
  ["sliding_window", "Sliding window", ["sliding_window"]],
  ["rope_theta", "RoPE base", ["rope_theta", "rotary_emb_base"]],
  ["norm_eps", "Norm epsilon", ["rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"]],
  ["ssm_state_size", "SSM state size", ["state_size", "d_state", "ssm_state_size"]],
];

function factsFromHF(config: Record<string, unknown>, facts: Facts, notes: string[]): string | null {
  let lm = config;
  let src = "config.json";
  for (const sub of ["text_config", "llm_config", "language_config"]) {
    const v = config[sub];
    if (typeof v === "object" && v !== null && !Array.isArray(v)) {
      lm = { ...config, ...(v as Record<string, unknown>) };
      src = `config.json:${sub}`;
      notes.push(`language-model hyperparameters taken from '${sub}' (multimodal config)`);
      break;
    }
  }
  const mt = config.model_type;
  if (typeof mt === "string") facts.add("architecture", "Architecture", mt, "verified", "config.json:model_type");
  const archs = config.architectures;
  if (Array.isArray(archs) && typeof archs[0] === "string") facts.add("model_class", "Model class", archs[0], "verified", "config.json:architectures[0]");
  for (const [key, label, names] of HF_KEYS) {
    for (const n of names) {
      const v = lm[n];
      if (isNum(v)) {
        facts.add(key, label, v, "verified", `${src}:${n}`);
        break;
      }
      if (Array.isArray(v) && v.length && v.every(isNum)) {
        facts.add(key, label, scalarize(v, n, notes), "verified", `${src}:${n}`);
        break;
      }
    }
  }
  if (facts.get("n_kv_heads") === null && lm.multi_query === true) facts.add("n_kv_heads", "KV heads", 1, "verified", `${src}:multi_query`);
  if (typeof lm.tie_word_embeddings === "boolean") facts.add("tied_embeddings", "Tied embeddings", lm.tie_word_embeddings, "verified", `${src}:tie_word_embeddings`);
  const dt = lm.torch_dtype ?? lm.dtype;
  if (typeof dt === "string") facts.add("torch_dtype", "Declared dtype", dt, "verified", `${src}:torch_dtype`);
  return typeof mt === "string" ? mt : null;
}

function shapeDims(t: TensorRecord): [number, number] | null {
  if (t.shape.length < 2) return null;
  return t.shape_order === "ggml" ? [t.shape[0], t.shape[1]] : [t.shape[t.shape.length - 1], t.shape[t.shape.length - 2]];
}

export function estimateParamsFromDims(get: (k: string) => unknown): number | null {
  const L = get("n_layers"), H = get("hidden_size"), V = get("vocab_size");
  if (![L, H, V].every((x) => isInt(x) && x > 0)) return null;
  const [l, h, v] = [L as number, H as number, V as number];
  const nh = (get("n_heads") as number) || 1;
  const nkv = (get("n_kv_heads") as number) || nh;
  const hd = (get("head_dim") as number) || Math.floor(h / nh);
  const F = (get("ffn_size") as number) || 4 * h;
  const E = (get("n_experts") as number) || 1;
  const attn = h * nh * hd * 2 + h * nkv * hd * 2;
  const ffn = 3 * h * F * E;
  const emb = v * h * (get("tied_embeddings") !== false ? 1 : 2);
  return l * (attn + ffn + 2 * h) + emb + h;
}

function completeFacts(facts: Facts, tensors: TensorRecord[], notes: string[], packed: boolean) {
  facts.add("n_params", "Parameters (tensor elements)", tensors.reduce((a, t) => a + t.n_elements, 0), "derived", "sum of tensor element counts");
  const byRole = (c: string, r: string | null) => tensors.filter((t) => t.component === c && t.role === r);

  const emb = byRole("embedding", "token").find((t) => t.shape.length === 2);
  if (emb && !packed) {
    const [hidden, vocab] = shapeDims(emb)!;
    if (facts.get("hidden_size") === null) facts.add("hidden_size", "Hidden size", hidden, "inferred", `shape of ${emb.name}`);
    else if (facts.get("hidden_size") !== hidden) notes.push(`metadata hidden size ${facts.get("hidden_size")} differs from ${emb.name} width ${hidden}`);
    if (facts.get("vocab_size") === null) facts.add("vocab_size", "Vocabulary size", vocab, "inferred", `shape of ${emb.name}`);
  }

  const stackCounts = new Map<string, number>();
  for (const t of tensors) if (t.stack !== null) stackCounts.set(t.stack, (stackCounts.get(t.stack) ?? 0) + 1);
  if (stackCounts.size) {
    // most common stack; ties resolved by first occurrence, as Counter.most_common does
    let primary = "";
    let best = -1;
    for (const [s, c] of stackCounts) if (c > best) [primary, best] = [s, c];
    const nL = new Set(tensors.filter((t) => t.stack === primary).map((t) => t.layer)).size;
    if (facts.get("n_layers") === null) facts.add("n_layers", "Layers", nL, "inferred", `distinct layer indices under '${primary}'`);
    else if (facts.get("n_layers") !== nL) notes.push(`metadata declares ${facts.get("n_layers")} layers but tensors under '${primary}' cover ${nL}`);
  }

  const nHeads = facts.get("n_heads");
  const hidden = facts.get("hidden_size");
  if (facts.get("head_dim") === null && isInt(nHeads) && isInt(hidden) && nHeads > 0) facts.add("head_dim", "Head dim (K)", Math.floor(hidden / nHeads), "derived", "hidden_size / n_heads");
  const headDim = facts.get("head_dim");
  if (facts.get("head_dim_v") === null && headDim !== null) facts.add("head_dim_v", "Head dim (V)", headDim, "derived", "assumed equal to K head dim");

  if (facts.get("n_kv_heads") === null && isInt(headDim) && headDim > 0 && !packed) {
    const k = byRole("attention", "k").find((t) => t.shape.length === 2 && t.layer === 0);
    if (k) {
      const d = shapeDims(k)!;
      if (d[1] % headDim === 0) facts.add("n_kv_heads", "KV heads", d[1] / headDim, "inferred", `${k.name} output width / head_dim`);
    } else if (isInt(nHeads)) facts.add("n_kv_heads", "KV heads", nHeads, "inferred", "no K projection found; assumed equal to attention heads");
  }

  if (facts.get("ffn_size") === null && !packed) {
    const up = [...byRole("ffn", "up"), ...byRole("ffn", "gate")].find((t) => t.shape.length === 2);
    if (up) facts.add("ffn_size", "FFN size", shapeDims(up)![1], "inferred", `shape of ${up.name}`);
  }

  const nk = facts.get("n_kv_heads");
  const nh = facts.get("n_heads");
  if (isInt(nk) && isInt(nh) && nk > 0) {
    const kind = nk === nh ? "MHA" : nk === 1 ? "MQA" : `GQA (${nh % nk === 0 ? nh / nk : +(nh / nk).toPrecision(6)} query heads per KV head)`;
    facts.add("attention_type", "Attention type", kind, "derived", "n_heads vs n_kv_heads");
  }

  if (facts.get("tied_embeddings") === null && emb) {
    const hasOut = byRole("output", "lm_head").length > 0;
    facts.add("tied_embeddings", "Tied embeddings", !hasOut, "inferred", hasOut ? "separate output/lm_head tensor present" : "no separate output/lm_head tensor");
  }

  if (packed) {
    const est = estimateParamsFromDims((k) => facts.get(k));
    if (est) {
      facts.add("n_params_logical", "Parameters (logical, estimated)", est, "inferred", "computed from config dimensions");
      notes.push("checkpoint stores packed quantized weights: tensor element count is not the logical parameter count");
    }
  }
}

// ------------------------------------------------------------------ summaries

function dtypeBreakdown(tensors: TensorRecord[]): DTypeRow[] {
  const agg = new Map<string, DTypeRow>();
  for (const t of tensors) {
    const r = agg.get(t.dtype) ?? { dtype: t.dtype, n_tensors: 0, n_elements: 0, n_bytes: 0 };
    r.n_tensors++;
    r.n_elements += t.n_elements;
    r.n_bytes += t.n_bytes ?? 0;
    agg.set(t.dtype, r);
  }
  return [...agg.values()].sort((a, b) => b.n_bytes - a.n_bytes);
}

function effectiveBpw(tensors: TensorRecord[]): number | null {
  const s = tensors.filter((t) => t.n_bytes !== null);
  const n = s.reduce((a, t) => a + t.n_elements, 0);
  const b = s.reduce((a, t) => a + t.n_bytes!, 0);
  return n ? Math.round(((b * 8) / n) * 1000) / 1000 : null;
}

function dominantMatrixDtype(tensors: TensorRecord[]): string | null {
  const c = new Map<string, number>();
  for (const t of tensors) if (t.shape.length >= 2) c.set(t.dtype, (c.get(t.dtype) ?? 0) + t.n_elements);
  let best: string | null = null;
  let bv = -1;
  for (const [k, v] of c) if (v > bv) [best, bv] = [k, v];
  return best;
}

function makeTensor(name: string, shape: number[], order: "ggml" | "row_major", dtype: string, nEl: number, nBytes: number | null, offset: number | null, fi: number): TensorRecord {
  const c = classify(name);
  const row = shape.length ? (order === "ggml" ? shape[0] : shape[shape.length - 1]) : 1;
  return { name, shape, shape_order: order, row_length: row, dtype, n_elements: nEl, n_bytes: nBytes, offset, file_index: fi, stack: c.stack, layer: c.layer, component: c.component, role: c.role };
}

function jsonSafe(v: unknown): unknown {
  if (typeof v === "number" && !Number.isFinite(v)) return String(v);
  if (Array.isArray(v)) return v.map(jsonSafe);
  if (typeof v === "object" && v !== null) return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, jsonSafe(x)]));
  return v;
}

interface Partial {
  kind: ScanResult["kind"];
  name: string;
  path: string;
  files: SourceFile[];
  format: FormatInfo;
  errors: string[];
  warnings: string[];
  notes: string[];
  facts: Facts;
  family: string | null;
  tensors: TensorRecord[];
  metadata: MetadataEntry[];
  quant: QuantizationSummary;
  packed: boolean;
}

// ------------------------------------------------------------------ GGUF

async function scanGGUF(file: File, all: File[]): Promise<Partial> {
  const g0 = await readGGUF(file);
  const errors = [...g0.errors];
  const warnings = [...g0.warnings];
  const notes: string[] = [];
  const parsed: { file: File; g: GGUFFile }[] = [{ file, g: g0 }];

  const count = ggufGet(g0, "split.count");
  if (isInt(count) && count > 1) {
    const m = SPLIT_RE.exec(file.name);
    if (!m) errors.push(`split.count=${count} but the file name does not follow the '-NNNNN-of-NNNNN.gguf' pattern`);
    else {
      if (Number(m[2]) !== 1) errors.push("this is not the first shard of a split GGUF; select the -00001-of-NNNNN file");
      const byName = new Map(all.map((f) => [f.name, f]));
      for (let i = 2; i <= count; i++) {
        const nm = `${m[1]}-${String(i).padStart(5, "0")}-of-${String(count).padStart(5, "0")}.gguf`;
        const sf = byName.get(nm);
        if (!sf) {
          errors.push(`missing split shard ${nm} (select all shard files together)`);
          continue;
        }
        try {
          const gs = await readGGUF(sf);
          errors.push(...gs.errors.map((x) => `${nm}: ${x}`));
          warnings.push(...gs.warnings.map((x) => `${nm}: ${x}`));
          parsed.push({ file: sf, g: gs });
        } catch (e) {
          errors.push(`${nm}: ${(e as Error).message}`);
        }
      }
    }
  }

  const tensors: TensorRecord[] = [];
  const seen = new Set<string>();
  parsed.forEach(({ g }, fi) => {
    for (const t of g.tensors) {
      if (seen.has(t.name)) errors.push(`tensor '${t.name}' appears in more than one shard`);
      seen.add(t.name);
      tensors.push(makeTensor(t.name, t.shape, "ggml", t.ggml ? t.ggml.name : `unknown(${t.typeId})`, t.nElements, t.nBytes, g.dataOffset + t.offset, fi));
    }
  });
  const expected = ggufGet(g0, "split.tensors.count");
  if (isInt(expected) && expected !== tensors.length) errors.push(`split.tensors.count=${expected} but ${tensors.length} tensors were found across shards`);

  const facts = new Facts();
  const arch = factsFromGGUF(g0, facts, notes);
  const ft = ggufGet(g0, "general.file_type");
  const isQ = tensors.some((t) => GGML_BY_NAME.get(t.dtype)?.quantized);
  let label: string | null;
  let labelProv: Provenance;
  if (isInt(ft)) [label, labelProv] = [ftypeLabel(ft), "verified"];
  else {
    [label, labelProv] = [dominantMatrixDtype(tensors), "inferred"];
    if (label) notes.push("general.file_type missing; quantization label inferred from the dominant matrix dtype");
  }
  const qnotes: string[] = [];
  const qv = ggufGet(g0, "general.quantization_version");
  if (qv !== undefined) qnotes.push(`general.quantization_version = ${qv}`);
  if (label && !UNQUANTIZED_FTYPES.has(label.split(" ")[0]) && isQ) qnotes.push("llama.cpp quantization presets mix tensor types; see the dtype breakdown for the actual per-tensor types");

  const files: SourceFile[] = parsed.map(({ file: f, g }, i) => ({
    path: relPath(f), size_bytes: g.fileSize, role: parsed.length === 1 ? "gguf" : `gguf shard ${i + 1}/${parsed.length}`, parsed: true, note: null,
  }));
  const metadata: MetadataEntry[] = [...g0.metadata.entries()].map(([k, v]) => ({ key: k, type: v.type, value: jsonSafe(v.value), length: v.length, truncated: v.truncated }));
  const name = ggufGet(g0, "general.name");
  return {
    kind: "gguf",
    name: typeof name === "string" && name ? name : file.name.replace(/\.gguf$/i, ""),
    path: relPath(file),
    files,
    format: {
      name: "GGUF",
      version: `v${g0.version}`,
      details: { byte_order: g0.byteOrder, alignment: g0.alignment, tensor_count: tensors.length, metadata_count: g0.metadata.size, data_offset: g0.dataOffset, shards: parsed.length },
    },
    errors,
    warnings,
    notes,
    facts,
    family: arch,
    tensors,
    metadata,
    quant: { label, label_provenance: label ? labelProv : null, is_quantized: isQ, effective_bits_per_weight: effectiveBpw(tensors), packed_checkpoint: false, config: null, notes: qnotes },
    packed: false,
  };
}

// ------------------------------------------------------------------ safetensors / HF

function stTensors(shards: SafetensorsFile[], base = 0): TensorRecord[] {
  const out: TensorRecord[] = [];
  shards.forEach((sf, fi) => {
    for (const t of sf.tensors) out.push(makeTensor(t.name, t.shape, "row_major", t.dtype, t.nElements, t.end - t.begin, sf.dataOffset + t.begin, fi + base));
  });
  return out;
}

async function scanHF(files: File[]): Promise<Partial> {
  const hf = await readHFFolder(files);
  const notes: string[] = [];
  const facts = new Facts();
  const family = factsFromHF(hf.config, facts, notes);

  let qc = hf.config.quantization_config as Record<string, unknown> | undefined;
  if (typeof qc !== "object" || qc === null) {
    const tc = hf.config.text_config as Record<string, unknown> | undefined;
    qc = tc && typeof tc.quantization_config === "object" && tc.quantization_config !== null ? (tc.quantization_config as Record<string, unknown>) : undefined;
  }
  const method = qc ? String(qc.quant_method ?? "unknown").toLowerCase() : null;
  const packed = method !== null && PACKED.has(method);

  const sourceFiles: SourceFile[] = [{ path: relPath(hf.configFile), size_bytes: hf.configFile.size, role: "config", parsed: true, note: null }];
  if (hf.indexFile) sourceFiles.push({ path: relPath(hf.indexFile), size_bytes: hf.indexFile.size, role: "safetensors index", parsed: true, note: null });
  const tensors = stTensors(hf.shards.map((s) => s.st), sourceFiles.length);
  completeFacts(facts, tensors, notes, packed);
  for (const s of hf.shards) sourceFiles.push({ path: relPath(s.file), size_bytes: s.st.fileSize, role: "safetensors shard", parsed: true, note: null });
  for (const f of hf.sideFiles) sourceFiles.push({ path: relPath(f), size_bytes: f.size, role: "tokenizer/config", parsed: false, note: "present; not needed for inspection" });
  for (const u of hf.unparsed) sourceFiles.push({ path: relPath(u.file), size_bytes: u.file.size, role: "unparsed weights", parsed: false, note: u.reason });

  const qnotes: string[] = [];
  let label: string | null;
  let labelProv: Provenance;
  if (qc) {
    const bits = qc.bits ?? qc.w_bit ?? (qc.load_in_4bit ? "4" : qc.load_in_8bit ? "8" : null);
    label = bits ? `${method} ${bits}-bit` : method;
    labelProv = "verified";
    qnotes.push(`quantization_config.quant_method = ${method}`);
    if (packed) qnotes.push("weights are stored packed; per-tensor dtypes describe storage containers, not logical precision");
  } else [label, labelProv] = [dominantMatrixDtype(tensors), "derived"];
  const isQ = !!qc || tensors.some((t) => ["I8", "U8", "F8_E4M3", "F8_E5M2", "F4"].includes(t.dtype) && t.shape.length >= 2);

  const metadata: MetadataEntry[] = Object.entries(hf.config).map(([k, v]) => ({
    key: `config.${k}`, type: Array.isArray(v) ? "list" : v === null ? "NoneType" : typeof v === "object" ? "dict" : typeof v === "number" ? (Number.isInteger(v) ? "int" : "float") : typeof v === "boolean" ? "bool" : "str",
    value: jsonSafe(v), length: null, truncated: false,
  }));
  for (const s of hf.shards.slice(0, 1)) for (const [k, v] of Object.entries(s.st.metadata)) metadata.push({ key: `safetensors.__metadata__.${k}`, type: "str", value: v, length: null, truncated: false });

  const warnings = [...hf.warnings];
  if (hf.unparsed.length) warnings.push(`${hf.unparsed.length} file(s) in non-safetensors formats were listed but not opened`);
  const np = hf.config._name_or_path;
  const dirName = hf.dir.split("/").pop() || "model";
  return {
    kind: "hf_safetensors",
    name: typeof np === "string" && np ? np : dirName,
    path: hf.dir || dirName,
    files: sourceFiles,
    format: { name: "Hugging Face safetensors", version: null, details: { shards: hf.shards.length, index_file: !!hf.indexFile, tensor_count: tensors.length, side_files: hf.sideFiles.map((f) => f.name) } },
    errors: hf.errors,
    warnings,
    notes,
    facts,
    family,
    tensors,
    metadata,
    quant: {
      label, label_provenance: label ? labelProv : null, is_quantized: isQ,
      effective_bits_per_weight: packed ? null : effectiveBpw(tensors), packed_checkpoint: packed,
      config: qc ? (jsonSafe(qc) as Record<string, unknown>) : null, notes: qnotes,
    },
    packed,
  };
}

async function scanSingleSafetensors(file: File): Promise<Partial> {
  const sf = await readSafetensors(file);
  const tensors = stTensors([sf]);
  return {
    kind: "safetensors",
    name: file.name.replace(/\.safetensors$/i, ""),
    path: relPath(file),
    files: [{ path: relPath(file), size_bytes: sf.fileSize, role: "safetensors", parsed: true, note: null }],
    format: { name: "safetensors", version: null, details: { header_bytes: sf.headerSize, tensor_count: tensors.length } },
    errors: sf.errors,
    warnings: sf.warnings,
    notes: ["single safetensors file without config.json: architecture hyperparameters are inferred from tensor shapes only"],
    facts: new Facts(),
    family: null,
    tensors,
    metadata: Object.entries(sf.metadata).map(([k, v]) => ({ key: `__metadata__.${k}`, type: "str", value: v, length: null, truncated: false })),
    quant: { label: dominantMatrixDtype(tensors), label_provenance: "derived", is_quantized: false, effective_bits_per_weight: effectiveBpw(tensors), packed_checkpoint: false, config: null, notes: [] },
    packed: false,
  };
}

// ------------------------------------------------------------------ entry point

/** Scan files the user picked: a .gguf (plus split shards), a .safetensors, or a whole HF folder. */
export async function scanFiles(files: File[]): Promise<ScanResult> {
  if (!files.length) throw new ScanError("no files selected");
  let data: Partial;
  let redirect: string | null = null;
  try {
    const config = findConfig(files);
    const ggufs = files.filter((f) => suffix(f.name) === ".gguf");
    if (config) {
      if (files.length === 1) throw new ScanError("config.json alone has no weights: pick the whole model folder instead");
      data = await scanHF(files);
      if (files.some((f) => suffix(f.name) === ".gguf")) redirect = "folder also contains GGUF files; they were not part of this scan";
    } else if (ggufs.length) {
      const first = ggufs.find((f) => /-00001-of-\d{5}\.gguf$/i.test(f.name)) ?? ggufs[0];
      if (!(await isGGUF(first))) throw new FormatError("file has a .gguf extension but does not start with the GGUF magic bytes");
      if (ggufs.length > 1 && !SPLIT_RE.test(first.name)) redirect = `several GGUF files selected; scanned ${first.name}`;
      data = await scanGGUF(first, files);
    } else if (files.length === 1) {
      const f = files[0];
      const s = suffix(f.name);
      if (await isGGUF(f)) data = await scanGGUF(f, files);
      else if (s === ".safetensors") data = await scanSingleSafetensors(f);
      else if (PICKLE_SUFFIXES.has(s))
        throw new UnsupportedFormatError(
          `${s} files are Python pickles. Model Surgery never opens pickle files because loading them can execute arbitrary code. Convert the model to safetensors with trusted tooling first.`,
        );
      else if (s in UNSUPPORTED) throw new UnsupportedFormatError(UNSUPPORTED[s]);
      else throw new UnsupportedFormatError("unrecognised file type (supported: .gguf, .safetensors, Hugging Face model folder)");
    } else if (files.some((f) => suffix(f.name) === ".safetensors")) {
      throw new ScanError("several safetensors files but no config.json: pick the whole Hugging Face folder, or a single .safetensors file");
    } else {
      throw new ScanError("no supported model files found (expected .gguf, .safetensors, or a folder with config.json)");
    }
  } catch (e) {
    if (e instanceof ScanError) throw e;
    throw new ScanError((e as Error).message);
  }

  if (data.kind !== "hf_safetensors") completeFacts(data.facts, data.tensors, data.notes, data.packed);
  if (redirect) data.notes.unshift(redirect);
  const arch = buildArchitecture(data.tensors, data.family, data.family ? "verified" : null);
  arch.notes = [...data.notes, ...arch.notes];
  return {
    id: crypto.randomUUID().replace(/-/g, ""),
    scanned_at: new Date().toISOString().replace(/\.\d{3}Z$/, "+00:00"),
    app_version: APP_VERSION,
    kind: data.kind,
    name: data.name,
    path: data.path,
    files: data.files,
    total_size_bytes: data.files.reduce((a, f) => a + f.size_bytes, 0),
    format: data.format,
    integrity: { valid: data.errors.length === 0, errors: data.errors, warnings: data.warnings },
    hyperparameters: data.facts.ordered(),
    quantization: data.quant,
    dtype_breakdown: dtypeBreakdown(data.tensors),
    architecture: arch,
    metadata: data.metadata,
    tensors: data.tensors,
  };
}

export function factValue(scan: ScanResult, key: string): unknown {
  return scan.hyperparameters.find((f) => f.key === key)?.value ?? null;
}

import { describe, expect, it } from "vitest";
import { MemFile, gguf, ggufBytes, llamaLike, safetensorsBytes, tensorSize } from "../test/fixtures";
import type { MemoryRequest, ScanResult } from "../types";
import { classify } from "./architecture";
import { buildArchitectureGraph } from "./graph";
import { estimateMemory } from "./memory";
import { compare, tensorsCsv } from "./reports";
import { factValue, scanFiles } from "./scanner";

const fact = (s: ScanResult, k: string) => s.hyperparameters.find((f) => f.key === k);
const req = (o: Partial<MemoryRequest> = {}): MemoryRequest => ({
  weight_format: "current", context_length: 4096, batch_size: 1, kv_cache_type: "f16", flash_attention: true, n_ubatch: 512, device_id: null, custom_device: null, ...o,
});

describe("GGUF metadata inspection", () => {
  it("reads hyperparameters with provenance and derives what it computes", async () => {
    const s = await scanFiles([gguf("tiny.gguf")]);
    expect(s.kind).toBe("gguf");
    expect(s.integrity.valid).toBe(true);
    expect(fact(s, "n_layers")).toMatchObject({ value: 2, provenance: "verified", source: "llama.block_count" });
    expect(fact(s, "n_heads")?.value).toBe(4);
    expect(fact(s, "n_kv_heads")?.value).toBe(2);
    expect(fact(s, "head_dim")).toMatchObject({ value: 16, provenance: "derived" });
    expect(fact(s, "attention_type")?.value).toMatch(/^GQA \(2 query heads per KV head\)/);
    expect(s.quantization.label).toBe("F16");
    expect(s.quantization.label_provenance).toBe("verified");
  });

  it("parameter count is the exact sum of tensor element counts", async () => {
    const { tensors } = llamaLike();
    const expected = tensors.reduce((a, [, shape]) => a + shape.reduce((x, y) => x * y, 1), 0);
    const s = await scanFiles([gguf("tiny.gguf")]);
    expect(factValue(s, "n_params")).toBe(expected);
    expect(s.tensors.reduce((a, t) => a + (t.n_bytes ?? 0), 0)).toBe(tensors.reduce((a, [, shape, dt]) => a + tensorSize(dt, shape), 0));
  });

  it("keeps GGML shape order and per-tensor dtypes", async () => {
    const s = await scanFiles([gguf("q.gguf", { wtype: "Q8_0", ftype: 7, hidden: 64, ffn: 128 })]);
    const k = s.tensors.find((t) => t.name === "blk.0.attn_k.weight")!;
    expect(k).toMatchObject({ shape: [64, 32], shape_order: "ggml", dtype: "Q8_0", n_elements: 2048, layer: 0, component: "attention", role: "k" });
    expect(s.quantization.label).toBe("Q8_0");
    expect(s.tensors.find((t) => t.name === "blk.0.attn_norm.weight")!.dtype).toBe("F32");
  });

  it("reads only the header, not tensor data", async () => {
    const { kv, tensors } = llamaLike({ hidden: 512, ffn: 2048, vocab: 4096, nLayers: 4 });
    const f = new MemFile("big.gguf", ggufBytes(kv, tensors));
    await scanFiles([f]);
    expect(f.size).toBeGreaterThan(10_000_000);
    expect(f.bytesRead).toBeLessThan(f.size); // bounded by the 8 MB header window, never the whole file
    expect(f.bytesRead).toBeLessThanOrEqual(8 * 1024 * 1024 + 4);
  });

  it("ignores non-numeric hyperparameters instead of computing with them (audit B2)", async () => {
    const { kv, tensors } = llamaLike();
    const bad = kv.map((x) => (x[0] === "llama.attention.head_count" ? (["llama.attention.head_count", "str", "ab"] as typeof x) : x));
    const s = await scanFiles([new MemFile("bad.gguf", ggufBytes(bad, tensors))]);
    expect(fact(s, "n_heads")).toBeUndefined();
    expect(s.architecture.notes).toContain("llama.attention.head_count has a non-numeric value (str); ignored");
    expect(estimateMemory(s, req({ flash_attention: false })).total_bytes).toBeGreaterThan(0);
  });
});

describe("invalid and unsupported inputs", () => {
  it("reports a truncated header honestly", async () => {
    const { kv, tensors } = llamaLike();
    const full = ggufBytes(kv, tensors);
    const headerOnly = full.slice(0, 200);
    await expect(scanFiles([new MemFile("cut.gguf", headerOnly)])).rejects.toThrow(/truncated/);
  });

  it("flags tensor data that extends past the end of the file", async () => {
    const { kv, tensors } = llamaLike();
    const s = await scanFiles([new MemFile("short.gguf", ggufBytes(kv, tensors, { truncate: 64 }))]);
    expect(s.integrity.valid).toBe(false);
    expect(s.integrity.errors.join(" ")).toMatch(/beyond|exceeds|past|out of bounds/i);
  });

  it("rejects a .gguf without the GGUF magic", async () => {
    await expect(scanFiles([new MemFile("fake.gguf", new TextEncoder().encode("not a model at all"))])).rejects.toThrow(/magic/);
  });

  it("refuses pickle formats without opening them", async () => {
    const f = new MemFile("pytorch_model.bin", new Uint8Array([0x80, 0x02]));
    await expect(scanFiles([f])).rejects.toThrow(/pickle/i);
    expect(f.bytesRead).toBeLessThanOrEqual(8); // at most the magic-byte sniff
  });

  it("names unsupported formats", async () => {
    await expect(scanFiles([new MemFile("model.onnx", new Uint8Array(16))])).rejects.toThrow(/ONNX/i);
  });

  it("rejects a safetensors header that claims more bytes than the file has", async () => {
    const bytes = safetensorsBytes({ w: ["F32", [4, 4]] });
    new DataView(bytes.buffer).setBigUint64(0, 10_000n, true);
    await expect(scanFiles([new MemFile("x.safetensors", bytes)])).rejects.toThrow(/exceeds file size/);
  });

  it("rejects duplicate keys in a safetensors header", async () => {
    const raw = '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}';
    const bytes = safetensorsBytes({ w: ["F32", [1]] }, undefined, raw);
    await expect(scanFiles([new MemFile("dup.safetensors", bytes)])).rejects.toThrow(/duplicate/);
  });
});

describe("split GGUF (audit F2)", () => {
  const shard = (no: number, count: number, tensors: ReturnType<typeof llamaLike>["tensors"], kv: ReturnType<typeof llamaLike>["kv"]) =>
    new MemFile(
      `m-${String(no).padStart(5, "0")}-of-${String(count).padStart(5, "0")}.gguf`,
      ggufBytes(
        [...(no === 1 ? kv : [["general.architecture", "str", "llama"] as const]), ["split.no", "u32", no - 1], ["split.count", "u32", count], ["split.tensors.count", "i32", llamaLike().tensors.length]] as never,
        tensors,
      ),
    );
  const { kv, tensors } = llamaLike();
  const a = tensors.slice(0, 12);
  const b = tensors.slice(12);

  it("scans all shards when they are selected together", async () => {
    const s = await scanFiles([shard(2, 2, b, kv), shard(1, 2, a, kv)]);
    expect(s.integrity.errors).toEqual([]);
    expect(s.tensors).toHaveLength(tensors.length);
    expect(s.files.map((f) => f.role)).toEqual(["gguf shard 1/2", "gguf shard 2/2"]);
  });

  it("does not duplicate a lone second shard and reports the missing first shard", async () => {
    const s = await scanFiles([shard(2, 2, b, kv)]);
    expect(s.tensors).toHaveLength(b.length);
    expect(s.integrity.errors.some((e) => e.includes("appears in more than one shard"))).toBe(false);
    expect(s.integrity.errors).toContain("missing split shard m-00001-of-00002.gguf (select all shard files together)");
  });
});

describe("Hugging Face folders", () => {
  const folder = () => {
    const cfg = { architectures: ["LlamaForCausalLM"], model_type: "llama", hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 256, tie_word_embeddings: true };
    const t1: Record<string, [string, number[]]> = { "model.embed_tokens.weight": ["BF16", [256, 64]] };
    const t2: Record<string, [string, number[]]> = {};
    for (let i = 0; i < 2; i++) {
      const tgt = i === 0 ? t1 : t2;
      tgt[`model.layers.${i}.input_layernorm.weight`] = ["BF16", [64]];
      tgt[`model.layers.${i}.self_attn.q_proj.weight`] = ["BF16", [64, 64]];
      tgt[`model.layers.${i}.self_attn.k_proj.weight`] = ["BF16", [32, 64]];
      tgt[`model.layers.${i}.post_attention_layernorm.weight`] = ["BF16", [64]];
      tgt[`model.layers.${i}.mlp.up_proj.weight`] = ["BF16", [128, 64]];
    }
    t2["model.norm.weight"] = ["BF16", [64]];
    const index = { weight_map: { ...Object.fromEntries(Object.keys(t1).map((k) => [k, "model-00001-of-00002.safetensors"])), ...Object.fromEntries(Object.keys(t2).map((k) => [k, "model-00002-of-00002.safetensors"])) } };
    const enc = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));
    return [
      new MemFile("config.json", enc(cfg), "m/config.json"),
      new MemFile("model.safetensors.index.json", enc(index), "m/model.safetensors.index.json"),
      new MemFile("model-00001-of-00002.safetensors", safetensorsBytes(t1), "m/model-00001-of-00002.safetensors"),
      new MemFile("model-00002-of-00002.safetensors", safetensorsBytes(t2), "m/model-00002-of-00002.safetensors"),
    ];
  };

  it("merges sharded checkpoints and config facts", async () => {
    const s = await scanFiles(folder());
    expect(s.kind).toBe("hf_safetensors");
    expect(s.integrity.valid).toBe(true);
    expect(fact(s, "n_layers")).toMatchObject({ value: 2, source: "config.json:num_hidden_layers" });
    expect(fact(s, "tied_embeddings")).toMatchObject({ value: true, provenance: "verified" });
    expect(s.tensors.find((t) => t.name === "model.layers.1.self_attn.k_proj.weight")).toMatchObject({ shape: [32, 64], shape_order: "row_major", file_index: 3 }); // index into files[] (config, index, shard 1, shard 2)
    expect(s.architecture.stacks[0]).toMatchObject({ prefix: "model.layers", n_layers: 2 });
  });

  it("reads DeepSeek MLA head dimensions from the config (audit B9)", async () => {
    const files = folder();
    const cfg = JSON.parse(new TextDecoder().decode(new Uint8Array(await files[0].arrayBuffer())));
    Object.assign(cfg, { model_type: "deepseek_v3", qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128 });
    files[0] = new MemFile("config.json", new TextEncoder().encode(JSON.stringify(cfg)), "m/config.json");
    const s = await scanFiles(files);
    expect(factValue(s, "head_dim")).toBe(192);
    expect(factValue(s, "head_dim_v")).toBe(128);
  });
});

describe("tensor-name classification", () => {
  it.each([
    ["blk.0.attn_q.weight", "attention", "q"],
    ["model.layers.3.self_attn.o_proj.weight", "attention", "o"],
    ["model.layers.3.post_attention_layernorm.weight", "norm", "ffn_norm"],
    ["blk.3.post_attention_norm.weight", "norm", "post_attn_norm"],
    ["blk.0.ffn_gate_inp.weight", "router", "router"],
    ["blk.0.ffn_up_exps.weight", "experts", "up"],
    ["blk.0.ffn_up_shexp.weight", "ffn", "up"],
    ["model.layers.3.mlp.shared_experts.gate_proj.weight", "ffn", "gate"],
    ["model.layers.3.mlp.experts.7.gate_proj.weight", "experts", "gate"],
    ["backbone.layers.1.mixer.q_proj.weight", "attention", "q"],
    ["backbone.layers.0.mixer.in_proj.weight", "ssm", null],
    ["model.layers.0.mamba.A_log", "ssm", null],
    ["blk.0.something_new.weight", "other", null],
  ])("%s → %s", (name, component, role) => {
    const c = classify(name);
    expect(c.component).toBe(component);
    if (role !== null) expect(c.role).toBe(role);
  });

  it("classifies globals", () => {
    expect(classify("token_embd.weight")).toMatchObject({ component: "embedding", role: "token", layer: null });
    expect(classify("output.weight")).toMatchObject({ component: "output", role: "lm_head" });
    expect(classify("output_norm.weight")).toMatchObject({ component: "norm", role: "final" });
  });
});

describe("memory estimation", () => {
  it("KV cache follows layers × tokens × kv_heads × (k+v) × bytes, labelled as an estimate", async () => {
    const s = await scanFiles([gguf("tiny.gguf")]);
    const est = estimateMemory(s, req({ context_length: 1000, batch_size: 2 }));
    const kv = est.inference.find((c) => c.key === "kv_cache")!;
    expect(kv.bytes).toBe(2 * 2000 * 2 * (16 + 16) * 2);
    expect(kv.formula).toBe("2 layers × 2,000 tokens × 2 KV heads × (16+16) dims × 2 B");
    expect(est.is_measurement).toBe(false);
    expect(est.disclaimer).toMatch(/not measurements/);
    const q8 = estimateMemory(s, req({ context_length: 1000, batch_size: 2, kv_cache_type: "q8_0" })).inference.find((c) => c.key === "kv_cache")!;
    expect(q8.bytes).toBe(Math.floor(2 * 2000 * 2 * 32 * (34 / 32)));
  });

  it("weight size for the current format equals the bytes stored in the file", async () => {
    const s = await scanFiles([gguf("tiny.gguf")]);
    const cur = estimateMemory(s, req()).weights.find((w) => w.id === "current")!;
    expect(cur.bytes).toBe(s.tensors.reduce((a, t) => a + (t.n_bytes ?? 0), 0));
  });

  it("never reports that a model fits just because the weights are smaller than the device", async () => {
    const s = await scanFiles([gguf("tiny.gguf", { vocab: 2048 })]);
    const weights = s.tensors.reduce((a, t) => a + (t.n_bytes ?? 0), 0);
    const device = { id: "x", name: "x", memory_kind: "unified" as const, capacity_bytes: weights * 2, reserved_bytes: 0, runtime_overhead_bytes: 300 * 2 ** 20, host_ram_bytes: null, bandwidth_gbps: null, assumptions: [] };
    const fit = estimateMemory(s, req({ custom_device: device })).fit!;
    expect(fit.required_bytes).toBeGreaterThan(weights * 2);
    expect(fit.verdict).toBe("does_not_fit");
  });

  it("all layers on the GPU is not reported as partial offload (audit B13)", async () => {
    const s = await scanFiles([gguf("tiny.gguf")]);
    const total = estimateMemory(s, req({ context_length: 512 })).total_bytes;
    const gpu = { id: "g", name: "g", memory_kind: "discrete_gpu" as const, capacity_bytes: total - 1, reserved_bytes: 0, runtime_overhead_bytes: 200 * 2 ** 20, host_ram_bytes: 2 ** 34, bandwidth_gbps: null, assumptions: [] };
    const fit = estimateMemory(s, req({ context_length: 512, custom_device: gpu })).fit!;
    expect(fit.gpu_layers).toBe(fit.total_layers);
    expect(fit.verdict).toBe("tight");
  });
});

describe("architecture graph", () => {
  it("builds a pre-norm residual block from real tensors and marks virtual nodes as illustrative", async () => {
    const s = await scanFiles([gguf("tiny.gguf")]);
    const g = buildArchitectureGraph(s);
    expect(g.schema).toBe("model-surgery/architecture-graph@1");
    expect(g.n_layers).toBe(2);
    const L0 = g.layers[0];
    expect(L0.layout).toBe("sequential_prenorm");
    const ids = L0.nodes.map((n) => n.id);
    expect(ids).toEqual(["L0.in", "L0.attn_norm", "L0.attn", "L0.add1", "L0.ffn_norm", "L0.ffn", "L0.add2", "L0.out"]);
    expect(L0.edges).toContainEqual({ from: "L0.in", to: "L0.add1", kind: "residual" });
    expect(L0.edges).toContainEqual({ from: "L0.add1", to: "L0.add2", kind: "residual" });
    const attn = L0.nodes.find((n) => n.id === "L0.attn")!;
    expect(attn.tensors).toEqual(["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight"]);
    expect(attn.illustrative).toBe(false);
    expect(attn.dims.find((d) => d.label === "KV heads")).toEqual({ label: "KV heads", value: "2", provenance: "verified" });
    expect(L0.nodes.filter((n) => n.illustrative).map((n) => n.kind)).toEqual(["input", "residual_add", "residual_add", "input"]);
    expect(g.head.map((n) => n.id)).toEqual(["tokens", "embedding"]);
    expect(g.tail.map((n) => n.id)).toEqual(["final_norm", "lm_head", "logits"]);
    expect(g.unrepresented).toEqual([]);
  });

  it("represents tied output heads and unknown tensors explicitly", async () => {
    const s = await scanFiles([gguf("tied.gguf", { tied: true, extraTensors: [["blk.0.mystery.weight", [64], "F32"]] })]);
    const g = buildArchitectureGraph(s);
    const head = g.tail.find((n) => n.id === "lm_head")!;
    expect(head.illustrative).toBe(true);
    expect(head.label).toMatch(/tied/);
    expect(g.unrepresented).toEqual(["blk.0.mystery.weight"]);
  });

  it("models MoE layers with router, routed experts and shared expert", async () => {
    const extra: [string, number[], string][] = [];
    const { kv, tensors } = llamaLike({ nLayers: 1 });
    const moe = tensors.filter((t) => !/ffn_(gate|up|down)\.weight/.test(t[0]));
    moe.push(["blk.0.ffn_gate_inp.weight", [64, 4], "F32"], ["blk.0.ffn_up_exps.weight", [64, 32, 4], "F16"], ["blk.0.ffn_down_exps.weight", [32, 64, 4], "F16"], ["blk.0.ffn_up_shexp.weight", [64, 32], "F16"]);
    const s = await scanFiles([new MemFile("moe.gguf", ggufBytes([...kv, ["llama.expert_count", "u32", 4], ["llama.expert_used_count", "u32", 2]], [...moe, ...extra]))]);
    const L0 = buildArchitectureGraph(s).layers[0];
    expect(L0.nodes.map((n) => n.kind)).toEqual(expect.arrayContaining(["router", "experts", "ffn"]));
    expect(L0.edges).toContainEqual({ from: "L0.router", to: "L0.experts", kind: "data" });
    expect(L0.nodes.find((n) => n.kind === "experts")!.dims.find((d) => d.label === "experts per token")?.value).toBe("2");
  });
});

describe("reports", () => {
  it("neutralises spreadsheet formulas in CSV export", async () => {
    const s = await scanFiles([gguf("tiny.gguf", { extraTensors: [["=HYPERLINK(1).weight", [4], "F32"]] })]);
    const csv = tensorsCsv(s);
    expect(csv).toContain("'=HYPERLINK(1).weight");
    expect(csv).not.toMatch(/^=HYPERLINK/m);
  });

  it("compares two scans and counts dtype changes", async () => {
    const a = await scanFiles([gguf("a.gguf")]);
    const b = await scanFiles([gguf("b.gguf", { wtype: "Q8_0", ftype: 7 })]);
    const c = compare(a, b, req());
    expect(c.tensors_dtype_changed).toBe(16); // 2 layers × 7 matrices + embedding + output
    expect(c.dtype_changes).toEqual([{ from: "F16", to: "Q8_0", count: 16 }]);
  });
});

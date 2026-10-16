// Cross-engine parity: the browser edition (this TypeScript engine) and the local app (Python engine) must
// produce the same scan and the same estimates for the same bytes. Runs the backend from ../.venv; skipped
// when that interpreter is not available (set MODEL_SURGERY_PYTHON to point at another one).

import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { MemFile, ggufBytes, llamaLike, safetensorsBytes } from "../test/fixtures";
import type { MemoryRequest, ScanResult } from "../types";
import { estimateMemory } from "./memory";
import { scanFiles } from "./scanner";

const candidates = [process.env.MODEL_SURGERY_PYTHON, resolve("../.venv/Scripts/python.exe"), resolve("../.venv/bin/python")].filter(Boolean) as string[];
const PY = candidates.find((p) => existsSync(p));

const PY_SCRIPT = `
import json, sys
from model_surgery.config import Settings
from model_surgery.memory import MemoryRequest, estimate_memory
from model_surgery.scanner import scan_path
import tempfile
s = scan_path(sys.argv[1], Settings(data_dir=__import__("pathlib").Path(tempfile.mkdtemp())))
reqs = [MemoryRequest(**r) for r in json.loads(sys.argv[2])]
print(json.dumps({"scan": s.model_dump(mode="json"), "memory": [estimate_memory(s, r).model_dump(mode="json") for r in reqs]}))
`;

const REQS: MemoryRequest[] = [
  { weight_format: "current", context_length: 4096, batch_size: 1, kv_cache_type: "f16", flash_attention: true, n_ubatch: 512, device_id: "jetson-orin-nano-8gb", custom_device: null },
  { weight_format: "q4_k_m", context_length: 32768, batch_size: 2, kv_cache_type: "q8_0", flash_attention: false, n_ubatch: 256, device_id: "laptop-gpu-8gb", custom_device: null },
];

/** Drop fields that legitimately differ between runs or editions (ids, timestamps, absolute paths). */
function normalize(s: ScanResult) {
  return {
    kind: s.kind,
    name: s.name,
    total_size_bytes: s.total_size_bytes,
    format: { name: s.format.name, version: s.format.version },
    integrity: s.integrity,
    hyperparameters: s.hyperparameters,
    quantization: s.quantization,
    dtype_breakdown: s.dtype_breakdown,
    architecture: s.architecture,
    tensors: s.tensors,
    files: s.files.map((f) => ({ size_bytes: f.size_bytes, role: f.role, parsed: f.parsed })),
  };
}

const FIXTURES: [string, () => Uint8Array][] = [
  ["llama-f16.gguf", () => { const { kv, tensors } = llamaLike(); return ggufBytes(kv, tensors); }],
  ["llama-q8-tied.gguf", () => { const { kv, tensors } = llamaLike({ wtype: "Q8_0", ftype: 7, tied: true, nLayers: 3 }); return ggufBytes(kv, tensors); }],
  ["llama-mqa-truncated.gguf", () => { const { kv, tensors } = llamaLike({ nKv: 1 }); return ggufBytes(kv, tensors, { truncate: 100 }); }],
  ["unknown-tensors.gguf", () => { const { kv, tensors } = llamaLike({ extraTensors: [["blk.0.mystery.weight", [64], "F32"], ["rope_freqs.weight", [8], "F32"]] }); return ggufBytes(kv, tensors); }],
  ["bad-hparam.gguf", () => { const { kv, tensors } = llamaLike(); return ggufBytes(kv.map((x) => (x[0] === "llama.block_count" ? (["llama.block_count", "str", "two"] as typeof x) : x)), tensors); }],
  ["single.safetensors", () => safetensorsBytes({ "model.embed_tokens.weight": ["F16", [256, 64]], "model.layers.0.self_attn.q_proj.weight": ["F16", [64, 64]], "model.layers.0.mlp.up_proj.weight": ["BF16", [128, 64]] }, { format: "pt" })],
];

describe.skipIf(!PY)("Python ↔ TypeScript engine parity", () => {
  const dir = mkdtempSync(join(tmpdir(), "ms-parity-"));
  it.each(FIXTURES)("%s", async (name, make) => {
    const bytes = make();
    const path = join(dir, name);
    writeFileSync(path, bytes);
    const out = JSON.parse(execFileSync(PY!, ["-c", PY_SCRIPT, path, JSON.stringify(REQS)], { encoding: "utf8", maxBuffer: 64 << 20 }));
    const py = out.scan as ScanResult;
    const ts = await scanFiles([new MemFile(name, bytes)]);
    expect(normalize(ts)).toEqual(normalize(py));
    REQS.forEach((r, i) => {
      const a = estimateMemory(ts, r);
      const b = out.memory[i];
      expect({ ...a, weights: a.weights.map((w) => ({ ...w, bits_per_weight: Math.round(w.bits_per_weight * 100) })) }).toEqual({
        ...b,
        weights: b.weights.map((w: { bits_per_weight: number }) => ({ ...w, bits_per_weight: Math.round(w.bits_per_weight * 100) })),
      });
    });
  });
});

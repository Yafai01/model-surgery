// Synthetic GGUF / safetensors builders for the engine tests. Mirrors backend/tests/helpers.py so the
// two engines can be fed byte-identical files. Tensor data is all zeros: only headers matter.

import { GGML_BY_NAME } from "../engine/ggml";
import type { FileLike } from "../engine/source";

export type KVType = "u8" | "u32" | "i32" | "u64" | "f32" | "bool" | "str" | "arr_str" | "arr_i32";
export type KV = [key: string, type: KVType, value: unknown];
export type TensorSpec = [name: string, shape: number[], dtype: string];

const TID: Record<string, number> = { u8: 0, i8: 1, u16: 2, i16: 3, u32: 4, i32: 5, f32: 6, bool: 7, str: 8, arr: 9, u64: 10, i64: 11, f64: 12 };

class Writer {
  private buf = new Uint8Array(1024);
  private view = new DataView(this.buf.buffer);
  len = 0;
  private grow(n: number) {
    if (this.len + n <= this.buf.length) return;
    const next = new Uint8Array(Math.max(this.buf.length * 2, this.len + n));
    next.set(this.buf);
    this.buf = next;
    this.view = new DataView(next.buffer);
  }
  u8(v: number) { this.grow(1); this.view.setUint8(this.len, v); this.len += 1; }
  u32(v: number) { this.grow(4); this.view.setUint32(this.len, v, true); this.len += 4; }
  i32(v: number) { this.grow(4); this.view.setInt32(this.len, v, true); this.len += 4; }
  u64(v: number | bigint) { this.grow(8); this.view.setBigUint64(this.len, BigInt(v), true); this.len += 8; }
  f32(v: number) { this.grow(4); this.view.setFloat32(this.len, v, true); this.len += 4; }
  bytes(b: Uint8Array) { this.grow(b.length); this.buf.set(b, this.len); this.len += b.length; }
  str(s: string) { const b = new TextEncoder().encode(s); this.u64(b.length); this.bytes(b); }
  zeros(n: number) { this.grow(n); this.len += n; }
  out() { return this.buf.slice(0, this.len); }
}

export function tensorSize(dtype: string, shape: number[]): number {
  const t = GGML_BY_NAME.get(dtype);
  if (!t) throw new Error(`unknown dtype ${dtype}`);
  const n = shape.reduce((a, b) => a * b, 1);
  return (n / t.blockSize) * t.typeSize;
}

function writeValue(w: Writer, type: KVType, v: unknown) {
  switch (type) {
    case "u8": return w.u8(v as number);
    case "u32": return w.u32(v as number);
    case "i32": return w.i32(v as number);
    case "u64": return w.u64(v as number | bigint);
    case "f32": return w.f32(v as number);
    case "bool": return w.u8(v ? 1 : 0);
    case "str": return w.str(v as string);
    case "arr_str":
      w.u32(TID.str); w.u64((v as string[]).length);
      for (const s of v as string[]) w.str(s);
      return;
    case "arr_i32":
      w.u32(TID.i32); w.u64((v as number[]).length);
      for (const x of v as number[]) w.i32(x);
      return;
  }
}

export function ggufBytes(kv: KV[], tensors: TensorSpec[], opts: { alignment?: number; truncate?: number } = {}): Uint8Array {
  const align = opts.alignment ?? 32;
  const w = new Writer();
  w.bytes(new TextEncoder().encode("GGUF"));
  w.u32(3);
  w.u64(tensors.length);
  w.u64(kv.length);
  for (const [key, type, v] of kv) {
    w.str(key);
    w.u32(type.startsWith("arr") ? TID.arr : TID[type]);
    writeValue(w, type, v);
  }
  let cursor = 0;
  for (const [name, shape, dtype] of tensors) {
    w.str(name);
    w.u32(shape.length);
    for (const d of shape) w.u64(d);
    w.u32(GGML_BY_NAME.get(dtype)!.id);
    w.u64(cursor);
    const size = tensorSize(dtype, shape);
    cursor += size + ((align - (size % align)) % align);
  }
  w.zeros((align - (w.len % align)) % align);
  w.zeros(cursor);
  const out = w.out();
  return opts.truncate ? out.slice(0, out.length - opts.truncate) : out;
}

export interface LlamaOpts {
  nLayers?: number;
  hidden?: number;
  nHeads?: number;
  nKv?: number;
  ffn?: number;
  vocab?: number;
  wtype?: string;
  ftype?: number;
  tied?: boolean;
  extraKv?: KV[];
  extraTensors?: TensorSpec[];
}

/** Same layout as helpers.llama_like_gguf. */
export function llamaLike(o: LlamaOpts = {}): { kv: KV[]; tensors: TensorSpec[] } {
  const { nLayers = 2, hidden = 64, nHeads = 4, nKv = 2, ffn = 128, vocab = 256, wtype = "F16", ftype = 1, tied = false } = o;
  const hd = hidden / nHeads;
  const kv: KV[] = [
    ["general.architecture", "str", "llama"],
    ["general.name", "str", "tiny-llama"],
    ["llama.block_count", "u32", nLayers],
    ["llama.embedding_length", "u32", hidden],
    ["llama.feed_forward_length", "u32", ffn],
    ["llama.attention.head_count", "u32", nHeads],
    ["llama.attention.head_count_kv", "u32", nKv],
    ["llama.context_length", "u32", 2048],
    ["llama.rope.freq_base", "f32", 10000.0],
    ["general.file_type", "u32", ftype],
    ["tokenizer.ggml.tokens", "arr_str", Array.from({ length: vocab }, (_, i) => `t${i}`)],
    ...(o.extraKv ?? []),
  ];
  const tensors: TensorSpec[] = [["token_embd.weight", [hidden, vocab], wtype]];
  for (let i = 0; i < nLayers; i++) {
    tensors.push(
      [`blk.${i}.attn_norm.weight`, [hidden], "F32"],
      [`blk.${i}.attn_q.weight`, [hidden, hidden], wtype],
      [`blk.${i}.attn_k.weight`, [hidden, nKv * hd], wtype],
      [`blk.${i}.attn_v.weight`, [hidden, nKv * hd], wtype],
      [`blk.${i}.attn_output.weight`, [hidden, hidden], wtype],
      [`blk.${i}.ffn_norm.weight`, [hidden], "F32"],
      [`blk.${i}.ffn_gate.weight`, [hidden, ffn], wtype],
      [`blk.${i}.ffn_up.weight`, [hidden, ffn], wtype],
      [`blk.${i}.ffn_down.weight`, [ffn, hidden], wtype],
    );
  }
  tensors.push(["output_norm.weight", [hidden], "F32"]);
  if (!tied) tensors.push(["output.weight", [hidden, vocab], wtype]);
  tensors.push(...(o.extraTensors ?? []));
  return { kv, tensors };
}

const ST_BITS: Record<string, number> = { F64: 64, F32: 32, F16: 16, BF16: 16, I64: 64, I32: 32, I16: 16, I8: 8, U8: 8, BOOL: 8 };

export function safetensorsBytes(tensors: Record<string, [string, number[]]>, metadata?: Record<string, string>, rawHeader?: string): Uint8Array {
  const header: Record<string, unknown> = {};
  if (metadata) header.__metadata__ = metadata;
  let cursor = 0;
  for (const [name, [dtype, shape]] of Object.entries(tensors)) {
    const n = shape.reduce((a, b) => a * b, 1);
    const size = Math.ceil((n * ST_BITS[dtype]) / 8);
    header[name] = { dtype, shape, data_offsets: [cursor, cursor + size] };
    cursor += size;
  }
  const hb = new TextEncoder().encode(rawHeader ?? JSON.stringify(header));
  const out = new Uint8Array(8 + hb.length + cursor);
  new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true);
  out.set(hb, 8);
  return out;
}

/** An in-memory File stand-in with byte-range reads that records how many bytes were read. */
export class MemFile implements FileLike {
  bytesRead = 0;
  constructor(
    readonly name: string,
    private data: Uint8Array,
    readonly webkitRelativePath = "",
  ) {}
  get size() {
    return this.data.length;
  }
  slice(start: number, end: number) {
    const part = this.data.slice(start, end);
    return {
      arrayBuffer: async () => {
        this.bytesRead += part.length;
        return part.buffer;
      },
    };
  }
  async arrayBuffer() {
    this.bytesRead += this.data.length;
    return this.data.slice().buffer;
  }
}

export const gguf = (name: string, o: LlamaOpts = {}, path = "") => {
  const { kv, tensors } = llamaLike(o);
  return new MemFile(name, ggufBytes(kv, tensors), path);
};

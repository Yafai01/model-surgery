// Browser GGUF header reader. Port of backend/model_surgery/formats/gguf.py.
// Only a prefix of the file (the header) is ever read; the prefix grows on demand.
// All counts, lengths, shapes and offsets are validated against the real file size.

import { GGML_BY_ID, type GGMLType, tensorNBytes } from "./ggml";

export class FormatError extends Error {}
class NeedMore extends Error {
  constructor(public need: number) {
    super("need more data");
  }
}

const LIMITS = {
  maxTensors: 200_000,
  maxKv: 100_000,
  maxStringBytes: 64 * 1024 * 1024,
  maxArrayLen: 50_000_000,
  maxArrayDepth: 3,
  maxAlignment: 1024 * 1024,
  maxNameBytes: 1024,
  keepStringChars: 16_384,
  keepArrayItems: 32,
  keepNumericArrayItems: 1024,
  keepArrayItemChars: 256,
  maxHeaderBytes: 1024 * 1024 * 1024,
};
const GGML_MAX_DIMS = 4;
const GGML_MAX_NAME = 64;

// value type id -> [byte size, name]
const SCALARS: Record<number, [number, string]> = {
  0: [1, "uint8"], 1: [1, "int8"], 2: [2, "uint16"], 3: [2, "int16"], 4: [4, "uint32"], 5: [4, "int32"],
  6: [4, "float32"], 7: [1, "bool"], 10: [8, "uint64"], 11: [8, "int64"], 12: [8, "float64"],
};
const T_STRING = 8;
const T_ARRAY = 9;
const typeName = (t: number) => (t === T_STRING ? "string" : t === T_ARRAY ? "array" : SCALARS[t]?.[1]);

export interface GGUFValue {
  type: string;
  value: unknown;
  length: number | null;
  truncated: boolean;
}

export interface GGUFTensor {
  name: string;
  shape: number[];
  typeId: number;
  ggml: GGMLType | undefined;
  offset: number;
  nElements: number;
  nBytes: number | null;
}

export interface GGUFFile {
  fileSize: number;
  version: number;
  byteOrder: "little" | "big";
  alignment: number;
  dataOffset: number;
  metadata: Map<string, GGUFValue>;
  tensors: GGUFTensor[];
  errors: string[];
  warnings: string[];
}

export function ggufGet(g: GGUFFile, key: string): unknown {
  return g.metadata.get(key)?.value;
}

const decoder = new TextDecoder("utf-8", { fatal: false });

class Reader {
  dv: DataView;
  pos = 0;
  le = true;
  constructor(
    private buf: ArrayBuffer,
    public fileSize: number,
  ) {
    this.dv = new DataView(buf);
  }
  remaining() {
    return this.fileSize - this.pos;
  }
  need(n: number, what: string) {
    if (n < 0 || n > this.remaining()) throw new FormatError(`truncated file: need ${n} bytes for ${what} at offset ${this.pos}, ${this.remaining()} left`);
    if (this.pos + n > this.dv.byteLength) throw new NeedMore(this.pos + n);
  }
  skip(n: number, what: string) {
    if (n < 0 || n > this.remaining()) throw new FormatError(`truncated file: cannot skip ${n} bytes for ${what}`);
    this.pos += n;
  }
  scalar(t: number, what: string): number | boolean {
    const size = SCALARS[t][0];
    this.need(size, what);
    const p = this.pos;
    this.pos += size;
    const d = this.dv;
    switch (t) {
      case 0: return d.getUint8(p);
      case 1: return d.getInt8(p);
      case 2: return d.getUint16(p, this.le);
      case 3: return d.getInt16(p, this.le);
      case 4: return d.getUint32(p, this.le);
      case 5: return d.getInt32(p, this.le);
      case 6: return d.getFloat32(p, this.le);
      case 7: return d.getUint8(p) !== 0;
      case 10: return Number(d.getBigUint64(p, this.le));
      case 11: return Number(d.getBigInt64(p, this.le));
      case 12: return d.getFloat64(p, this.le);
    }
    throw new FormatError(`bad scalar type ${t}`);
  }
  u32(what: string) {
    return this.scalar(4, what) as number;
  }
  u64(what: string) {
    return this.scalar(10, what) as number;
  }
  rawLen(what: string, max: number) {
    const n = this.u64(`${what} length`);
    if (n > max) throw new FormatError(`${what} length ${n} exceeds limit ${max}`);
    return n;
  }
  string(what: string, max: number): string {
    const n = this.rawLen(what, max);
    this.need(n, what);
    const s = decoder.decode(new Uint8Array(this.buf, this.pos, n));
    this.pos += n;
    return s;
  }
}

function truncate(s: string, n: number): [string, boolean] {
  return s.length <= n ? [s, false] : [s.slice(0, n), true];
}

function readValue(r: Reader, t: number, key: string, depth = 0): GGUFValue {
  if (t in SCALARS) return { type: SCALARS[t][1], value: r.scalar(t, key), length: null, truncated: false };
  if (t === T_STRING) {
    const n = r.rawLen(key, LIMITS.maxStringBytes);
    r.need(n, key);
    const full = decoder.decode(new Uint8Array(r.dv.buffer, r.pos, n));
    r.pos += n;
    const [s, tr] = truncate(full, LIMITS.keepStringChars);
    return { type: "string", value: s, length: n, truncated: tr };
  }
  if (t === T_ARRAY) {
    if (depth >= LIMITS.maxArrayDepth) throw new FormatError(`metadata '${key}': array nesting deeper than ${LIMITS.maxArrayDepth}`);
    const et = r.u32(`${key} array type`);
    const n = r.u64(`${key} array length`);
    if (typeName(et) === undefined) throw new FormatError(`metadata '${key}': unknown array element type ${et}`);
    if (n > LIMITS.maxArrayLen) throw new FormatError(`metadata '${key}': array length ${n} exceeds limit ${LIMITS.maxArrayLen}`);
    const preview: unknown[] = [];
    let truncated = false;
    if (et in SCALARS) {
      const es = SCALARS[et][0];
      if (n * es > r.remaining()) throw new FormatError(`truncated file: metadata '${key}' array needs ${n * es} bytes`);
      const keep = n <= LIMITS.keepNumericArrayItems ? n : LIMITS.keepArrayItems;
      for (let i = 0; i < keep; i++) preview.push(r.scalar(et, key));
      r.skip((n - keep) * es, key);
      truncated = n > keep;
    } else if (et === T_STRING) {
      if (n * 8 > r.remaining()) throw new FormatError(`truncated file: metadata '${key}' array of ${n} elements cannot fit`);
      for (let i = 0; i < n; i++) {
        if (i < LIMITS.keepArrayItems) {
          const [s, tr] = truncate(r.string(key, LIMITS.maxStringBytes), LIMITS.keepArrayItemChars);
          truncated ||= tr;
          preview.push(s);
        } else {
          r.skip(r.rawLen(key, LIMITS.maxStringBytes), key); // length-checked skip, no decode
        }
      }
      truncated ||= n > LIMITS.keepArrayItems;
    } else {
      for (let i = 0; i < n; i++) {
        const item = readValue(r, et, key, depth + 1);
        if (i < LIMITS.keepArrayItems) preview.push(item.value);
      }
      truncated ||= n > LIMITS.keepArrayItems;
    }
    return { type: `array[${typeName(et)}]`, value: preview, length: n, truncated };
  }
  throw new FormatError(`metadata '${key}': unknown value type ${t}`);
}

function parse(buf: ArrayBuffer, fileSize: number): GGUFFile {
  const r = new Reader(buf, fileSize);
  r.need(8, "magic and version");
  const magic = new Uint8Array(buf, 0, 4);
  if (String.fromCharCode(...magic) !== "GGUF") throw new FormatError("missing GGUF magic bytes");
  const le = r.dv.getUint32(4, true);
  const be = r.dv.getUint32(4, false);
  let version: number;
  let byteOrder: "little" | "big";
  if (le >= 1 && le <= 0xffff) [version, byteOrder] = [le, "little"];
  else if (be >= 1 && be <= 0xffff) [version, byteOrder] = [be, "big"];
  else throw new FormatError(`implausible GGUF version field 0x${le.toString(16)}`);
  if (version === 1) throw new FormatError("GGUF v1 (pre-August-2023) is not supported; re-convert the model with current llama.cpp");
  if (version > 3) throw new FormatError(`GGUF version ${version} is newer than this reader supports (2-3)`);
  r.le = byteOrder === "little";
  r.pos = 8;
  const warnings: string[] = [];
  const errors: string[] = [];
  if (byteOrder === "big") warnings.push("big-endian GGUF: most llama.cpp builds only load little-endian files");

  const nTensors = r.u64("tensor count");
  const nKv = r.u64("metadata count");
  if (nTensors > LIMITS.maxTensors) throw new FormatError(`tensor count ${nTensors} exceeds limit ${LIMITS.maxTensors}`);
  if (nKv > LIMITS.maxKv) throw new FormatError(`metadata count ${nKv} exceeds limit ${LIMITS.maxKv}`);

  const metadata = new Map<string, GGUFValue>();
  for (let i = 0; i < nKv; i++) {
    const key = r.string(`metadata key #${i}`, LIMITS.maxNameBytes);
    if (metadata.has(key)) throw new FormatError(`duplicate metadata key '${key}'`);
    const vt = r.u32(`type of '${key}'`);
    metadata.set(key, readValue(r, vt, key));
  }

  let alignment = 32;
  const av = metadata.get("general.alignment");
  if (av) {
    const a = av.value as number;
    if (av.type !== "uint32" || !(a > 0) || (a & (a - 1)) !== 0 || a > LIMITS.maxAlignment) throw new FormatError(`invalid general.alignment ${String(a)}`);
    alignment = a;
  }

  const tensors: GGUFTensor[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < nTensors; i++) {
    const name = r.string(`tensor name #${i}`, LIMITS.maxNameBytes);
    if (seen.has(name)) throw new FormatError(`duplicate tensor name '${name}'`);
    seen.add(name);
    if (new TextEncoder().encode(name).length >= GGML_MAX_NAME) warnings.push(`tensor name longer than ggml's ${GGML_MAX_NAME}-byte limit: '${name.slice(0, 80)}'`);
    const nDims = r.u32(`'${name}' n_dims`);
    if (nDims < 1 || nDims > GGML_MAX_DIMS) throw new FormatError(`tensor '${name}' has ${nDims} dimensions (allowed 1..${GGML_MAX_DIMS})`);
    const shape: number[] = [];
    for (let d = 0; d < nDims; d++) shape.push(r.u64(`'${name}' dim`));
    const typeId = r.u32(`'${name}' type`);
    const offset = r.u64(`'${name}' offset`);
    const nEl = shape.reduce((a, b) => a * b, 1);
    if (nEl > 2 ** 50) throw new FormatError(`tensor '${name}' shape [${shape}] is implausibly large`);
    const gt = GGML_BY_ID.get(typeId);
    let nBytes: number | null = null;
    if (!gt) errors.push(`tensor '${name}' uses unknown or removed ggml type id ${typeId}`);
    else if (shape[0] % gt.blockSize) errors.push(`tensor '${name}': row length ${shape[0]} not divisible by ${gt.name} block size ${gt.blockSize}`);
    else nBytes = tensorNBytes(gt, shape);
    if (shape.includes(0)) warnings.push(`tensor '${name}' has a zero-length dimension`);
    if (offset % alignment) errors.push(`tensor '${name}' offset ${offset} is not aligned to ${alignment}`);
    tensors.push({ name, shape, typeId, ggml: gt, offset, nElements: nEl, nBytes });
  }

  const headerEnd = r.pos;
  const dataOffset = headerEnd + ((alignment - (headerEnd % alignment)) % alignment);
  let dataSize = 0;
  if (dataOffset > fileSize) {
    if (nTensors) errors.push("file ends before the tensor data section begins");
  } else dataSize = fileSize - dataOffset;

  const sized = tensors.filter((t) => t.nBytes !== null).sort((a, b) => a.offset - b.offset);
  let prev: GGUFTensor | null = null;
  for (const t of sized) {
    if (t.offset + t.nBytes! > dataSize) errors.push(`tensor '${t.name}' data [${t.offset}, ${t.offset + t.nBytes!}) exceeds data section of ${dataSize} bytes (truncated file?)`);
    if (prev && prev.offset + prev.nBytes! > t.offset) errors.push(`tensors '${prev.name}' and '${t.name}' overlap`);
    prev = t;
  }
  if (sized.length) {
    const last = sized[sized.length - 1];
    const slack = dataSize - (last.offset + last.nBytes!);
    if (slack > 64 * 1024 * 1024) warnings.push(`${slack} unreferenced bytes after the last tensor`);
  }
  const errs = errors.length > 200 ? [...errors.slice(0, 200), `... ${errors.length - 200} more integrity errors`] : errors;
  return { fileSize, version, byteOrder, alignment, dataOffset, metadata, tensors, errors: errs, warnings };
}

export async function isGGUF(file: Blob): Promise<boolean> {
  const b = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  return b.length === 4 && String.fromCharCode(...b) === "GGUF";
}

/** Parse a GGUF header, reading only as much of the file as the header needs. */
export async function readGGUF(file: Blob): Promise<GGUFFile> {
  let size = Math.min(file.size, 8 * 1024 * 1024);
  for (;;) {
    const buf = await file.slice(0, size).arrayBuffer();
    try {
      return parse(buf, file.size);
    } catch (e) {
      if (!(e instanceof NeedMore)) throw e;
      const next = Math.min(file.size, Math.max(e.need + 1024 * 1024, size * 2));
      if (next > LIMITS.maxHeaderBytes) throw new FormatError("GGUF header exceeds 1 GiB; refusing to read further");
      if (next <= size) throw new FormatError("truncated file");
      size = next;
    }
  }
}

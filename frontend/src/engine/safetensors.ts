// Browser safetensors header reader. Port of backend/model_surgery/formats/safetensors_reader.py.

import { FormatError } from "./gguf";
import type { ByteSource } from "./source";

const MAX_HEADER_BYTES = 100 * 1024 * 1024;
const MAX_TENSORS = 1_000_000;
const MAX_DIMS = 16;

export const DTYPE_BITS: Record<string, number> = {
  BOOL: 8, U8: 8, I8: 8, F8_E5M2: 8, F8_E4M3: 8, F8_E8M0: 8, I16: 16, U16: 16, F16: 16, BF16: 16,
  I32: 32, U32: 32, F32: 32, I64: 64, U64: 64, F64: 64, F4: 4, F6_E2M3: 6, F6_E3M2: 6,
};

export interface STTensor {
  name: string;
  dtype: string;
  shape: number[];
  begin: number;
  end: number;
  nElements: number;
}

export interface SafetensorsFile {
  fileSize: number;
  headerSize: number;
  dataOffset: number;
  metadata: Record<string, string>;
  tensors: STTensor[];
  errors: string[];
  warnings: string[];
}

/** Keys of the top-level JSON object, in order (JSON.parse silently drops duplicates). */
export function topLevelKeys(s: string): string[] {
  const keys: string[] = [];
  let depth = 0;
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === '"') {
      let j = i + 1;
      while (j < s.length && s[j] !== '"') j += s[j] === "\\" ? 2 : 1;
      const raw = s.slice(i, j + 1);
      i = j + 1;
      if (depth === 1) {
        let k = i;
        while (k < s.length && /\s/.test(s[k])) k++;
        if (s[k] === ":") keys.push(JSON.parse(raw));
      }
      continue;
    }
    if (c === "{" || c === "[") depth++;
    else if (c === "}" || c === "]") depth--;
    i++;
  }
  return keys;
}

export function assertNoDuplicateKeys(text: string, what: string) {
  const seen = new Set<string>();
  for (const k of topLevelKeys(text)) {
    if (seen.has(k)) throw new FormatError(`duplicate key in ${what}: '${k}'`);
    seen.add(k);
  }
}

const isUint = (x: unknown): x is number => typeof x === "number" && Number.isInteger(x) && x >= 0;

export async function readSafetensors(file: ByteSource): Promise<SafetensorsFile> {
  const fileSize = file.size;
  if (fileSize < 8) throw new FormatError("file too small to be safetensors");
  const n = Number(new DataView(await file.slice(0, 8).arrayBuffer()).getBigUint64(0, true));
  if (n > MAX_HEADER_BYTES) throw new FormatError(`header length ${n} exceeds ${MAX_HEADER_BYTES} byte limit`);
  if (8 + n > fileSize) throw new FormatError(`header length ${n} exceeds file size ${fileSize}`);
  const raw = await file.slice(8, 8 + n).arrayBuffer(); // read errors must propagate as-is
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  } catch {
    throw new FormatError("header is not valid UTF-8");
  }
  if (!text.trimStart().startsWith("{")) throw new FormatError("safetensors header is not a JSON object");
  let header: Record<string, unknown>;
  try {
    header = JSON.parse(text);
  } catch (e) {
    throw new FormatError(`header is not valid JSON: ${(e as Error).message}`);
  }
  assertNoDuplicateKeys(text, "safetensors header");
  if (typeof header !== "object" || header === null || Array.isArray(header)) throw new FormatError("safetensors header is not a JSON object");

  const errors: string[] = [];
  const warnings: string[] = [];
  let metadata: Record<string, string> = {};
  const meta = header.__metadata__;
  delete header.__metadata__;
  if (meta !== undefined) {
    if (typeof meta !== "object" || meta === null || !Object.values(meta).every((v) => typeof v === "string")) errors.push("__metadata__ must map strings to strings");
    else metadata = Object.fromEntries(Object.entries(meta as Record<string, string>).map(([k, v]) => [k, v.slice(0, 16_384)]));
  }
  const entries = Object.entries(header);
  if (entries.length > MAX_TENSORS) throw new FormatError(`${entries.length} tensors exceeds limit ${MAX_TENSORS}`);

  const dataSize = fileSize - 8 - n;
  const tensors: STTensor[] = [];
  for (const [name, info] of entries) {
    if (typeof info !== "object" || info === null) {
      errors.push(`tensor '${name.slice(0, 120)}': descriptor is not an object`);
      continue;
    }
    const { dtype, shape, data_offsets: offs } = info as { dtype: unknown; shape: unknown; data_offsets: unknown };
    if (typeof dtype !== "string" || !(dtype in DTYPE_BITS)) {
      errors.push(`tensor '${name}': unknown dtype ${JSON.stringify(dtype)}`);
      continue;
    }
    if (!Array.isArray(shape) || shape.length > MAX_DIMS || !shape.every(isUint)) {
      errors.push(`tensor '${name}': invalid shape ${JSON.stringify(shape)}`);
      continue;
    }
    if (!(Array.isArray(offs) && offs.length === 2 && offs.every(isUint) && offs[0] <= offs[1])) {
      errors.push(`tensor '${name}': invalid data_offsets ${JSON.stringify(offs)}`);
      continue;
    }
    const nEl = (shape as number[]).reduce((a, b) => a * b, 1);
    if (nEl > 2 ** 50) {
      errors.push(`tensor '${name}': shape [${shape}] is implausibly large`);
      continue;
    }
    const expected = Math.ceil((nEl * DTYPE_BITS[dtype]) / 8);
    if (offs[1] - offs[0] !== expected) errors.push(`tensor '${name}': byte range ${offs[1] - offs[0]} != ${expected} expected for ${dtype}[${shape}]`);
    if (offs[1] > dataSize) errors.push(`tensor '${name}': data ends at ${offs[1]} beyond data section of ${dataSize} bytes (truncated file?)`);
    tensors.push({ name, dtype, shape: shape as number[], begin: offs[0], end: offs[1], nElements: nEl });
  }

  const ordered = [...tensors].sort((a, b) => a.begin - b.begin || a.end - b.end);
  let cursor = 0;
  for (const t of ordered) {
    if (t.begin < cursor) errors.push(`tensor '${t.name}' overlaps a preceding tensor`);
    else if (t.begin > cursor) warnings.push(`${t.begin - cursor} byte gap before tensor '${t.name}'`);
    cursor = Math.max(cursor, t.end);
  }
  if (cursor < dataSize) warnings.push(`${dataSize - cursor} trailing bytes not referenced by any tensor`);
  return {
    fileSize,
    headerSize: n,
    dataOffset: 8 + n,
    metadata,
    tensors,
    errors: errors.length > 200 ? [...errors.slice(0, 200), `... ${errors.length - 200} more integrity errors`] : errors,
    warnings: warnings.length > 50 ? [...warnings.slice(0, 50), `... ${warnings.length - 50} more warnings`] : warnings,
  };
}

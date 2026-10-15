// Hugging Face folder reader for files chosen in the browser. Port of backend/model_surgery/formats/hf.py.
// Pickle-based weights are listed but never read.

import { FormatError } from "./gguf";
import { assertNoDuplicateKeys, readSafetensors, type SafetensorsFile } from "./safetensors";
import type { FileLike } from "./source";

const MAX_JSON_BYTES = 8 * 1024 * 1024;
const MAX_SHARDS = 4096;
export const PICKLE_SUFFIXES = new Set([".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"]);
const OTHER_WEIGHTS: Record<string, string> = { ".h5": "TensorFlow/Keras HDF5", ".msgpack": "Flax msgpack", ".onnx": "ONNX", ".gguf": "GGUF" };
const SIDE_FILES = new Set([
  "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "generation_config.json",
  "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json",
]);

export class UnsupportedFormatError extends Error {}

export const relPath = (f: FileLike) => f.webkitRelativePath || f.name;
export const suffix = (name: string) => (name.includes(".") ? name.slice(name.lastIndexOf(".")).toLowerCase() : "");
export const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0); // Python sorted() order
const dirOf = (p: string) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");

export interface HFFolder {
  dir: string;
  config: Record<string, unknown>;
  configFile: FileLike;
  indexFile: FileLike | null;
  shards: { file: FileLike; st: SafetensorsFile }[];
  sideFiles: FileLike[];
  unparsed: { file: FileLike; reason: string }[];
  errors: string[];
  warnings: string[];
}

export async function mapPool<T, R>(items: T[], limit: number, fn: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

export async function readJsonLimited(f: FileLike, max = MAX_JSON_BYTES): Promise<unknown> {
  if (f.size > max) throw new FormatError(`${f.name} is ${f.size} bytes; refusing to parse JSON larger than ${max}`);
  const raw = await f.arrayBuffer(); // read errors (e.g. network) must propagate as-is
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  } catch {
    throw new FormatError(`${f.name} is not valid UTF-8`);
  }
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch (e) {
    throw new FormatError(`${f.name} is not valid JSON: ${(e as Error).message}`);
  }
  assertNoDuplicateKeys(text, f.name);
  return v;
}

/** Locate the model folder (the shallowest folder containing config.json) among the picked files. */
export function findConfig(files: FileLike[]): FileLike | null {
  const configs = files.filter((f) => f.name === "config.json").sort((a, b) => relPath(a).split("/").length - relPath(b).split("/").length);
  return configs[0] ?? null;
}

export async function readHFFolder(files: FileLike[]): Promise<HFFolder> {
  const configFile = findConfig(files);
  if (!configFile) throw new FormatError("folder has no config.json; not a Hugging Face model directory");
  const dir = dirOf(relPath(configFile));
  const inDir = files.filter((f) => dirOf(relPath(f)) === dir);
  const byName = new Map(inDir.map((f) => [f.name, f]));

  const config = await readJsonLimited(configFile);
  if (typeof config !== "object" || config === null || Array.isArray(config)) throw new FormatError("config.json is not a JSON object");

  const errors: string[] = [];
  const warnings: string[] = [];
  const unparsed: { file: FileLike; reason: string }[] = [];
  const sideFiles: FileLike[] = [];
  for (const f of [...inDir].sort((a, b) => byCodePoint(a.name, b.name))) {
    const s = suffix(f.name);
    if (PICKLE_SUFFIXES.has(s) && f.name !== "training_args.bin") unparsed.push({ file: f, reason: "pickle-based PyTorch weights are never opened (unpickling can execute code)" });
    else if (s in OTHER_WEIGHTS) unparsed.push({ file: f, reason: `${OTHER_WEIGHTS[s]} files are not parsed in a Hugging Face directory scan` });
    else if (SIDE_FILES.has(f.name)) sideFiles.push(f);
  }

  const indexFile = byName.get("model.safetensors.index.json") ?? null;
  let shardFiles: FileLike[];
  let weightMap: Record<string, string> | null = null;
  if (indexFile) {
    const index = (await readJsonLimited(indexFile, 64 * 1024 * 1024)) as { weight_map?: unknown };
    const wm = index?.weight_map;
    if (typeof wm !== "object" || wm === null || !Object.values(wm).every((v) => typeof v === "string")) throw new FormatError("model.safetensors.index.json has no valid weight_map");
    weightMap = wm as Record<string, string>;
    const names = [...new Set(Object.values(weightMap))].sort();
    if (names.length > MAX_SHARDS) throw new FormatError(`index references ${names.length} shards (limit ${MAX_SHARDS})`);
    for (const nm of names) {
      if (!nm || nm === "." || nm === ".." || /[\\/:]/.test(nm)) throw new FormatError(`refusing unsafe shard filename ${JSON.stringify(nm)} in index`);
    }
    const missing = names.filter((nm) => !byName.has(nm));
    if (missing.length) errors.push(`index references missing shard(s): ${missing.slice(0, 10).join(", ")}`);
    shardFiles = names.filter((nm) => byName.has(nm)).map((nm) => byName.get(nm)!);
  } else {
    shardFiles = inDir.filter((f) => suffix(f.name) === ".safetensors").sort((a, b) => byCodePoint(a.name, b.name));
    if (shardFiles.length > MAX_SHARDS) throw new FormatError(`${shardFiles.length} safetensors files (limit ${MAX_SHARDS})`);
  }

  if (!shardFiles.length) {
    if (unparsed.length)
      throw new UnsupportedFormatError(
        "no safetensors weights found; this folder only contains formats Model Surgery will not open: " + [...new Set(unparsed.map((u) => suffix(u.file.name)))].sort().join(", "),
      );
    throw new FormatError("no safetensors weight files found in the folder");
  }

  // Read shard headers concurrently (matters for remote Hub files), then process in order.
  const read = await mapPool(shardFiles, 6, async (f) => {
    try {
      return { f, st: await readSafetensors(f) };
    } catch (e) {
      return { f, err: (e as Error).message };
    }
  });
  const shards: { file: FileLike; st: SafetensorsFile }[] = [];
  const seen = new Map<string, string>();
  for (const r of read) {
    const f = r.f;
    if (!("st" in r) || !r.st) {
      errors.push(`${f.name}: ${r.err}`);
      continue;
    }
    const st = r.st;
    errors.push(...st.errors.map((m) => `${f.name}: ${m}`));
    warnings.push(...st.warnings.slice(0, 10).map((m) => `${f.name}: ${m}`));
    for (const t of st.tensors) {
      if (seen.has(t.name)) errors.push(`tensor '${t.name}' appears in both ${seen.get(t.name)} and ${f.name}`);
      seen.set(t.name, f.name);
    }
    shards.push({ file: f, st });
  }
  if (weightMap) {
    for (const [tname, fname] of Object.entries(weightMap)) {
      const actual = seen.get(tname);
      if (actual === undefined) errors.push(`index lists tensor '${tname}' but it is not in any shard`);
      else if (actual !== fname) errors.push(`index maps '${tname}' to ${fname} but it is stored in ${actual}`);
    }
    const extra = [...seen.keys()].filter((k) => !(k in weightMap!));
    if (extra.length) warnings.push(`${extra.length} tensor(s) present in shards but missing from the index`);
  }
  return {
    dir,
    config: config as Record<string, unknown>,
    configFile,
    indexFile,
    shards,
    sideFiles,
    unparsed,
    errors: errors.length > 200 ? [...errors.slice(0, 200), `... ${errors.length - 200} more`] : errors,
    warnings,
  };
}

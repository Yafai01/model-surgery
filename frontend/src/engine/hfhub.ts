// Inspect a model straight from the Hugging Face Hub without downloading its weights.
// Model files are read with HTTP Range requests (only header bytes); repository facts come from the Hub API.
// An optional access token is held in memory only and sent only to huggingface.co.

import type { HubInfo, ScanResult } from "../types";
import { FormatError } from "./gguf";
import { ScanError, scanFiles } from "./scanner";
import type { ByteRange, FileLike } from "./source";

const HUB = "https://huggingface.co";
const READ_AHEAD = 256 * 1024;
const MAX_WHOLE_FILE = 64 * 1024 * 1024; // JSON files read in full must be smaller than this
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;

export interface HubRef {
  repo: string;
  revision: string;
  file: string | null;
}

export interface HubOptions {
  file?: string | null;
  token?: string | null;
  onProgress?: (p: { requests: number; bytes: number; stage: string }) => void;
  /** Aborts every in-flight and future request of this scan. */
  signal?: AbortSignal;
}

/** Accepts https://huggingface.co/owner/name[/tree|blob|resolve/rev[/path]], hf.co/..., or owner/name. */
export function parseHubRef(input: string): HubRef {
  let s = input.trim();
  if (!s) throw new ScanError("enter a Hugging Face model link or owner/name");
  if (/^https?:\/\//i.test(s) || /^(www\.)?(huggingface\.co|hf\.co)\//i.test(s)) {
    let u: URL;
    try {
      u = new URL(/^https?:\/\//i.test(s) ? s : `https://${s}`);
    } catch {
      throw new ScanError("that does not look like a valid link");
    }
    if (!["huggingface.co", "www.huggingface.co", "hf.co"].includes(u.hostname)) throw new ScanError("only huggingface.co links are supported");
    s = decodeURIComponent(u.pathname).replace(/^\/+|\/+$/g, "");
  }
  const parts = s.split("/").filter(Boolean);
  if (["datasets", "spaces"].includes(parts[0])) throw new ScanError("that is a dataset or Space link; paste a model link");
  if (parts[0] === "models") parts.shift();
  if (parts.length < 2) throw new ScanError("expected owner/model, e.g. Qwen/Qwen2.5-0.5B-Instruct");
  const [owner, name, kind, rev, ...rest] = parts;
  if (!NAME.test(owner) || !NAME.test(name)) throw new ScanError("invalid repository name");
  let revision = "main";
  let file: string | null = null;
  if (kind && ["tree", "blob", "resolve"].includes(kind) && rev) {
    revision = rev;
    if (rest.length) file = rest.join("/");
  }
  if (!/^[A-Za-z0-9._-]{1,128}$/.test(revision)) throw new ScanError("invalid revision in link");
  if (file && (file.split("/").some((p) => p === ".." || p === "." || !p) || file.length > 512)) throw new ScanError("invalid file path in link");
  return { repo: `${owner}/${name}`, revision, file };
}

class HubClient {
  requests = 0;
  bytes = 0;
  constructor(
    private token: string | null,
    private onProgress?: HubOptions["onProgress"],
    private signal?: AbortSignal,
  ) {}

  get hasToken() {
    return !!this.token;
  }

  stage = "connecting";
  progress(stage?: string) {
    if (stage) this.stage = stage;
    this.onProgress?.({ requests: this.requests, bytes: this.bytes, stage: this.stage });
  }

  async fetch(url: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    // The token only ever goes to huggingface.co itself (redirects to the CDN drop it, per the Fetch spec).
    if (this.token && new URL(url).origin === HUB) headers.set("Authorization", `Bearer ${this.token}`);
    if (this.signal?.aborted) throw new ScanError("cancelled");
    this.requests++;
    this.progress();
    const signals = [init.signal, this.signal].filter((x): x is AbortSignal => !!x);
    const signal = signals.length > 1 && "any" in AbortSignal ? AbortSignal.any(signals) : signals[signals.length - 1];
    let r: Response;
    try {
      r = await fetch(url, { ...init, signal, headers, credentials: "omit", referrerPolicy: "no-referrer" });
    } catch {
      if (this.signal?.aborted) throw new ScanError("cancelled");
      throw new ScanError("could not reach huggingface.co (offline, blocked by a firewall, or a browser extension?)");
    }
    if (r.status === 401 || r.status === 403) {
      const code = r.headers.get("X-Error-Code");
      if (code === "GatedRepo")
        throw new ScanError(
          this.token
            ? "this token has no access to the gated model yet: request access on its Hugging Face page and wait for approval"
            : "this model is gated: request access on its Hugging Face page, then add a read access token below",
        );
      if (code === "RepoNotFound" || (!this.token && !code)) throw new ScanError("repository not found (or it is private: add an access token that can read it)");
      throw new ScanError(this.token ? "access denied: check that the token is valid and has read permission" : "access denied by the Hugging Face Hub");
    }
    if (r.status === 404) throw new ScanError("not found on the Hugging Face Hub (check the link, revision or file name)");
    if (r.status === 429) throw new ScanError("the Hugging Face Hub is rate-limiting requests; try again in a minute");
    if (!r.ok && r.status !== 206) throw new ScanError(`Hugging Face Hub returned HTTP ${r.status}`);
    return r;
  }
}

/** A Hub file read lazily with Range requests and a small read-ahead cache. */
class RemoteFile implements FileLike {
  readonly name: string;
  readonly webkitRelativePath: string;
  readonly headerChunk = 1024 * 1024;
  private cache: { start: number; buf: ArrayBuffer } | null = null;

  constructor(
    private client: HubClient,
    private repo: string,
    private sha: string,
    readonly path: string,
    readonly size: number,
  ) {
    this.name = path.split("/").pop()!;
    this.webkitRelativePath = `${repo}/${path}`;
  }

  private url() {
    return `${HUB}/${this.repo}/resolve/${this.sha}/${this.path.split("/").map(encodeURIComponent).join("/")}`;
  }

  async read(start: number, end: number): Promise<ArrayBuffer> {
    start = Math.max(0, Math.min(start, this.size));
    end = Math.max(start, Math.min(end, this.size));
    if (end === start) return new ArrayBuffer(0);
    const c = this.cache;
    const cEnd = c ? c.start + c.buf.byteLength : 0;
    if (c && start >= c.start && end <= cEnd) return c.buf.slice(start - c.start, end - c.start);
    // Growing a prefix we already hold (GGUF header reads): fetch only the missing tail.
    const extend = !!c && start >= c.start && start <= cEnd && end > cEnd;
    const from = extend ? cEnd : start;
    const fetchEnd = Math.min(this.size, Math.max(end, from + READ_AHEAD));
    const ctrl = new AbortController();
    const r = await this.client.fetch(this.url(), { headers: { Range: `bytes=${from}-${fetchEnd - 1}` }, signal: ctrl.signal });
    let buf: ArrayBuffer;
    let bufStart = from;
    if (r.status === 206) {
      buf = await r.arrayBuffer();
      if (buf.byteLength !== fetchEnd - from) throw new FormatError(`short read from the Hub for ${this.name}`);
    } else if (this.size <= MAX_WHOLE_FILE) {
      buf = await r.arrayBuffer(); // small non-LFS file served whole
      bufStart = 0;
      if (buf.byteLength !== this.size) throw new FormatError(`unexpected length from the Hub for ${this.name}`);
    } else {
      ctrl.abort(); // never stream a whole multi-GB file because a server ignored the Range header
      throw new ScanError(`the Hub did not honour a byte-range request for ${this.name} (HTTP ${r.status})`);
    }
    this.client.bytes += buf.byteLength;
    this.client.progress();
    if (extend && bufStart === from) {
      const merged = new Uint8Array(c!.buf.byteLength + buf.byteLength);
      merged.set(new Uint8Array(c!.buf), 0);
      merged.set(new Uint8Array(buf), c!.buf.byteLength);
      this.cache = { start: c!.start, buf: merged.buffer };
    } else this.cache = { start: bufStart, buf };
    const k = this.cache;
    return k.buf.slice(start - k.start, end - k.start);
  }

  slice(start: number, end: number): ByteRange {
    return { arrayBuffer: () => this.read(start, end) };
  }

  async arrayBuffer(): Promise<ArrayBuffer> {
    if (this.size > MAX_WHOLE_FILE) throw new FormatError(`${this.name} is too large to read in full`);
    return this.read(0, this.size);
  }
}

// ------------------------------------------------------------------ repository metadata

interface ApiSibling {
  rfilename: string;
  size?: number;
  lfs?: { size?: number };
}

interface ApiModel {
  id: string;
  sha: string;
  author?: string;
  private?: boolean;
  gated?: string | boolean;
  downloads?: number;
  likes?: number;
  createdAt?: string;
  lastModified?: string;
  library_name?: string;
  pipeline_tag?: string;
  tags?: string[];
  cardData?: Record<string, unknown>;
  siblings?: ApiSibling[];
  safetensors?: { parameters?: Record<string, number>; total?: number };
  gguf?: { total?: number; architecture?: string; context_length?: number };
  usedStorage?: number;
}

const asList = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x) => typeof x === "string") : typeof v === "string" ? [v] : []);
const QUANT_LABEL = /(?:^|[-_.])((?:I?Q\d(?:_[A-Z0-9]+)*)|BF16|F16|F32|MXFP4(?:_MOE)?)(?=[-_.]|$)/i;
const SPLIT = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i;

function ggufVariants(sibs: { path: string; size: number | null }[]) {
  const out: HubInfo["gguf_variants"] = [];
  const groups = new Map<string, { path: string; size: number; shards: number }>();
  for (const s of sibs) {
    if (!s.path.toLowerCase().endsWith(".gguf")) continue;
    const m = SPLIT.exec(s.path);
    if (m) {
      const g = groups.get(m[1]) ?? { path: "", size: 0, shards: 0 };
      if (Number(m[2]) === 1) g.path = s.path;
      g.size += s.size ?? 0;
      g.shards++;
      groups.set(m[1], g);
    } else out.push({ path: s.path, size: s.size, label: null, shards: 1 });
  }
  for (const g of groups.values()) if (g.path) out.push({ path: g.path, size: g.size, label: null, shards: g.shards });
  for (const v of out) v.label = QUANT_LABEL.exec(v.path.split("/").pop()!.replace(/\.gguf$/i, ""))?.[1]?.toUpperCase() ?? null;
  return out.sort((a, b) => (a.size ?? 0) - (b.size ?? 0));
}

function pickDefaultGGUF(vs: HubInfo["gguf_variants"]): string | null {
  const llm = vs.filter((v) => !/mmproj|imatrix/i.test(v.path));
  const pool = llm.length ? llm : vs;
  return (pool.find((v) => v.label === "Q4_K_M") ?? pool.find((v) => v.label?.startsWith("Q4")) ?? pool[0])?.path ?? null;
}

async function readmeExcerpt(client: HubClient, repo: string, sha: string, sibs: Map<string, number | null>): Promise<string | null> {
  const size = sibs.get("README.md");
  if (size === undefined) return null;
  try {
    const f = new RemoteFile(client, repo, sha, "README.md", size ?? 0);
    const buf = await f.read(0, Math.min(size ?? 0, 48 * 1024));
    let text = new TextDecoder().decode(buf).replace(/^---[\s\S]*?\n---\s*/, ""); // drop YAML front matter (shown separately)
    if (text.length > 12_000) text = text.slice(0, 12_000) + "\n…";
    return text.trim() || null;
  } catch {
    return null;
  }
}

// ------------------------------------------------------------------ entry point

export async function scanHub(input: string, opts: HubOptions = {}): Promise<ScanResult> {
  const ref = parseHubRef(input);
  const target = opts.file ?? ref.file;
  const client = new HubClient(opts.token?.trim() || null, opts.onProgress, opts.signal);

  client.progress("reading repository metadata");
  const infoUrl = `${HUB}/api/models/${ref.repo}${ref.revision !== "main" ? `/revision/${encodeURIComponent(ref.revision)}` : ""}?blobs=true`;
  const info = (await (await client.fetch(infoUrl)).json()) as ApiModel;
  if (!info?.sha || !Array.isArray(info.siblings)) throw new ScanError("unexpected response from the Hugging Face Hub API");
  const repo = info.id || ref.repo;
  if (info.gated && !client.hasToken)
    throw new ScanError(
      `${repo} is gated (${info.gated} approval): request access on huggingface.co/${repo}, then add a read access token to inspect its files`,
    );
  const sibs = info.siblings.map((s) => ({ path: s.rfilename, size: s.size ?? s.lfs?.size ?? null }));
  const sizes = new Map(sibs.map((s) => [s.path, s.size]));
  const remote = (path: string) => {
    const size = sizes.get(path);
    if (size === undefined || size === null) throw new ScanError(`the Hub did not report a size for ${path}`);
    return new RemoteFile(client, repo, info.sha, path, size);
  };
  const variants = ggufVariants(sibs);
  const root = sibs.filter((s) => !s.path.includes("/"));
  const rootNames = new Set(root.map((s) => s.path));

  let files: FileLike[];
  let scanned: string;
  const ggufTarget = target?.toLowerCase().endsWith(".gguf") ? target : null;
  if (target && !sizes.has(target)) throw new ScanError(`${target} is not in this repository`);
  const hasHFLayout = rootNames.has("config.json") && root.some((s) => s.path.endsWith(".safetensors"));
  if (ggufTarget || (!target && !hasHFLayout && variants.length)) {
    const path = ggufTarget ?? pickDefaultGGUF(variants)!;
    const m = SPLIT.exec(path);
    const shardPaths = m ? sibs.filter((s) => s.path.startsWith(m[1] + "-") && SPLIT.test(s.path)).map((s) => s.path) : [path];
    files = shardPaths.map(remote);
    scanned = path;
  } else if (hasHFLayout && (!target || !target.includes("/"))) {
    client.progress("reading config and safetensors headers");
    files = root.map((s) => remote(s.path));
    scanned = "repository root";
  } else if (target?.endsWith(".safetensors")) {
    files = [remote(target)];
    scanned = target;
  } else {
    const kinds = [...new Set(root.map((s) => s.path.slice(s.path.lastIndexOf(".")).toLowerCase()).filter((x) => [".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".msgpack"].includes(x)))];
    throw new ScanError(
      kinds.length
        ? `this repository has no safetensors or GGUF weights (only ${kinds.join(", ")}); Model Surgery never opens pickle formats`
        : "no safetensors or GGUF model weights found in this repository",
    );
  }

  client.progress("reading model headers");
  const scan = await scanFiles(files);
  client.progress("reading model card");
  const readme = await readmeExcerpt(client, repo, info.sha, sizes);

  const card = info.cardData ?? {};
  const hub: HubInfo = {
    repo,
    url: `${HUB}/${repo}`,
    revision: ref.revision,
    sha: info.sha,
    scanned_target: scanned,
    author: info.author ?? repo.split("/")[0],
    pipeline_tag: info.pipeline_tag ?? null,
    library_name: info.library_name ?? null,
    license: asList(card.license)[0] ?? (info.tags ?? []).find((t) => t.startsWith("license:"))?.slice(8) ?? null,
    downloads: info.downloads ?? null,
    likes: info.likes ?? null,
    created_at: info.createdAt ?? null,
    last_modified: info.lastModified ?? null,
    gated: info.gated ? String(info.gated) : null,
    private: !!info.private,
    tags: (info.tags ?? []).filter((t) => !/^(license|base_model|dataset|region|arxiv):/.test(t)).slice(0, 40),
    base_models: asList(card.base_model),
    datasets: asList(card.datasets),
    languages: asList(card.language),
    api_parameters: info.safetensors ? { total: info.safetensors.total ?? null, by_dtype: info.safetensors.parameters ?? {} } : null,
    api_gguf: info.gguf ? { total: info.gguf.total ?? null, architecture: info.gguf.architecture ?? null, context_length: info.gguf.context_length ?? null } : null,
    used_storage_bytes: info.usedStorage ?? null,
    file_count: sibs.length,
    total_file_bytes: sibs.reduce((a, s) => a + (s.size ?? 0), 0),
    files: [...sibs].sort((a, b) => (b.size ?? 0) - (a.size ?? 0)).slice(0, 300),
    gguf_variants: variants,
    readme_excerpt: readme,
    bytes_fetched: client.bytes,
    requests: client.requests,
  };

  const shortSha = info.sha.slice(0, 12);
  const toHf = (p: string) => (p.startsWith(`${repo}/`) ? `hf://${repo}@${shortSha}/${p.slice(repo.length + 1)}` : `hf://${repo}@${shortSha}/${p}`);
  scan.path = scanned === "repository root" ? `hf://${repo}@${shortSha}` : toHf(scanned);
  for (const f of scan.files) f.path = toHf(f.path);
  if (scan.kind === "hf_safetensors") scan.name = repo; // config._name_or_path is often a stale local path
  scan.architecture.notes.unshift(`inspected remotely from huggingface.co/${repo} @ ${shortSha}: ${client.requests} requests, ${(client.bytes / 1048576).toFixed(1)} MiB of headers fetched (weights not downloaded)`);
  const apiTotal = hub.api_parameters?.total ?? hub.api_gguf?.total ?? null;
  const ours = scan.hyperparameters.find((f) => f.key === "n_params")?.value;
  if (apiTotal && typeof ours === "number" && apiTotal !== ours)
    scan.architecture.notes.push(`the Hub API reports ${apiTotal.toLocaleString("en-US")} parameters; the tensors in the scanned files sum to ${ours.toLocaleString("en-US")}`);
  scan.hub = hub;
  return scan;
}

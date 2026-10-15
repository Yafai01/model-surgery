import type {
  Comparison,
  DeviceProfile,
  FsListing,
  FsRoot,
  Job,
  MemoryEstimate,
  MemoryRequest,
  Plan,
  PlanRequest,
  ScanResult,
  ScanSummary,
  ToolDryRun,
  ToolsReport,
} from "./types";

/**
 * Two editions share one UI:
 *  - local (default): talks to the FastAPI backend on this machine (full feature set incl. quantization)
 *  - browser (`vite build --mode browser`, used for the hosted site): everything runs in the page;
 *    model files are read locally with the File API and never uploaded.
 */
export const BROWSER_EDITION = import.meta.env.MODE === "browser";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export type ExportFormat = "json" | "csv" | "md";

export interface HubScanOptions {
  file?: string | null;
  token?: string | null;
  onProgress?: (p: { requests: number; bytes: number; stage: string }) => void;
}

/** Read a model from the Hugging Face Hub in this page (header bytes only). Shared by both editions. */
async function scanHubInPage(input: string, opts: HubScanOptions): Promise<ScanResult> {
  const { scanHub } = await import("./engine/hfhub");
  try {
    return await scanHub(input, opts);
  } catch (e) {
    throw new ApiError((e as Error).message, 400);
  }
}

export interface Api {
  roots(): Promise<FsRoot[]>;
  list(path: string): Promise<FsListing>;
  scan(path: string): Promise<ScanResult>;
  scanFiles(files: File[]): Promise<ScanResult>;
  scanHub(input: string, opts?: HubScanOptions): Promise<ScanResult>;
  scans(): Promise<ScanSummary[]>;
  getScan(id: string): Promise<ScanResult>;
  deleteScan(id: string): Promise<unknown>;
  exportText(id: string, format: ExportFormat): Promise<string>;
  reportMd(id: string, req: MemoryRequest): Promise<string>;
  devices(): Promise<DeviceProfile[]>;
  memory(id: string, req: MemoryRequest): Promise<MemoryEstimate>;
  compare(a: string, b: string, ctx: number, device: string): Promise<Comparison>;
  compareMd(a: string, b: string, ctx: number, device: string): Promise<string>;
  tools(refresh?: boolean): Promise<ToolsReport>;
  plan(req: PlanRequest): Promise<Plan>;
  toolDryRun(planId: string): Promise<ToolDryRun>;
  execute(planId: string, token: string): Promise<Job>;
  jobs(): Promise<Job[]>;
  job(id: string): Promise<Job>;
  cancel(id: string): Promise<Job>;
}

// ---------------------------------------------------------------- local backend

// Every state-changing request carries this header; the backend rejects those without it.
const WRITE_HEADERS = { "Content-Type": "application/json", "X-Model-Surgery": "1" };

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: method === "GET" ? undefined : WRITE_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j?.detail) msg = typeof j.detail === "string" ? j.detail : j.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(msg, r.status);
  }
  const ct = r.headers.get("content-type") ?? "";
  return (ct.includes("application/json") ? r.json() : r.text()) as Promise<T>;
}

const q = encodeURIComponent;

const httpApi: Api = {
  roots: () => request<FsRoot[]>("GET", "/api/fs/roots"),
  list: (path) => request<FsListing>("GET", `/api/fs/list?path=${q(path)}`),
  scan: (path) => request<ScanResult>("POST", "/api/scan", { path }),
  scanFiles: () => Promise.reject(new ApiError("file picking is only used by the browser edition", 400)),
  // The page reads the Hub directly, then stores the result in the local history.
  scanHub: async (input, opts = {}) => request<ScanResult>("POST", "/api/scans/import", await scanHubInPage(input, opts)),
  scans: () => request<ScanSummary[]>("GET", "/api/scans"),
  getScan: (id) => request<ScanResult>("GET", `/api/scans/${q(id)}`),
  deleteScan: (id) => request<unknown>("DELETE", `/api/scans/${q(id)}`),
  exportText: async (id, format) => {
    const r = await fetch(`/api/scans/${q(id)}/export?format=${format}`);
    if (!r.ok) throw new ApiError(`${r.status} ${r.statusText}`, r.status);
    return r.text();
  },
  reportMd: (id, req) => request<string>("POST", `/api/scans/${q(id)}/report.md`, req),
  devices: () => request<DeviceProfile[]>("GET", "/api/devices"),
  memory: (id, req) => request<MemoryEstimate>("POST", `/api/scans/${q(id)}/memory`, req),
  compare: (a, b, ctx, device) => request<Comparison>("GET", `/api/compare?a=${q(a)}&b=${q(b)}&context_length=${ctx}&device_id=${q(device)}`),
  compareMd: (a, b, ctx, device) => request<string>("GET", `/api/compare.md?a=${q(a)}&b=${q(b)}&context_length=${ctx}&device_id=${q(device)}`),
  tools: (refresh = false) => request<ToolsReport>("GET", `/api/tools${refresh ? "?refresh=true" : ""}`),
  plan: (req) => request<Plan>("POST", "/api/quantize/plan", req),
  toolDryRun: (planId) => request<ToolDryRun>("POST", `/api/quantize/plan/${q(planId)}/tool-dry-run`),
  execute: (planId, token) => request<Job>("POST", "/api/quantize/jobs", { plan_id: planId, confirm: true, confirm_token: token }),
  jobs: () => request<Job[]>("GET", "/api/quantize/jobs"),
  job: (id) => request<Job>("GET", `/api/quantize/jobs/${q(id)}`),
  cancel: (id) => request<Job>("POST", `/api/quantize/jobs/${q(id)}/cancel`),
};

// ---------------------------------------------------------------- in-browser engine

const notInBrowser = () => Promise.reject(new ApiError("this feature needs the local Model Surgery app", 501));

async function loadEngine() {
  const [scanner, memory, reports, store] = await Promise.all([
    import("./engine/scanner"),
    import("./engine/memory"),
    import("./engine/reports"),
    import("./engine/store"),
  ]);
  return { scanner, memory, reports, store };
}

async function must(id: string): Promise<ScanResult> {
  const { store } = await loadEngine();
  const s = await store.getScan(id);
  if (!s) throw new ApiError("scan not found", 404);
  return s;
}

const memReq = (ctx: number, device: string): MemoryRequest => ({
  weight_format: "current", context_length: ctx, batch_size: 1, kv_cache_type: "f16", flash_attention: true, n_ubatch: 512, device_id: device, custom_device: null,
});

const browserApi: Api = {
  roots: notInBrowser,
  list: notInBrowser,
  scan: notInBrowser,
  async scanFiles(files) {
    const { scanner, store } = await loadEngine();
    try {
      const r = await scanner.scanFiles(files);
      await store.saveScan(r);
      return r;
    } catch (e) {
      throw new ApiError((e as Error).message, 400);
    }
  },
  async scanHub(input, opts = {}) {
    const r = await scanHubInPage(input, opts);
    await (await loadEngine()).store.saveScan(r);
    return r;
  },
  scans: async () => (await loadEngine()).store.listScans(),
  getScan: must,
  deleteScan: async (id) => (await loadEngine()).store.deleteScan(id),
  async exportText(id, format) {
    const s = await must(id);
    const { reports } = await loadEngine();
    return format === "json" ? JSON.stringify(s, null, 2) : format === "csv" ? reports.tensorsCsv(s) : reports.markdownReport(s);
  },
  async reportMd(id, req) {
    const s = await must(id);
    const { memory, reports } = await loadEngine();
    return reports.markdownReport(s, memory.estimateMemory(s, req));
  },
  devices: async () => (await loadEngine()).memory.DEFAULT_PROFILES,
  async memory(id, req) {
    const s = await must(id);
    return (await loadEngine()).memory.estimateMemory(s, req);
  },
  async compare(a, b, ctx, device) {
    const [sa, sb] = await Promise.all([must(a), must(b)]);
    return (await loadEngine()).reports.compare(sa, sb, memReq(ctx, device));
  },
  async compareMd(a, b, ctx, device) {
    const [sa, sb] = await Promise.all([must(a), must(b)]);
    const { reports } = await loadEngine();
    return reports.comparisonMarkdown(reports.compare(sa, sb, memReq(ctx, device)));
  },
  tools: notInBrowser,
  plan: notInBrowser,
  toolDryRun: notInBrowser,
  execute: notInBrowser,
  jobs: notInBrowser,
  job: notInBrowser,
  cancel: notInBrowser,
};

export const api: Api = BROWSER_EDITION ? browserApi : httpApi;

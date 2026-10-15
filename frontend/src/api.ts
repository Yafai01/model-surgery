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

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

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

export const api = {
  roots: () => request<FsRoot[]>("GET", "/api/fs/roots"),
  list: (path: string) => request<FsListing>("GET", `/api/fs/list?path=${q(path)}`),
  scan: (path: string) => request<ScanResult>("POST", "/api/scan", { path }),
  scans: () => request<ScanSummary[]>("GET", "/api/scans"),
  getScan: (id: string) => request<ScanResult>("GET", `/api/scans/${q(id)}`),
  deleteScan: (id: string) => request<unknown>("DELETE", `/api/scans/${q(id)}`),
  exportUrl: (id: string, format: "json" | "csv" | "md") => `/api/scans/${q(id)}/export?format=${format}`,
  reportMd: (id: string, req: MemoryRequest) => request<string>("POST", `/api/scans/${q(id)}/report.md`, req),
  devices: () => request<DeviceProfile[]>("GET", "/api/devices"),
  memory: (id: string, req: MemoryRequest) => request<MemoryEstimate>("POST", `/api/scans/${q(id)}/memory`, req),
  compare: (a: string, b: string, ctx: number, device: string) =>
    request<Comparison>("GET", `/api/compare?a=${q(a)}&b=${q(b)}&context_length=${ctx}&device_id=${q(device)}`),
  compareMd: (a: string, b: string, ctx: number, device: string) =>
    request<string>("GET", `/api/compare.md?a=${q(a)}&b=${q(b)}&context_length=${ctx}&device_id=${q(device)}`),
  tools: (refresh = false) => request<ToolsReport>("GET", `/api/tools${refresh ? "?refresh=true" : ""}`),
  plan: (req: PlanRequest) => request<Plan>("POST", "/api/quantize/plan", req),
  toolDryRun: (planId: string) => request<ToolDryRun>("POST", `/api/quantize/plan/${q(planId)}/tool-dry-run`),
  execute: (planId: string, token: string) =>
    request<Job>("POST", "/api/quantize/jobs", { plan_id: planId, confirm: true, confirm_token: token }),
  jobs: () => request<Job[]>("GET", "/api/quantize/jobs"),
  job: (id: string) => request<Job>("GET", `/api/quantize/jobs/${q(id)}`),
  cancel: (id: string) => request<Job>("POST", `/api/quantize/jobs/${q(id)}/cancel`),
};

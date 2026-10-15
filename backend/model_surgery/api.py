"""HTTP API. Bound to loopback by default.

Defences for a local web app:
  * Host-header allowlist (blocks DNS-rebinding)
  * state-changing requests must carry ``X-Model-Surgery: 1``; browsers cannot
    add that header cross-origin without a CORS preflight, which is never granted
  * no CORS headers at all, so other origins cannot read responses
  * strict CSP on the bundled frontend
  * no endpoint accepts a command, executable path or free-form tool argument
"""

from __future__ import annotations

import os
import string
import sys
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .config import Settings, load_settings
from .devices import DEFAULT_PROFILES, DeviceProfile
from .history import History
from .jobs import ExecuteRequest, Job, JobManager, Plan, PlanRequest
from .memory import SCHEMES, MemoryEstimate, MemoryRequest, estimate_memory
from .reports import Comparison, compare, comparison_markdown, markdown_report, tensors_csv
from .scanner import ScanError, scan_path
from .schemas import ScanResult, ScanSummary
from .tooling import ToolsReport

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
CSRF_HEADER = "x-model-surgery"
MODEL_SUFFIXES = {".gguf": "gguf", ".safetensors": "safetensors", ".json": "json", ".bin": "pickle", ".pt": "pickle", ".pth": "pickle", ".ckpt": "pickle", ".onnx": "onnx"}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "connect-src 'self'; font-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)


class _Guard(BaseHTTPMiddleware):
    def __init__(self, app, allowed_hosts: set[str]):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        hostname = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if hostname.lower() not in self.allowed_hosts:
            return PlainTextResponse("invalid host header", status_code=421)
        if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
            if request.headers.get(CSRF_HEADER) != "1":
                return JSONResponse({"detail": f"missing {CSRF_HEADER} header"}, status_code=403)
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Frame-Options"] = "DENY"
        if not request.url.path.startswith("/api/"):
            resp.headers["Content-Security-Policy"] = CSP
        return resp


class ScanRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class FsEntry(BaseModel):
    name: str
    path: str
    type: Literal["dir", "file"]
    kind: str | None = None
    size_bytes: int | None = None
    is_model_dir: bool = False


class FsListing(BaseModel):
    path: str
    parent: str | None
    entries: list[FsEntry]
    truncated: bool


class FsRoot(BaseModel):
    label: str
    path: str


def create_app(settings: Settings | None = None, job_manager: JobManager | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_dirs()
    history = History(settings.db_path)
    jobs = job_manager or JobManager(settings, history)

    app = FastAPI(title="Model Surgery", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.settings = settings
    app.state.history = history
    app.state.jobs = jobs
    app.add_middleware(_Guard, allowed_hosts=LOOPBACK_HOSTS | {h.lower() for h in settings.extra_allowed_hosts})

    @app.exception_handler(ScanError)
    async def _scan_err(_: Request, e: ScanError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(LookupError)
    async def _lookup(_: Request, e: LookupError):
        return JSONResponse({"detail": str(e).strip("'\"")}, status_code=404)

    @app.exception_handler(PermissionError)
    async def _perm(_: Request, e: PermissionError):
        return JSONResponse({"detail": str(e)}, status_code=403)

    @app.exception_handler(RuntimeError)
    async def _conflict(_: Request, e: RuntimeError):
        return JSONResponse({"detail": str(e)}, status_code=409)

    def _scan(scan_id: str) -> ScanResult:
        r = history.get_scan(scan_id)
        if r is None:
            raise HTTPException(404, "scan not found")
        return r

    # -- meta --------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "platform": sys.platform, "data_dir": str(settings.data_dir), "outputs_dir": str(settings.outputs_dir)}

    # -- filesystem browsing (read-only, listing only) --------------------------
    @app.get("/api/fs/roots", response_model=list[FsRoot])
    def fs_roots() -> list[FsRoot]:
        roots: list[FsRoot] = []
        if settings.allowed_roots:
            roots += [FsRoot(label=str(r), path=str(r)) for r in settings.allowed_roots if r.is_dir()]
        else:
            roots.append(FsRoot(label="Home", path=str(Path.home())))
            if sys.platform == "win32":
                roots += [FsRoot(label=f"{d}:", path=f"{d}:\\") for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            else:
                roots.append(FsRoot(label="/", path="/"))
        roots.append(FsRoot(label="Model Surgery outputs", path=str(settings.outputs_dir)))
        return roots

    @app.get("/api/fs/list", response_model=FsListing)
    def fs_list(path: str = Query(min_length=1, max_length=4096)) -> FsListing:
        p = Path(path).expanduser()
        if not p.is_absolute() or not p.is_dir():
            raise HTTPException(400, "not an existing absolute directory path")
        p = p.resolve()
        if not (settings.is_allowed(p) or p == settings.outputs_dir.resolve()):
            raise HTTPException(403, "directory is outside the configured allowed roots")
        entries: list[FsEntry] = []
        truncated = False
        try:
            items = sorted(os.scandir(p), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError as e:
            raise HTTPException(403, f"cannot list directory: {e.strerror}") from None
        for e in items:
            if len(entries) >= 2000:
                truncated = True
                break
            if e.name.startswith((".", "$")):
                continue
            try:
                if e.is_dir():
                    entries.append(FsEntry(name=e.name, path=str(Path(e.path)), type="dir", is_model_dir=os.path.isfile(os.path.join(e.path, "config.json"))))
                elif e.is_file():
                    kind = MODEL_SUFFIXES.get(Path(e.name).suffix.lower())
                    if kind:
                        entries.append(FsEntry(name=e.name, path=str(Path(e.path)), type="file", kind=kind, size_bytes=e.stat().st_size))
            except OSError:
                continue
        parent = str(p.parent) if p.parent != p else None
        return FsListing(path=str(p), parent=parent, entries=entries, truncated=truncated)

    # -- scans --------------------------------------------------------------
    @app.post("/api/scan", response_model=ScanResult)
    def scan(req: ScanRequest) -> ScanResult:
        r = scan_path(req.path, settings)
        history.save_scan(r)
        return r

    @app.get("/api/scans", response_model=list[ScanSummary])
    def list_scans() -> list[ScanSummary]:
        return history.list_scans()

    @app.get("/api/scans/{scan_id}", response_model=ScanResult)
    def get_scan(scan_id: str) -> ScanResult:
        return _scan(scan_id)

    @app.delete("/api/scans/{scan_id}")
    def delete_scan(scan_id: str) -> dict:
        if not history.delete_scan(scan_id):
            raise HTTPException(404, "scan not found")
        return {"deleted": scan_id}

    @app.get("/api/scans/{scan_id}/export")
    def export(scan_id: str, format: Literal["json", "csv", "md"] = "json") -> Response:
        r = _scan(scan_id)
        stem = "".join(c if c.isalnum() or c in "-_." else "-" for c in r.name)[:60] or "model"
        if format == "json":
            body, media = r.model_dump_json(indent=2), "application/json"
        elif format == "csv":
            body, media = tensors_csv(r), "text/csv; charset=utf-8"
        else:
            body, media = markdown_report(r), "text/markdown; charset=utf-8"
        suffix = {"json": "scan.json", "csv": "tensors.csv", "md": "report.md"}[format]
        return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{stem}.{suffix}"'})

    # -- memory -------------------------------------------------------------
    @app.get("/api/devices", response_model=list[DeviceProfile])
    def devices() -> list[DeviceProfile]:
        return DEFAULT_PROFILES

    @app.get("/api/schemes")
    def schemes() -> list[dict]:
        return [{"id": s.id, "label": s.label, "family": s.family, "ftype": s.ftype} for s in SCHEMES]

    @app.post("/api/scans/{scan_id}/memory", response_model=MemoryEstimate)
    def memory(scan_id: str, req: MemoryRequest = Body(...)) -> MemoryEstimate:
        return estimate_memory(_scan(scan_id), req)

    @app.post("/api/scans/{scan_id}/report.md", response_class=PlainTextResponse)
    def report_with_memory(scan_id: str, req: MemoryRequest = Body(...)) -> str:
        r = _scan(scan_id)
        return markdown_report(r, estimate_memory(r, req))

    # -- comparison ---------------------------------------------------------
    @app.get("/api/compare", response_model=Comparison)
    def compare_scans(a: str, b: str, context_length: int = Query(4096, ge=1, le=10_000_000), device_id: str = "laptop-cpu-16gb"):
        return compare(_scan(a), _scan(b), MemoryRequest(context_length=context_length, device_id=device_id))

    @app.get("/api/compare.md", response_class=PlainTextResponse)
    def compare_md(a: str, b: str, context_length: int = Query(4096, ge=1, le=10_000_000), device_id: str = "laptop-cpu-16gb") -> str:
        return comparison_markdown(compare(_scan(a), _scan(b), MemoryRequest(context_length=context_length, device_id=device_id)))

    # -- tools & jobs -------------------------------------------------------------
    @app.get("/api/tools", response_model=ToolsReport)
    def tools(refresh: bool = False) -> ToolsReport:
        return jobs.tools(refresh)

    @app.post("/api/quantize/plan", response_model=Plan)
    def plan(req: PlanRequest) -> Plan:
        return jobs.create_plan(req, _scan(req.scan_id))

    @app.post("/api/quantize/plan/{plan_id}/tool-dry-run")
    def tool_dry_run(plan_id: str) -> dict:
        try:
            return jobs.tool_dry_run(plan_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/quantize/jobs", response_model=Job)
    def execute(req: ExecuteRequest) -> Job:
        return jobs.execute(req)

    @app.get("/api/quantize/jobs", response_model=list[Job])
    def list_jobs() -> list[Job]:
        return jobs.list()

    @app.get("/api/quantize/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        return jobs.get(job_id)

    @app.post("/api/quantize/jobs/{job_id}/cancel", response_model=Job)
    def cancel_job(job_id: str) -> Job:
        return jobs.cancel(job_id)

    # -- bundled frontend ---------------------------------------------------------
    if settings.frontend_dist and (settings.frontend_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    else:

        @app.get("/", response_class=PlainTextResponse)
        def no_frontend() -> str:
            return "Model Surgery API is running. Build the frontend (cd frontend && npm run build) or use the Vite dev server. API docs: /api/docs"

    return app

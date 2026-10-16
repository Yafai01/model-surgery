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

import asyncio
import json
import os
import string
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from urllib.parse import quote
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
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
from .runtime import GenerateRequest, Measurement, RuntimeManager, RuntimeStartRequest, RuntimeStatus, TokenizeRequest
from .scanner import ScanError, scan_path
from .schemas import ScanResult, ScanSummary
from .system import system_memory
from .tooling import ToolsReport

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
MAX_IMPORT_BYTES = 64 * 1024 * 1024
CSRF_HEADER = "x-model-surgery"
MODEL_SUFFIXES = {".gguf": "gguf", ".safetensors": "safetensors", ".json": "json", ".bin": "pickle", ".pt": "pickle", ".pth": "pickle", ".ckpt": "pickle", ".onnx": "onnx"}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co; font-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
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


def create_app(settings: Settings | None = None, job_manager: JobManager | None = None, runtime_manager: RuntimeManager | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_dirs()
    history = History(settings.db_path)
    jobs = job_manager or JobManager(settings, history)
    runtime = runtime_manager or RuntimeManager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        runtime.shutdown()  # never leave a llama-server process behind

    app = FastAPI(title="Model Surgery", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.history = history
    app.state.jobs = jobs
    app.state.runtime = runtime
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

    @app.post("/api/scans/import", response_model=ScanResult)
    async def import_remote_scan(request: Request) -> ScanResult:
        """Store a scan the page produced by reading a model from the Hugging Face Hub.

        Only remote (hf://) scans are accepted, so an imported record can never point at a local path
        that the quantization workflow would act on.
        """
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_IMPORT_BYTES):
            raise HTTPException(413, "scan too large")
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_IMPORT_BYTES:
                raise HTTPException(413, "scan too large")
        try:
            r = ScanResult.model_validate_json(body)
        except ValueError as e:
            raise HTTPException(422, f"invalid scan: {str(e)[:300]}") from None
        if not r.is_remote or not all(f.path.startswith("hf://") for f in r.files):
            raise HTTPException(400, "only Hugging Face Hub (hf://) scans can be imported")
        r.id = uuid.uuid4().hex  # never trust a client-chosen id: it could overwrite an existing local scan
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
        # HTTP headers are Latin-1: keep the fallback filename ASCII and send the real name as RFC 5987 filename*.
        stem = "".join(c if (c.isascii() and c.isalnum()) or c in "-_." else "-" for c in r.name)[:60].strip("-.") or "model"
        if format == "json":
            body, media = r.model_dump_json(indent=2), "application/json"
        elif format == "csv":
            body, media = tensors_csv(r), "text/csv; charset=utf-8"
        else:
            body, media = markdown_report(r), "text/markdown; charset=utf-8"
        suffix = {"json": "scan.json", "csv": "tensors.csv", "md": "report.md"}[format]
        utf8_name = quote(f"{r.name[:60]}.{suffix}", safe="")
        return Response(body, media_type=media, headers={"Content-Disposition": f"attachment; filename=\"{stem}.{suffix}\"; filename*=UTF-8''{utf8_name}"})

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
    def tools() -> ToolsReport:
        return jobs.tools(False)

    # Re-detection runs subprocesses, so it is a POST (CSRF header required), not a GET any page could trigger.
    @app.post("/api/tools/refresh", response_model=ToolsReport)
    def tools_refresh() -> ToolsReport:
        return jobs.tools(True)

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

    # -- local inference (llama-server) ---------------------------------------------
    @app.get("/api/runtime", response_model=RuntimeStatus)
    def runtime_status() -> RuntimeStatus:
        return runtime.status()

    @app.post("/api/runtime/start", response_model=RuntimeStatus)
    def runtime_start(req: RuntimeStartRequest) -> RuntimeStatus:
        return runtime.start(req, _scan(req.scan_id))

    @app.post("/api/runtime/stop", response_model=RuntimeStatus)
    def runtime_stop() -> RuntimeStatus:
        return runtime.stop()

    @app.post("/api/runtime/tokenize")
    def runtime_tokenize(req: TokenizeRequest) -> dict:
        return runtime.tokenize(req)

    @app.post("/api/runtime/cancel")
    def runtime_cancel() -> dict:
        return {"cancelled": runtime.cancel_generation()}

    @app.get("/api/runtime/measurements", response_model=list[Measurement])
    def runtime_measurements() -> list[Measurement]:
        return runtime.measurements()

    @app.post("/api/runtime/generate")
    async def runtime_generate(req: GenerateRequest, request: Request) -> StreamingResponse:
        gen = runtime.generate(req)  # raises 409 before the stream starts if not ready / busy
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def pump() -> None:
            for ev in gen.events():
                loop.call_soon_threadsafe(queue.put_nowait, ev)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=pump, name=f"generate-{gen.id}", daemon=True).start()

        async def body():
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            gen.cancel()
                        continue
                    if ev is None:
                        return
                    yield json.dumps(ev) + "\n"
            finally:
                gen.cancel()  # client went away or the stream ended; no-op once finished

        return StreamingResponse(body(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    # -- system -------------------------------------------------------------------
    @app.get("/api/system")
    def system() -> dict:
        return {
            "version": __version__,
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "memory": system_memory().model_dump(),
            "data_dir": str(settings.data_dir),
            "outputs_dir": str(settings.outputs_dir),
            "allowed_roots": [str(r) for r in settings.allowed_roots],
            "llama_cpp_bin_dir": str(settings.llama_cpp_bin_dir) if settings.llama_cpp_bin_dir else None,
            "network": "Model Surgery itself makes no outbound requests. The page contacts huggingface.co only when you inspect a Hub link.",
        }

    # -- bundled frontend ---------------------------------------------------------
    if settings.frontend_dist and (settings.frontend_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    else:

        @app.get("/", response_class=PlainTextResponse)
        def no_frontend() -> str:
            return "Model Surgery API is running. Build the frontend (cd frontend && npm run build) or use the Vite dev server. API docs: /api/docs"

    return app

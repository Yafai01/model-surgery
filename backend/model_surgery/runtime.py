"""Optional local inference through llama.cpp's ``llama-server``.

Everything this module reports comes from the runtime or the operating system:

  * generated text, streamed token by token
  * token ids and pieces (``/tokenize`` and ``return_tokens``)
  * per-token log-probabilities and top-k alternatives (``n_probs``)
  * prompt-processing and decode timings measured inside llama.cpp (``timings``)
  * buffer allocations that llama.cpp logs while loading (weights, KV cache, compute)
  * the resident memory of the server process, sampled from the OS

llama-server has no API for intermediate activations, attention maps or
per-layer timings, so none are produced here, and the UI says so instead of
animating anything that looks like a forward-pass trace.

Safety: the executable comes from server-side configuration only (basename
allowlisted, fixed argv, no shell). The server binds to 127.0.0.1 on a random
port and requires a random API key that never leaves this process, because
llama-server answers cross-origin requests and would otherwise be reachable by
any web page the user visits. Only one model is loaded at a time.
"""

from __future__ import annotations

import http.client
import json
import re
import atexit
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .memory import MemoryRequest, estimate_memory
from .schemas import ScanResult
from .system import process_memory, system_memory
from .tooling import NO_WINDOW, ServerTool, detect_server

MiB = 1024**2

_ALLOC_RE = re.compile(r"^\s*[\w.]+:\s+(\S+)\s+(model|KV|output|compute|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
_OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_NCTX_RE = re.compile(r"llama_context:\s+n_ctx\s+=\s+(\d+)")
_LISTENING = "server is listening"

TRACE_NOTE = (
    "llama-server exposes the token stream, token ids, top-k log-probabilities and its own timings. "
    "It has no API for intermediate activations, attention maps or per-layer timings, so Model Surgery "
    "does not show a layer-by-layer forward-pass trace."
)

CAPABILITIES = {
    "token_stream": True,
    "token_ids": True,
    "top_k_logprobs": True,
    "runtime_timings": True,
    "runtime_allocations": True,
    "layer_trace": False,
    "attention_maps": False,
    "activations": False,
}


class RuntimeBlocked(RuntimeError):
    """A start/generate request that cannot proceed in the current state (HTTP 409)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Allocation(BaseModel):
    kind: str  # model | KV | output | compute | RS
    device: str  # CPU, CPU_Mapped, CUDA0, CUDA_Host, Metal, ...
    bytes: int
    line: str


class RuntimeStatus(BaseModel):
    state: Literal["unavailable", "stopped", "starting", "ready", "failed"]
    tool: ServerTool
    scan_id: str | None = None
    model_name: str | None = None
    context_length: int | None = None
    n_ctx_runtime: int | None = None
    n_gpu_layers_requested: int | None = None
    gpu_layers_offloaded: int | None = None
    total_layers: int | None = None
    pid: int | None = None
    started_at: str | None = None
    load_seconds: float | None = None
    allocations: list[Allocation] = Field(default_factory=list)
    process_rss_bytes: int | None = None
    generating: bool = False
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=lambda: dict(CAPABILITIES))
    trace_note: str = TRACE_NOTE


class RuntimeStartRequest(BaseModel):
    scan_id: str = Field(min_length=1, max_length=100)
    context_length: int = Field(4096, ge=128, le=262_144)
    n_gpu_layers: int | None = Field(None, ge=0, le=1000)  # None = llama.cpp decides (fit to free memory)
    threads: int | None = Field(None, ge=1, le=256)
    allow_memory_risk: bool = False


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    chat: bool = True
    system: str | None = Field(None, max_length=8_000)
    n_predict: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0, le=2)
    top_k: int = Field(40, ge=0, le=1000)
    top_p: float = Field(0.95, gt=0, le=1)
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    n_probs: int = Field(5, ge=0, le=10)


class TokenizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=32_000)
    add_special: bool = True


class Measurement(BaseModel):
    """One completed generation. Runtime timings come from llama.cpp; wall-clock values from this process."""

    at: str
    scan_id: str | None
    model_name: str | None
    n_ctx: int | None
    gpu_layers_offloaded: int | None
    prompt_tokens: int | None
    prompt_ms: float | None
    prompt_tokens_per_s: float | None
    generated_tokens: int | None
    decode_ms: float | None
    decode_tokens_per_s: float | None
    wall_first_token_ms: float | None
    wall_total_ms: float | None
    process_rss_bytes: int | None
    stop_type: str | None


def _kill_on_close_job():
    """Windows: a Job Object that kills its processes when the last handle closes, i.e. when this process dies.

    Without it, a crash or a forced kill of Model Surgery would leave llama-server running with the model in memory.
    Returns the job handle (keep it alive) or None if unavailable.
    """
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC),
            ("IoInfo", IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):  # JobObjectExtendedLimitInformation
        kernel32.CloseHandle(wintypes.HANDLE(job))
        return None
    return job


def _assign_to_job(job, proc: subprocess.Popen) -> bool:
    import ctypes
    from ctypes import wintypes

    handle = getattr(proc, "_handle", None)
    if not job or handle is None:
        return False
    return bool(ctypes.WinDLL("kernel32").AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(handle))))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class Generation:
    """A single streamed completion. ``events()`` yields plain dicts; ``cancel()`` may be called from any thread."""

    def __init__(self, mgr: RuntimeManager, req: GenerateRequest):
        self.id = secrets.token_hex(8)
        self.mgr = mgr
        self.req = req
        self.cancelled = False
        self._conn: http.client.HTTPConnection | None = None
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            conn, sock = self._conn, self._sock
        # http.client detaches the socket from the connection when the server sends "Connection: close",
        # so shut down our own reference to it; that unblocks the reader thread immediately.
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if conn is not None:
            conn.close()

    def events(self) -> Iterator[dict[str, Any]]:
        t0 = time.perf_counter()
        first_token_ms: float | None = None
        try:
            r = self.req
            text = r.prompt
            if r.chat:
                messages = ([{"role": "system", "content": r.system}] if r.system else []) + [{"role": "user", "content": r.prompt}]
                status, body = self.mgr._request("POST", "/apply-template", {"messages": messages})
                if status == 200 and isinstance(body.get("prompt"), str):
                    text = body["prompt"]
                else:
                    yield {"type": "notice", "message": "this model has no usable chat template; the prompt was sent as raw text"}
            # Tokenize first so the prompt tokens shown are exactly the ones evaluated.
            # A chat template already contains BOS, so only raw prompts get special tokens added.
            status, body = self.mgr._request("POST", "/tokenize", {"content": text, "add_special": not r.chat, "with_pieces": True})
            if status != 200 or not isinstance(body.get("tokens"), list):
                raise RuntimeBlocked(f"tokenize failed (HTTP {status})")
            toks = body["tokens"]
            ids = [t["id"] if isinstance(t, dict) else t for t in toks]
            n_ctx = self.mgr.status().n_ctx_runtime or self.mgr.status().context_length
            if n_ctx and len(ids) + r.n_predict > n_ctx:
                raise RuntimeBlocked(
                    f"prompt ({len(ids)} tokens) + max new tokens ({r.n_predict}) exceeds the loaded context of {n_ctx} tokens; "
                    "shorten the prompt, lower max tokens or restart the runtime with a larger context"
                )
            yield {
                "type": "prompt",
                "formatted": text[:20_000],
                "chat_template_applied": r.chat and text != r.prompt,
                "n_tokens": len(ids),
                "tokens": [t if isinstance(t, dict) else {"id": t, "piece": None} for t in toks[:4096]],
            }
            payload: dict[str, Any] = {
                "prompt": ids,
                "n_predict": r.n_predict,
                "stream": True,
                "return_tokens": True,
                "n_probs": r.n_probs,
                "temperature": r.temperature,
                "top_k": r.top_k,
                "top_p": r.top_p,
                "cache_prompt": True,
            }
            if r.seed is not None:
                payload["seed"] = r.seed
            conn = self.mgr._connection(timeout=600)
            conn.connect()
            with self._lock:
                if self.cancelled:
                    conn.close()
                    raise _Cancelled
                self._conn, self._sock = conn, conn.sock
            conn.request("POST", "/completion", body=json.dumps(payload), headers=self.mgr._headers())
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeBlocked(f"llama-server returned HTTP {resp.status}: {resp.read(2000).decode('utf-8', 'replace')}")
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                if "error" in ev:
                    raise RuntimeBlocked(f"llama-server error: {ev['error']}")
                elapsed = (time.perf_counter() - t0) * 1000
                if not ev.get("stop"):
                    if first_token_ms is None:
                        first_token_ms = elapsed
                    yield {
                        "type": "token",
                        "content": ev.get("content", ""),
                        "ids": ev.get("tokens", []),
                        "probs": [
                            {
                                "id": p.get("id"),
                                "token": p.get("token"),
                                "logprob": p.get("logprob"),
                                "top": [{"id": a.get("id"), "token": a.get("token"), "logprob": a.get("logprob")} for a in p.get("top_logprobs", [])],
                            }
                            for p in ev.get("completion_probabilities", []) or []
                        ],
                        "t_ms": round(elapsed, 2),
                    }
                    continue
                timings = ev.get("timings") or {}
                total_ms = (time.perf_counter() - t0) * 1000
                done = {
                    "type": "done",
                    "stop_type": ev.get("stop_type"),
                    "truncated": bool(ev.get("truncated")),
                    "tokens_predicted": ev.get("tokens_predicted"),
                    "tokens_evaluated": ev.get("tokens_evaluated"),
                    "tokens_cached": ev.get("tokens_cached"),
                    "timings": {k: timings.get(k) for k in ("cache_n", "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second")},
                    "wall": {"first_token_ms": None if first_token_ms is None else round(first_token_ms, 2), "total_ms": round(total_ms, 2)},
                    "sampling": {k: (ev.get("generation_settings") or {}).get(k) for k in ("temperature", "top_k", "top_p", "seed", "n_probs")},
                }
                self.mgr._record(done)
                yield done
                return
            if self.cancelled:
                raise _Cancelled
            raise RuntimeBlocked("llama-server closed the stream before finishing")
        except _Cancelled:
            yield {"type": "cancelled", "wall": {"first_token_ms": first_token_ms, "total_ms": round((time.perf_counter() - t0) * 1000, 2)}}
        except (OSError, http.client.HTTPException) as e:
            if self.cancelled:
                yield {"type": "cancelled", "wall": {"first_token_ms": first_token_ms, "total_ms": round((time.perf_counter() - t0) * 1000, 2)}}
            else:
                yield {"type": "error", "message": f"connection to llama-server failed: {e}"}
        except RuntimeBlocked as e:
            yield {"type": "error", "message": str(e)}
        finally:
            with self._lock:
                conn, self._conn, self._sock = self._conn, None, None
            if conn is not None:
                conn.close()
            self.mgr._generation_finished(self)


class _Cancelled(Exception):
    pass


class RuntimeManager:
    def __init__(
        self,
        settings: Settings,
        tool_fn: Callable[[bool], ServerTool] | None = None,
        popen: Callable[..., subprocess.Popen] | None = None,
        load_timeout: float = 600.0,
    ):
        self.settings = settings
        self._tool_fn = tool_fn or (lambda refresh=False: detect_server(settings, refresh))
        self._popen = popen or subprocess.Popen
        self.load_timeout = load_timeout
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._key: str | None = None
        self._log: deque[str] = deque(maxlen=400)
        self._st = RuntimeStatus(state="stopped", tool=ServerTool(available=False, detail="not checked"))
        self._gen: Generation | None = None
        self._measurements: deque[Measurement] = deque(maxlen=100)
        self._job = _kill_on_close_job() if sys.platform == "win32" else None
        atexit.register(self.shutdown)  # normal interpreter exit; the job object also covers crashes on Windows

    # -- status -------------------------------------------------------------------
    def status(self, refresh_tool: bool = False) -> RuntimeStatus:
        tool = self._tool_fn(refresh_tool)
        with self._lock:
            st = self._st.model_copy(deep=True)
            proc = self._proc
            st.generating = self._gen is not None
            st.log_tail = list(self._log)[-120:]
        st.tool = tool
        if st.state == "stopped" and not tool.available:
            st.state = "unavailable"
        if proc is not None and proc.poll() is None and st.state in ("starting", "ready"):
            st.process_rss_bytes = process_memory(proc.pid)
        return st

    def measurements(self) -> list[Measurement]:
        with self._lock:
            return list(self._measurements)[::-1]

    # -- lifecycle ------------------------------------------------------------------
    def start(self, req: RuntimeStartRequest, scan: ScanResult) -> RuntimeStatus:
        tool = self._tool_fn(False)
        if not tool.available or not tool.path:
            raise RuntimeBlocked(f"llama-server is not available: {tool.detail or 'not found'}")
        if scan.kind != "gguf" or scan.is_remote:
            raise RuntimeBlocked("inference needs a local GGUF file; safetensors and Hugging Face links can be inspected but not run here")
        if not scan.integrity.valid:
            raise RuntimeBlocked("this file failed integrity checks; refusing to load it")
        model = Path(scan.files[0].path)
        if not model.is_file() or model.stat().st_size != scan.files[0].size_bytes:
            raise RuntimeBlocked("the model file is missing or has changed since it was scanned; scan it again")
        if not self.settings.is_allowed(model):
            raise PermissionError("model is outside the configured allowed roots")

        est = estimate_memory(scan, MemoryRequest(context_length=req.context_length, device_id=None))
        mem = system_memory()
        if mem.available_bytes is not None and est.total_bytes > mem.available_bytes and not req.allow_memory_risk:
            raise RuntimeBlocked(
                f"estimated requirement {est.total_bytes / 2**30:.2f} GiB (weights + KV cache + buffers, an estimate) exceeds the "
                f"{mem.available_bytes / 2**30:.2f} GiB of RAM the OS reports as available right now. Lower the context length, "
                "offload layers to a GPU, close other programs, or confirm that you want to try anyway."
            )

        self.stop()
        port, key = _free_port(), secrets.token_urlsafe(24)
        argv = [tool.path, "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(req.context_length), "-np", "1", "--no-webui", "--api-key", key]
        if req.n_gpu_layers is not None:
            argv += ["-ngl", str(req.n_gpu_layers)]
        if req.threads is not None:
            argv += ["-t", str(req.threads)]
        try:
            proc = self._popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                cwd=str(self.settings.data_dir),
                creationflags=NO_WINDOW,
            )
        except OSError as e:
            raise RuntimeBlocked(f"could not start llama-server: {e}") from None
        if self._job is not None:
            _assign_to_job(self._job, proc)
        with self._lock:
            self._proc, self._port, self._key = proc, port, key
            self._log.clear()
            self._st = RuntimeStatus(
                state="starting",
                tool=tool,
                scan_id=scan.id,
                model_name=scan.name,
                context_length=req.context_length,
                n_gpu_layers_requested=req.n_gpu_layers,
                pid=proc.pid,
                started_at=_now(),
            )
        t0 = time.perf_counter()
        threading.Thread(target=self._read_output, args=(proc,), name="llama-server-log", daemon=True).start()
        threading.Thread(target=self._wait_ready, args=(proc, t0), name="llama-server-health", daemon=True).start()
        return self.status()

    def stop(self) -> RuntimeStatus:
        with self._lock:
            proc, gen = self._proc, self._gen
            self._proc = None
        if gen is not None:
            gen.cancel()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        with self._lock:
            if proc is not None or self._st.state != "failed":
                self._st = RuntimeStatus(state="stopped", tool=self._st.tool)
            self._port = self._key = None
        return self.status()

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001 - best effort at interpreter exit
            pass

    def _read_output(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            with self._lock:
                if self._proc is not proc:
                    continue
                self._log.append(line)
                st = self._st
                if m := _ALLOC_RE.match(line):
                    st.allocations.append(Allocation(kind=m.group(2), device=m.group(1), bytes=int(float(m.group(3)) * MiB), line=line.strip()))
                elif m := _OFFLOAD_RE.search(line):
                    st.gpu_layers_offloaded, st.total_layers = int(m.group(1)), int(m.group(2))
                elif m := _NCTX_RE.search(line):
                    st.n_ctx_runtime = int(m.group(1))

    def _wait_ready(self, proc: subprocess.Popen, t0: float) -> None:
        deadline = time.monotonic() + self.load_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                with self._lock:
                    if self._proc is proc:
                        self._proc = None
                        tail = [l for l in list(self._log)[-30:] if re.search(r"error|failed|unable|cannot|out of memory", l, re.I)] or list(self._log)[-5:]
                        self._st.state = "failed"
                        self._st.error = f"llama-server exited with code {proc.returncode} while loading: " + " | ".join(tail[-5:])
                return
            try:
                status, _ = self._request("GET", "/health", timeout=2)
            except (OSError, http.client.HTTPException, RuntimeBlocked):
                status = 0
            if status == 200:
                with self._lock:
                    if self._proc is proc:
                        self._st.state = "ready"
                        self._st.load_seconds = round(time.perf_counter() - t0, 2)
                threading.Thread(target=self._watch_exit, args=(proc,), name="llama-server-exit", daemon=True).start()
                return
            time.sleep(0.25)
        with self._lock:
            still_ours = self._proc is proc
        if still_ours:
            self.stop()
            with self._lock:
                self._st.state = "failed"
                self._st.error = f"llama-server did not become ready within {self.load_timeout:.0f}s"

    def _watch_exit(self, proc: subprocess.Popen) -> None:
        code = proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None
                self._st.state = "failed"
                self._st.error = f"llama-server exited unexpectedly with code {code}"

    # -- requests -------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}

    def _connection(self, timeout: float = 30) -> http.client.HTTPConnection:
        with self._lock:
            port = self._port
        if port is None:
            raise RuntimeBlocked("no model is loaded")
        return http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    def _request(self, method: str, path: str, body: Any = None, timeout: float = 60) -> tuple[int, dict[str, Any]]:
        conn = self._connection(timeout)
        try:
            conn.request(method, path, body=None if body is None else json.dumps(body), headers=self._headers())
            resp = conn.getresponse()
            raw = resp.read(8 * MiB)
            try:
                data = json.loads(raw) if raw else {}
            except ValueError:
                data = {}
            return resp.status, data if isinstance(data, dict) else {"value": data}
        finally:
            conn.close()

    def _require_ready(self) -> None:
        with self._lock:
            if self._st.state != "ready" or self._proc is None:
                raise RuntimeBlocked("no model is loaded and ready; start the runtime first")

    def tokenize(self, req: TokenizeRequest) -> dict[str, Any]:
        self._require_ready()
        status, body = self._request("POST", "/tokenize", {"content": req.text, "add_special": req.add_special, "with_pieces": True})
        if status != 200:
            raise RuntimeBlocked(f"tokenize failed (HTTP {status})")
        toks = body.get("tokens", [])
        return {"n_tokens": len(toks), "tokens": toks[:8192], "truncated": len(toks) > 8192}

    def generate(self, req: GenerateRequest) -> Generation:
        self._require_ready()
        with self._lock:
            if self._gen is not None:
                raise RuntimeBlocked("a generation is already running; stop it first")
            self._gen = Generation(self, req)
            return self._gen

    def cancel_generation(self) -> bool:
        with self._lock:
            gen = self._gen
        if gen is None:
            return False
        gen.cancel()
        return True

    def _generation_finished(self, gen: Generation) -> None:
        with self._lock:
            if self._gen is gen:
                self._gen = None

    def _record(self, done: dict[str, Any]) -> None:
        st = self.status()
        t, w = done["timings"], done["wall"]
        m = Measurement(
            at=_now(),
            scan_id=st.scan_id,
            model_name=st.model_name,
            n_ctx=st.n_ctx_runtime or st.context_length,
            gpu_layers_offloaded=st.gpu_layers_offloaded,
            prompt_tokens=t.get("prompt_n"),
            prompt_ms=_num(t.get("prompt_ms")),
            prompt_tokens_per_s=_num(t.get("prompt_per_second")),
            generated_tokens=t.get("predicted_n"),
            decode_ms=_num(t.get("predicted_ms")),
            decode_tokens_per_s=_num(t.get("predicted_per_second")),
            wall_first_token_ms=w.get("first_token_ms"),
            wall_total_ms=w.get("total_ms"),
            process_rss_bytes=st.process_rss_bytes,
            stop_type=done.get("stop_type"),
        )
        with self._lock:
            self._measurements.append(m)

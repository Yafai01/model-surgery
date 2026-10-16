"""Quantization / conversion workflow.

Flow: plan (dry-run summary, nothing executed) -> explicit confirmation with a
single-use token -> subprocess with a fixed argv -> output validation -> rename
into place. A job is reported as succeeded only after the output file has been
parsed and checked against the source.

Safety rules enforced here:
  * argv is built server-side from validated values; no shell is ever used
  * output goes to a directory separate from the source model
  * existing files are never overwritten; work happens in a ``.partial`` file
  * one heavy job at a time
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings
from .formats import FormatError
from .formats.ggml_types import LLAMA_FTYPE_GUESSED, LLAMA_FTYPES
from .formats.gguf import read_gguf
from .history import History
from .memory import estimate_tensor_bytes
from .scanner import ScanError, scan_path
from .schemas import ScanResult
from .tooling import NO_WINDOW, ToolsReport, detect_tools, run_fixed

PLAN_TTL_SECONDS = 15 * 60
IMATRIX_REQUIRED = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K_S"}
TERNARY_ONLY = {"TQ1_0", "TQ2_0"}
CONVERT_OUTTYPE_FTYPE = {"f32": 0, "f16": 1, "bf16": 32, "q8_0": 7, "tq1_0": 36, "tq2_0": 37}
_PROGRESS_QUANT = re.compile(r"^\[\s*(\d+)\s*/\s*(\d+)\]\s+(\S+)")
_PROGRESS_PCT = re.compile(r"(\d{1,3})%\|")
_DRY_SIZE = re.compile(r"(model|quant) size\s*=\s*([\d.]+)\s*MiB\s*\(([\d.]+)\s*BPW\)")


_NON_TEXT = re.compile(r"(vision|visual|image|audio|mm_projector|multi_modal_projector|mmproj)", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-.")[:80] or "model"


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- models


class PlanRequest(BaseModel):
    scan_id: str
    operation: Literal["quantize", "convert"]
    target: str = Field(min_length=1, max_length=32)
    output_dir: str | None = Field(default=None, max_length=4096)
    allow_requantize: bool = False
    threads: int | None = Field(default=None, ge=1, le=256)


class Plan(BaseModel):
    plan_id: str
    created_at: str
    expires_at: float
    operation: Literal["quantize", "convert"]
    scan_id: str
    source_path: str
    source_label: str | None
    target: str
    output_dir: str | None
    output_path: str | None
    partial_path: str | None
    argv: list[str]
    tool: str | None
    tool_version: str | None
    estimated_output_bytes: int | None
    estimate_note: str
    free_disk_bytes: int | None
    warnings: list[str]
    blockers: list[str]
    can_execute: bool
    confirm_token: str | None
    supports_tool_dry_run: bool


class ExecuteRequest(BaseModel):
    plan_id: str
    confirm: bool
    confirm_token: str


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class ValidationReport(BaseModel):
    passed: bool
    checks: list[ValidationCheck]


JobStatus = Literal["running", "validating", "succeeded", "failed", "cancelled", "interrupted"]


class Job(BaseModel):
    id: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    status: JobStatus
    plan: Plan
    progress: float | None = None
    progress_label: str | None = None
    exit_code: int | None = None
    log_tail: list[str] = Field(default_factory=list)
    output_path: str | None = None
    output_size_bytes: int | None = None
    validation: ValidationReport | None = None
    output_scan_id: str | None = None
    error: str | None = None


class _JobState:
    def __init__(self, job: Job):
        self.job = job
        self.log: deque[str] = deque(maxlen=3000)
        self.proc: subprocess.Popen | None = None
        self.cancel_requested = False


@dataclass
class _StoredPlan:
    plan: Plan
    scan: ScanResult
    fingerprint: list[tuple[str, int, int]]
    expected_ftype: int | None


def _fingerprint(scan: ScanResult) -> list[tuple[str, int, int]]:
    out = []
    for f in scan.files:
        if f.parsed:
            st = Path(f.path).stat()
            out.append((f.path, st.st_size, st.st_mtime_ns))
    return out


def parse_dry_run(text: str) -> dict:
    sizes = {m.group(1): {"mib": float(m.group(2)), "bpw": float(m.group(3))} for m in _DRY_SIZE.finditer(text)}
    fallback = re.search(r"WARNING:\s*(\d+) of (\d+) tensor\(s\) required fallback", text)
    errors = [l for l in text.splitlines() if "error" in l.lower() or "failed" in l.lower()][-5:]
    return {
        "model_size_mib": sizes.get("model", {}).get("mib"),
        "quant_size_mib": sizes.get("quant", {}).get("mib"),
        "quant_bpw": sizes.get("quant", {}).get("bpw"),
        "fallback_tensors": int(fallback.group(1)) if fallback else 0,
        "errors": errors,
    }


# --------------------------------------------------------------------------- validation


def validate_output(
    operation: str, path: Path, source: ScanResult, expected_ftype: int | None, exit_code: int | None
) -> ValidationReport:
    checks: list[ValidationCheck] = []

    def add(name: str, ok: bool, detail: str) -> bool:
        checks.append(ValidationCheck(name=name, passed=ok, detail=detail))
        return ok

    add("exit status", exit_code == 0, f"process exited with code {exit_code}")
    if not add("output exists", path.is_file() and path.stat().st_size > 0, str(path)):
        return ValidationReport(passed=False, checks=checks)
    try:
        g = read_gguf(path)
    except FormatError as e:
        add("GGUF readable", False, str(e))
        return ValidationReport(passed=False, checks=checks)
    add("GGUF readable", True, f"GGUF v{g.version}, {len(g.tensors)} tensors, {len(g.metadata)} metadata keys")
    add("tensor data integrity", g.valid, "all tensor ranges in bounds and non-overlapping" if g.valid else "; ".join(g.errors[:5]))

    out_params = sum(t.n_elements for t in g.tensors)
    src_params = sum(t.n_elements for t in source.tensors)
    if operation == "quantize":
        src = {t.name: t.n_elements for t in source.tensors}
        out = {t.name: t.n_elements for t in g.tensors}
        same = src == out
        add("tensor set preserved", same, f"{len(out)} tensors, {out_params:,} elements" if same else f"source {len(src)} tensors / output {len(out)}; names or element counts differ")
        src_arch = source.fact_value("architecture")
        add("architecture preserved", g.get("general.architecture") == src_arch, f"{g.get('general.architecture')!r} (source {src_arch!r})")
    else:
        ratio = out_params / src_params if src_params else 0
        # convert_hf_to_gguf writes only the language model; vision/audio towers go to a separate mmproj file.
        primary = next((s.prefix for s in source.architecture.stacks if s.primary), None)
        text_params = sum(
            t.n_elements for t in source.tensors if (t.stack is None or t.stack == primary) and not _NON_TEXT.search(t.name)
        )
        text_ratio = out_params / text_params if text_params else 0
        if 0.95 <= ratio <= 1.05:
            add("parameter count", True, f"{out_params:,} output elements vs {src_params:,} in safetensors ({ratio:.3%})")
        else:
            add(
                "parameter count",
                0.95 <= text_ratio <= 1.05,
                f"{out_params:,} output elements vs {text_params:,} language-model elements in safetensors ({text_ratio:.3%}); "
                f"{src_params - text_params:,} vision/audio/auxiliary elements are not part of the GGUF",
            )
        add("architecture present", isinstance(g.get("general.architecture"), str), f"general.architecture = {g.get('general.architecture')!r}")

    ft = g.get("general.file_type")
    if expected_ftype is not None:
        actual = ft & ~LLAMA_FTYPE_GUESSED if isinstance(ft, int) else None
        add(
            "file type",
            actual == expected_ftype,
            f"general.file_type = {LLAMA_FTYPES.get(actual, actual) if actual is not None else 'missing'} (expected {LLAMA_FTYPES.get(expected_ftype, expected_ftype)})",
        )
    else:
        add("file type", isinstance(ft, int), f"general.file_type = {LLAMA_FTYPES.get(ft, ft) if isinstance(ft, int) else 'missing'}")
    return ValidationReport(passed=all(c.passed for c in checks), checks=checks)


# --------------------------------------------------------------------------- manager


class JobManager:
    def __init__(self, settings: Settings, history: History, tools_fn=None):
        self.settings = settings
        self.history = history
        self._tools_fn = tools_fn or (lambda refresh=False: detect_tools(settings, refresh))
        self._plans: dict[str, _StoredPlan] = {}
        self._jobs: dict[str, _JobState] = {}
        self._lock = threading.Lock()
        self._active: str | None = None

    def tools(self, refresh: bool = False) -> ToolsReport:
        return self._tools_fn(refresh)

    # planning -----------------------------------------------------------
    def _output_location(self, req: PlanRequest, scan: ScanResult, out_name: str, target: str, blockers: list[str]) -> Path | None:
        src = Path(scan.path)
        src_dir = src if src.is_dir() else src.parent
        if req.output_dir:
            out = Path(req.output_dir.strip().strip('"')).expanduser()
            if not out.is_absolute():
                blockers.append("output directory must be an absolute path")
                return None
            if not out.exists() and not out.parent.is_dir():
                blockers.append(f"parent of the output directory does not exist: {out.parent}")
                return None
        else:
            # Default: a fresh folder per model+target. Pick the first candidate that cannot collide with
            # the source or an earlier output, so the default never needs to overwrite anything.
            base = self.settings.outputs_dir / f"{_slug(scan.name)}-{_slug(target)}"
            out = base
            for i in range(2, 1000):
                clash = (out / out_name).exists() or (out / (out_name + ".partial")).exists()
                if not clash and out.resolve() != src_dir.resolve() and not _inside(out, src_dir):
                    break
                out = base.with_name(f"{base.name}-{i}")
        out = out.resolve()
        if out.exists() and not out.is_dir():
            blockers.append("output path exists and is not a directory")
        if out == src_dir.resolve() or _inside(out, src_dir):
            blockers.append("output directory must be separate from the source model's directory")
        if not (self.settings.is_allowed(out) or _inside(out, self.settings.outputs_dir)):
            blockers.append("output directory is outside the configured allowed roots")
        return out

    def create_plan(self, req: PlanRequest, scan: ScanResult) -> Plan:
        tools = self.tools()
        blockers: list[str] = []
        warnings: list[str] = []
        argv: list[str] = []
        tool_path = tool_version = None
        est: int | None = None
        est_note = "estimate (not a measurement)"
        expected_ftype: int | None = None
        supports_dry = False
        src = Path(scan.path)
        target = req.target.strip().upper() if req.operation == "quantize" else req.target.strip().lower()
        if not re.fullmatch(r"[A-Za-z0-9_]+", target):
            blockers.append("invalid target name")
            target = "invalid"
        out_name = f"{src.stem}.{target}.gguf" if req.operation == "quantize" else f"{_slug(src.name)}.{target.upper()}.gguf"

        if scan.is_remote:
            raise PermissionError(
                "this scan was read remotely from the Hugging Face Hub; download the model, scan the local copy, then quantize that"
            )
        if not scan.integrity.valid:
            blockers.append("source failed integrity checks; fix or re-download the model first")

        out_dir = self._output_location(req, scan, out_name, target.upper(), blockers)

        if req.operation == "quantize":
            q = tools.quantize
            tool_path, tool_version, supports_dry = q.path, q.version, q.supports_dry_run
            names = {t.name: t for t in q.types}
            if scan.kind != "gguf":
                blockers.append("llama-quantize operates on GGUF files; convert this model to GGUF first")
            if len(scan.files) > 1:
                blockers.append("split (multi-file) GGUF input is not supported by this workflow")
            if not q.available:
                blockers.append(f"llama-quantize unavailable: {q.detail}")
            elif target not in names:
                blockers.append(f"{target} is not offered by the detected llama-quantize build")
            if target in IMATRIX_REQUIRED:
                blockers.append(f"{target} requires an importance matrix (imatrix), which this workflow does not provide")
            if target == "COPY":
                blockers.append("COPY is not a quantization")
            src_label = (scan.quantization.label or "").split(" ")[0]
            if src_label == target:
                blockers.append(f"source is already {target}")
            if scan.quantization.is_quantized:
                if req.allow_requantize:
                    warnings.append(f"requantizing from {src_label}: quality will be lower than quantizing from an F16/BF16/F32 source")
                else:
                    blockers.append(f"source is already quantized ({src_label}); enable 'allow requantize' to proceed (lower quality than quantizing from F16/BF16)")
            if target in TERNARY_ONLY:
                warnings.append(f"{target} is intended for ternary (BitNet-style) models; ordinary models lose most of their quality")
            if target == "MXFP4_MOE" and not any(t.component == "experts" for t in scan.tensors):
                warnings.append("MXFP4_MOE only quantizes MoE expert tensors; this model has none")
            if target in names:
                expected_ftype = names[target].ftype_id
                try:
                    est = sum(b for _, b in estimate_tensor_bytes(scan, target).values()) + int(scan.format.details.get("data_offset", 0))
                    est_note = "estimate from per-tensor replay of llama.cpp type selection (not a measurement)"
                except KeyError:
                    est = None
                    est_note = "no size estimate available for this type"
            if out_dir is not None and q.path:
                argv = [q.path]
                if req.allow_requantize and scan.quantization.is_quantized:
                    argv.append("--allow-requantize")
                argv += [str(src), str(out_dir / (out_name + ".partial")), target]
                if req.threads:
                    argv.append(str(req.threads))
        else:
            c = tools.convert
            tool_path = c.script
            if scan.kind != "hf_safetensors":
                blockers.append("conversion requires a Hugging Face directory with config.json and safetensors weights")
            if not c.available:
                blockers.append(f"HF→GGUF converter unavailable: {c.detail}")
            elif target not in c.outtypes:
                blockers.append(f"outtype '{target}' not offered by the detected converter ({', '.join(c.outtypes)})")
            model_class = scan.fact_value("model_class")
            if c.supported_architectures and model_class and model_class not in c.supported_architectures:
                blockers.append(f"{model_class} is not in the converter's supported model list")
            elif c.available and not model_class:
                warnings.append("config.json has no 'architectures' entry; the converter may reject this model")
            if scan.quantization.packed_checkpoint:
                blockers.append(f"packed quantized checkpoints ({scan.quantization.label}) cannot be converted; use the original FP16/BF16 weights")
            elif scan.quantization.config:
                warnings.append(f"checkpoint has a quantization_config ({scan.quantization.label}); converter support varies by method")
            src_dir = src
            py_files = sorted(p.name for p in src_dir.glob("*.py"))[:5] if src_dir.is_dir() else []
            auto_map = any(m.key == "config.auto_map" for m in scan.metadata)
            if py_files or auto_map:
                blockers.append(
                    "model directory contains custom Python code (" + (", ".join(py_files) if py_files else "config.auto_map")
                    + "); conversion is blocked because the converter may execute it via trust_remote_code"
                )
            names = {Path(f.path).name for f in scan.files}
            if not names & {"tokenizer.json", "tokenizer.model", "vocab.json"}:
                warnings.append("no tokenizer files found; the converter usually needs tokenizer.json or tokenizer.model")
            if any(f.role == "unparsed weights" for f in scan.files):
                warnings.append("directory also contains pickle/other weight files; the converter uses the safetensors files when present")
            expected_ftype = CONVERT_OUTTYPE_FTYPE.get(target)
            ft_name = {"f32": "F32", "f16": "F16", "bf16": "BF16", "q8_0": "Q8_0", "tq1_0": "TQ1_0", "tq2_0": "TQ2_0"}.get(target)
            if target == "auto":
                dt = str(scan.fact_value("torch_dtype") or "").lower()
                ft_name = "BF16" if "bfloat16" in dt else "F16"
                est_note = f"estimate assuming 'auto' resolves to {ft_name} (not a measurement)"
            if ft_name and not scan.quantization.packed_checkpoint:
                # Tokenizer metadata dominates the GGUF header: ~48 bytes per token (string, score, type) plus merges.
                vocab = scan.fact_value("vocab_size") or 0
                header = 64 * 1024 + 48 * int(vocab)
                est = sum(b for _, b in estimate_tensor_bytes(scan, ft_name).values()) + header
                if target != "auto":
                    est_note = f"estimate from tensor shapes plus ~{header / 1024**2:.1f} MiB of tokenizer metadata (not a measurement)"
            warnings.append("conversion reads every weight and can take several minutes and several GB of RAM")
            if out_dir is not None and c.available and c.python and c.script:
                argv = [c.python, c.script, str(src), "--outfile", str(out_dir / (out_name + ".partial")), "--outtype", target]

        fingerprint: list[tuple[str, int, int]] = []
        try:
            fingerprint = _fingerprint(scan)
            recorded = {f.path: f.size_bytes for f in scan.files if f.parsed}
            if any(recorded.get(p) != size for p, size, _ in fingerprint):
                blockers.append("source files changed size since they were scanned; rescan the model")
        except OSError:
            blockers.append("source files are no longer readable; rescan the model")

        out_path = partial = None
        if argv and out_dir is not None:
            partial = next(Path(a) for a in argv if a.endswith(".gguf.partial"))
            out_path = partial.with_name(partial.name[: -len(".partial")])
            if out_path.exists():
                blockers.append(f"{out_path.name} already exists in the output directory; Model Surgery never overwrites files")
            if partial.exists():
                blockers.append(f"a leftover {partial.name} exists; remove it or choose another output directory")
            if out_path.resolve() == src.resolve() or partial.resolve() == src.resolve():
                blockers.append("output path resolves to the source model")

        free = None
        if out_dir is not None:
            probe = out_dir
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            try:
                free = shutil.disk_usage(probe).free
            except OSError:
                free = None
            if free is not None and est is not None and est * 1.05 + 256 * 1024**2 > free:
                blockers.append(f"insufficient free disk space: ~{est / 1024**3:.2f} GiB needed, {free / 1024**3:.2f} GiB free")

        can = not blockers and bool(argv)
        plan = Plan(
            plan_id=uuid.uuid4().hex,
            created_at=_now(),
            expires_at=time.time() + PLAN_TTL_SECONDS,
            operation=req.operation,
            scan_id=scan.id,
            source_path=scan.path,
            source_label=scan.quantization.label,
            target=target,
            output_dir=str(out_dir) if out_dir else None,
            output_path=str(out_path) if out_path else None,
            partial_path=str(partial) if partial else None,
            argv=argv,
            tool=tool_path,
            tool_version=tool_version,
            estimated_output_bytes=est,
            estimate_note=est_note,
            free_disk_bytes=free,
            warnings=warnings,
            blockers=blockers,
            can_execute=can,
            confirm_token=secrets.token_urlsafe(16) if can else None,
            supports_tool_dry_run=bool(supports_dry and req.operation == "quantize" and can),
        )
        with self._lock:
            now = time.time()
            self._plans = {k: v for k, v in self._plans.items() if v.plan.expires_at > now}
            if can:
                self._plans[plan.plan_id] = _StoredPlan(plan, scan, fingerprint, expected_ftype)
        return plan

    def _get_plan(self, plan_id: str) -> _StoredPlan:
        with self._lock:
            item = self._plans.get(plan_id)
        if item is None or item.plan.expires_at < time.time():
            raise LookupError("plan not found or expired; create a new plan")
        return item

    def tool_dry_run(self, plan_id: str) -> dict:
        plan = self._get_plan(plan_id).plan
        if not plan.supports_tool_dry_run or not plan.tool:
            raise ValueError("the detected tool does not support --dry-run for this plan")
        argv = [plan.tool, "--dry-run"]
        if "--allow-requantize" in plan.argv:
            argv.append("--allow-requantize")
        argv += [plan.source_path, plan.target]
        code, text = run_fixed(argv, timeout=600)
        result = parse_dry_run(text)
        result.update(exit_code=code, argv=argv, tail=text.strip().splitlines()[-8:])
        return result

    # execution ----------------------------------------------------------
    def execute(self, req: ExecuteRequest) -> Job:
        if not req.confirm:
            raise PermissionError("explicit confirmation is required")
        stored = self._get_plan(req.plan_id)
        plan, scan, expected = stored.plan, stored.scan, stored.expected_ftype
        if not plan.confirm_token or not secrets.compare_digest(req.confirm_token, plan.confirm_token):
            raise PermissionError("confirmation token does not match this plan")
        with self._lock:
            if self._active and self._jobs[self._active].job.status in ("running", "validating"):
                raise RuntimeError("another job is already running; wait for it to finish or cancel it")
            self._plans.pop(plan.plan_id, None)  # single use
            if stored.fingerprint != _current_fingerprint(scan):
                raise RuntimeError("source files changed since the plan was created; rescan and plan again")
            assert plan.output_path and plan.partial_path
            if Path(plan.output_path).exists() or Path(plan.partial_path).exists():
                raise RuntimeError("output file appeared since planning; refusing to overwrite")
            job = Job(id=uuid.uuid4().hex, created_at=_now(), status="running", plan=plan)
            st = _JobState(job)
            self._jobs[job.id] = st
            self._active = job.id
        self._persist(st)
        threading.Thread(target=self._run, args=(st, scan, expected), name=f"job-{job.id[:8]}", daemon=True).start()
        return self.get(job.id)

    def _persist(self, st: _JobState) -> None:
        j = st.job.model_copy()
        j.log_tail = list(st.log)[-200:]
        self.history.save_job(j.id, j.created_at, j.status, j.model_dump(mode="json"))

    def _on_line(self, st: _JobState, line: str) -> None:
        st.log.append(line)
        m = _PROGRESS_QUANT.match(line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            if n:
                st.job.progress = min(1.0, i / n)
                st.job.progress_label = f"tensor {i}/{n}: {m.group(3)}"
            return
        m2 = _PROGRESS_PCT.search(line)
        if m2:
            st.job.progress = min(1.0, int(m2.group(1)) / 100)
            st.job.progress_label = line.strip()[:120]

    def _run(self, st: _JobState, scan: ScanResult, expected_ftype: int | None) -> None:
        job, plan = st.job, st.job.plan
        partial, final = Path(plan.partial_path or ""), Path(plan.output_path or "")
        job.started_at = _now()
        try:
            partial.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                plan.argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                cwd=str(partial.parent),
                creationflags=NO_WINDOW,
                env=env,
            )
        except OSError as e:
            self._finish(st, "failed", error=f"could not start tool: {e}")
            return
        st.proc = proc
        try:
            self._supervise(st, proc, scan, expected_ftype, partial, final)
        except Exception as e:  # noqa: BLE001 - a crashed worker must never leave the job slot blocked
            if proc.poll() is None:
                proc.kill()
            _remove(partial)
            self._finish(st, "failed", error=f"internal error while running the job: {type(e).__name__}: {e}; partial output removed")

    def _supervise(self, st: _JobState, proc: subprocess.Popen, scan: ScanResult, expected_ftype: int | None, partial: Path, final: Path) -> None:
        job, plan = st.job, st.job.plan
        buf = b""
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                if p.strip():
                    self._on_line(st, p.decode("utf-8", errors="replace"))
            if len(buf) > 1 << 20:
                buf = buf[-4096:]
        if buf.strip():
            self._on_line(st, buf.decode("utf-8", errors="replace"))
        code = proc.wait()
        job.exit_code = code

        if st.cancel_requested:
            _remove(partial)
            self._finish(st, "cancelled", error="cancelled by user; partial output removed")
            return
        if code != 0:
            tail = [l for l in list(st.log)[-40:] if l.strip()]
            errs = [l for l in tail if re.search(r"error|failed|exception|traceback", l, re.I)] or tail[-5:]
            _remove(partial)
            self._finish(st, "failed", error=f"tool exited with code {code}: " + " | ".join(errs[-5:]))
            return

        job.status = "validating"
        job.progress_label = "validating output"
        self._persist(st)
        report = validate_output(plan.operation, partial, scan, expected_ftype, code)
        job.validation = report
        if not report.passed:
            _remove(partial)
            failed = [c.name for c in report.checks if not c.passed]
            self._finish(st, "failed", error="output failed validation: " + ", ".join(failed) + "; partial output removed")
            return
        if final.exists():
            _remove(partial)
            self._finish(st, "failed", error=f"{final.name} appeared during the job; refusing to overwrite it")
            return
        os.rename(partial, final)
        job.output_path = str(final)
        job.output_size_bytes = final.stat().st_size
        try:
            out_scan = scan_path(str(final), self.settings)
            self.history.save_scan(out_scan)
            job.output_scan_id = out_scan.id
        except Exception as e:  # noqa: BLE001 - validation already passed; record but do not fail
            st.log.append(f"[model-surgery] post-validation scan failed: {e}")
        job.progress = 1.0
        self._finish(st, "succeeded")

    def _finish(self, st: _JobState, status: JobStatus, error: str | None = None) -> None:
        st.job.status = status
        st.job.error = error
        st.job.finished_at = _now()
        if status == "succeeded":
            st.job.progress_label = "validated"
        self._persist(st)

    def cancel(self, job_id: str) -> Job:
        st = self._jobs.get(job_id)
        if st is None:
            raise LookupError("job not found")
        if st.job.status != "running":
            raise RuntimeError(f"job is {st.job.status}; nothing to cancel")
        st.cancel_requested = True
        proc = st.proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        return self.get(job_id)

    def get(self, job_id: str) -> Job:
        st = self._jobs.get(job_id)
        if st is not None:
            j = st.job.model_copy()
            j.log_tail = list(st.log)[-300:]
            return j
        data = self.history.get_job(job_id)
        if data is None:
            raise LookupError("job not found")
        return Job.model_validate(data)

    def list(self) -> list[Job]:
        with self._lock:
            ids = list(self._jobs)
        live = {jid: self.get(jid) for jid in ids}
        stored = [Job.model_validate(d) for d in self.history.list_jobs()]
        out = [live.get(j.id, j) for j in stored]
        for jid, j in live.items():
            if all(o.id != jid for o in out):
                out.insert(0, j)
        for j in out:
            j.log_tail = j.log_tail[-5:]
        return out


def _current_fingerprint(scan: ScanResult) -> list[tuple[str, int, int]]:
    try:
        return _fingerprint(scan)
    except OSError:
        return []


def _remove(p: Path) -> None:
    try:
        if p.name.endswith(".partial") and p.is_file():
            p.unlink()
    except OSError:
        pass

"""Detection of llama.cpp utilities.

Executables are located from server-side configuration or PATH only, their
basenames are checked against an allowlist, and they are always invoked with
an argument list (never through a shell).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Settings

IS_WINDOWS = sys.platform == "win32"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""
QUANTIZE_NAME = "llama-quantize"
SERVER_NAME = "llama-server"
VERSION_NAMES = ("llama-cli", "llama-server")
CONVERT_SCRIPT_NAME = "convert_hf_to_gguf.py"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_TYPE_LINE = re.compile(r"^\s*(\d+)\s+or\s+([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$")
_VERSION_LINE = re.compile(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)")
_OUTTYPE = re.compile(r"--outtype\s+\{([^}]+)\}")
_ARCH_LINE = re.compile(r"^\s*-?\s*([A-Z][A-Za-z0-9_]+)\s*$")


class QuantType(BaseModel):
    name: str
    ftype_id: int
    description: str


class QuantizeTool(BaseModel):
    available: bool
    path: str | None = None
    version: str | None = None
    types: list[QuantType] = Field(default_factory=list)
    supports_dry_run: bool = False
    detail: str | None = None


class ConvertTool(BaseModel):
    available: bool
    script: str | None = None
    python: str | None = None
    outtypes: list[str] = Field(default_factory=list)
    supported_architectures: list[str] | None = None
    detail: str | None = None


class ServerTool(BaseModel):
    available: bool
    path: str | None = None
    version: str | None = None
    detail: str | None = None


class ToolsReport(BaseModel):
    quantize: QuantizeTool
    convert: ConvertTool
    server: ServerTool = Field(default_factory=lambda: ServerTool(available=False, detail="not checked"))
    checked_at: float
    configuration_help: list[str]


def run_fixed(argv: list[str], timeout: float, cwd: str | None = None) -> tuple[int | None, str]:
    """Run a fixed argument vector and capture combined output. Never uses a shell."""
    try:
        cp = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            shell=False,
            cwd=cwd,
            creationflags=NO_WINDOW,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:.0f}s"
    except OSError as e:
        return None, f"could not start: {e}"
    return cp.returncode, cp.stdout.decode("utf-8", errors="replace")[-400_000:]


def _exe(directory: Path | None, base: str) -> Path | None:
    name = base + EXE_SUFFIX
    if directory is not None:
        p = directory / name
        if p.is_file():
            return p
    found = shutil.which(name)
    return Path(found) if found else None


def parse_quant_types(help_text: str) -> list[QuantType]:
    out: list[QuantType] = []
    for line in help_text.splitlines():
        m = _TYPE_LINE.match(line)
        if not m:
            continue
        desc = m.group(3)
        if desc.lower().startswith("alias for"):
            continue
        out.append(QuantType(name=m.group(2).upper(), ftype_id=int(m.group(1)), description=desc))
    return out


def detect_quantize(settings: Settings) -> QuantizeTool:
    exe = _exe(settings.llama_cpp_bin_dir, QUANTIZE_NAME)
    if exe is None:
        where = f"in {settings.llama_cpp_bin_dir} or on PATH" if settings.llama_cpp_bin_dir else "on PATH"
        return QuantizeTool(available=False, detail=f"{QUANTIZE_NAME}{EXE_SUFFIX} not found {where}")
    if exe.name.lower() != (QUANTIZE_NAME + EXE_SUFFIX).lower():
        return QuantizeTool(available=False, detail=f"unexpected executable name {exe.name}")
    code, text = run_fixed([str(exe), "--help"], timeout=20)
    types = parse_quant_types(text)
    if not types:
        return QuantizeTool(available=False, path=str(exe), detail=f"could not read supported types from --help (exit {code}): {text[-300:]}")
    version = None
    for vname in VERSION_NAMES:
        vexe = exe.with_name(vname + EXE_SUFFIX)
        if vexe.is_file():
            _, vt = run_fixed([str(vexe), "--version"], timeout=20)
            m = _VERSION_LINE.search(vt)
            if m:
                version = f"b{m.group(1)} ({m.group(2)})"
                break
    return QuantizeTool(available=True, path=str(exe), version=version, types=types, supports_dry_run="--dry-run" in text)


def detect_convert(settings: Settings) -> ConvertTool:
    script, py = settings.convert_script, settings.convert_python
    if script is None:
        return ConvertTool(available=False, detail=f"no {CONVERT_SCRIPT_NAME} configured (set MODEL_SURGERY_CONVERT_SCRIPT to a llama.cpp checkout)")
    if script.name != CONVERT_SCRIPT_NAME or not script.is_file():
        return ConvertTool(available=False, script=str(script), detail=f"configured convert script must be an existing {CONVERT_SCRIPT_NAME}")
    if not py.is_file() or not py.name.lower().startswith("python"):
        return ConvertTool(available=False, script=str(script), python=str(py), detail="configured Python interpreter not found")
    code, text = run_fixed([str(py), str(script), "--help"], timeout=180, cwd=str(script.parent))
    if code != 0:
        tail = "\n".join(text.strip().splitlines()[-6:])
        return ConvertTool(
            available=False, script=str(script), python=str(py),
            detail=f"prerequisites missing: '{CONVERT_SCRIPT_NAME} --help' failed (exit {code}). Install llama.cpp/requirements/requirements-convert_hf_to_gguf.txt.\n{tail}",
        )
    m = _OUTTYPE.search(text)
    outtypes = [t.strip() for t in m.group(1).split(",")] if m else ["f32", "f16", "bf16", "q8_0"]
    archs: list[str] | None = None
    if "--print-supported-models" in text:
        c2, t2 = run_fixed([str(py), str(script), "--print-supported-models"], timeout=180, cwd=str(script.parent))
        if c2 == 0:
            archs = sorted({m.group(1) for line in t2.splitlines() if (m := _ARCH_LINE.match(line.split(":")[-1]))})
    return ConvertTool(available=True, script=str(script), python=str(py), outtypes=outtypes, supported_architectures=archs or None)


def _detect_server_uncached(settings: Settings) -> ServerTool:
    exe = _exe(settings.llama_cpp_bin_dir, SERVER_NAME)
    if exe is None:
        where = f"in {settings.llama_cpp_bin_dir} or on PATH" if settings.llama_cpp_bin_dir else "on PATH"
        return ServerTool(available=False, detail=f"{SERVER_NAME}{EXE_SUFFIX} not found {where}")
    if exe.name.lower() != (SERVER_NAME + EXE_SUFFIX).lower():
        return ServerTool(available=False, detail=f"unexpected executable name {exe.name}")
    code, text = run_fixed([str(exe), "--version"], timeout=20)
    m = _VERSION_LINE.search(text)
    if code != 0 or not m:
        return ServerTool(available=False, path=str(exe), detail=f"'{SERVER_NAME} --version' failed (exit {code}): {text[-300:]}")
    return ServerTool(available=True, path=str(exe), version=f"b{m.group(1)} ({m.group(2)})")


_server_cache: tuple[float, ServerTool] | None = None
_server_lock = threading.Lock()


def detect_server(settings: Settings, refresh: bool = False, ttl: float = 300.0) -> ServerTool:
    global _server_cache
    with _server_lock:
        if _server_cache and not refresh and time.time() - _server_cache[0] < ttl:
            return _server_cache[1]
        tool = _detect_server_uncached(settings)
        _server_cache = (time.time(), tool)
        return tool


_cache: tuple[float, ToolsReport] | None = None
_lock = threading.Lock()


def detect_tools(settings: Settings, refresh: bool = False, ttl: float = 300.0) -> ToolsReport:
    global _cache
    with _lock:
        if _cache and not refresh and time.time() - _cache[0] < ttl:
            return _cache[1]
        report = ToolsReport(
            quantize=detect_quantize(settings),
            convert=detect_convert(settings),
            server=detect_server(settings, refresh),
            checked_at=time.time(),
            configuration_help=[
                "Tool locations are configured on the server only (never from the browser):",
                "  MODEL_SURGERY_LLAMA_CPP_DIR   = folder containing llama-quantize and llama-server",
                "  MODEL_SURGERY_CONVERT_SCRIPT  = path to llama.cpp/convert_hf_to_gguf.py",
                "  MODEL_SURGERY_CONVERT_PYTHON  = Python with the convert requirements installed",
                f"or the same keys (llama_cpp_bin_dir, convert_script, convert_python) in {settings.data_dir / 'config.json'}",
            ],
        )
        _cache = (time.time(), report)
        return report


def reset_cache() -> None:
    global _cache, _server_cache
    with _lock:
        _cache = None
    with _server_lock:
        _server_cache = None

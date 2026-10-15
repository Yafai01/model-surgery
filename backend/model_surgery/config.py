"""Server-side settings.

Tool locations are configured only here (environment, CLI flags or the config
file in the data directory) and never through the HTTP API, so the browser can
never choose which executable the server runs.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    return Path(os.environ.get("MODEL_SURGERY_DATA_DIR", Path.home() / ".model_surgery")).expanduser()


@dataclass
class Settings:
    data_dir: Path = field(default_factory=default_data_dir)
    llama_cpp_bin_dir: Path | None = None
    convert_script: Path | None = None  # llama.cpp/convert_hf_to_gguf.py
    convert_python: Path = field(default_factory=lambda: Path(sys.executable))
    allowed_roots: list[Path] = field(default_factory=list)  # empty = any local path
    output_root: Path | None = None
    frontend_dist: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    extra_allowed_hosts: list[str] = field(default_factory=list)

    @property
    def outputs_dir(self) -> Path:
        return self.output_root or (self.data_dir / "outputs")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "history.sqlite3"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def is_allowed(self, path: Path) -> bool:
        if not self.allowed_roots:
            return True
        rp = path.resolve()
        for root in self.allowed_roots:
            try:
                rp.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False


def _opt_path(v: object) -> Path | None:
    return Path(str(v)).expanduser() if v else None


def load_settings(**overrides: object) -> Settings:
    """Precedence: explicit overrides > environment > data_dir/config.json > defaults."""
    s = Settings()
    if overrides.get("data_dir"):
        s.data_dir = Path(str(overrides["data_dir"])).expanduser()

    cfg_file = s.data_dir / "config.json"
    file_cfg: dict = {}
    if cfg_file.is_file() and cfg_file.stat().st_size < 1024 * 1024:
        try:
            loaded = json.loads(cfg_file.read_text(encoding="utf-8"))
            file_cfg = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            file_cfg = {}

    def pick(key: str, env: str) -> object:
        if overrides.get(key) is not None:
            return overrides[key]
        if os.environ.get(env):
            return os.environ[env]
        return file_cfg.get(key)

    s.llama_cpp_bin_dir = _opt_path(pick("llama_cpp_bin_dir", "MODEL_SURGERY_LLAMA_CPP_DIR"))
    s.convert_script = _opt_path(pick("convert_script", "MODEL_SURGERY_CONVERT_SCRIPT"))
    s.convert_python = _opt_path(pick("convert_python", "MODEL_SURGERY_CONVERT_PYTHON")) or Path(sys.executable)
    s.output_root = _opt_path(pick("output_root", "MODEL_SURGERY_OUTPUT_DIR"))
    s.frontend_dist = _opt_path(pick("frontend_dist", "MODEL_SURGERY_FRONTEND_DIST"))
    roots = pick("allowed_roots", "MODEL_SURGERY_ALLOWED_ROOTS")
    if isinstance(roots, str):
        roots = [r for r in roots.split(os.pathsep) if r]
    if isinstance(roots, list):
        s.allowed_roots = [Path(str(r)).expanduser() for r in roots]
    for k in ("host", "port"):
        if overrides.get(k) is not None:
            setattr(s, k, overrides[k])
    if s.frontend_dist is None:
        guess = Path(__file__).resolve().parents[2] / "frontend" / "dist"
        if guess.is_dir():
            s.frontend_dist = guess
    return s

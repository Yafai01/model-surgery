# Model Surgery

**A laptop-first workbench for inspecting and optimizing local LLM / SLM model files.**

Model Surgery opens GGUF files and Hugging Face safetensors directories, rebuilds their architecture from real header metadata, visualizes layers and tensors in 2D and 3D, estimates memory for laptops and the Jetson Orin Nano 8GB, and drives llama.cpp quantization and conversion with a dry-run plan, explicit confirmation and output validation.

It reads headers only. Inspecting a model needs no PyTorch or CUDA, and tensor data is never loaded.

**▶ Try it in your browser: [model-surgery.vercel.app](https://model-surgery.vercel.app).** The hosted *browser edition* parses the model in your tab with the File API, reading only the header bytes, so a multi-GB GGUF scans in well under a second. **Nothing is uploaded.** The site's Content-Security-Policy blocks connections to any other host, and scan history stays in your browser's IndexedDB. It covers inspection, the 2D/3D architecture view, the Memory Lab, exports and comparison. Quantization needs llama.cpp on your own machine, so use the local app below for that.

| | Browser edition (Vercel) | Local app |
|---|---|---|
| Inspect GGUF / safetensors / HF folders | ✅ | ✅ |
| Architecture 2D/3D, tensors, metadata | ✅ | ✅ |
| Memory Lab, device profiles | ✅ | ✅ |
| JSON / CSV / Markdown reports, compare | ✅ | ✅ |
| History | this browser (IndexedDB) | SQLite |
| Quantize / convert with llama.cpp | — | ✅ |

| | |
|---|---|
| Backend | Python 3.11+, FastAPI, SQLite (stdlib), a pure-Python GGUF and safetensors header parser |
| Frontend | React 18 + TypeScript + Vite, Three.js (lazy-loaded) |
| External tools | llama.cpp `llama-quantize` and `convert_hf_to_gguf.py`, both optional |
| Targets | Windows 11 first. Linux/aarch64 (Jetson Orin Nano) uses the same code |

---

## Features

### 1. Model import
- Pick a **GGUF** file (v2/v3, little- or big-endian, split `-00001-of-0000N` shards) or a **Hugging Face directory** (`config.json` + single or sharded safetensors with `model.safetensors.index.json`), using a path field or the built-in read-only file browser.
- Reads GGUF metadata, architecture, the `general.file_type` quantization label and every tensor descriptor.
- Parses safetensors headers (8-byte length + JSON) without touching tensor bytes.
- **Says plainly what it won't open.** `.bin` / `.pt` / `.ckpt` are Python pickles and are refused, because loading them can run arbitrary code. ONNX, TFLite, HDF5 and similar formats are reported as unsupported, and a `.gguf` file without the GGUF magic bytes is rejected.

### 2. Architecture explorer
- Groups tensors into layer stacks (`blk.N.*`, `model.layers.N.*`, `transformer.h.N.*`, …), classifies them as attention / FFN / MoE experts / router / norm / SSM / embedding / output, and shows a layer-anatomy diagram.
- Detects multimodal models (separate vision/language stacks) and non-uniform or hybrid layer patterns.
- Searchable, sortable, filterable tensor table with name, shape, dtype, element count, byte size and file offset. Filtered CSV export.
- **3D view** (Three.js, orbit, hover, click to inspect) with a **2D fallback** that is the default and is selected automatically when WebGL is unavailable.
- Every value carries a **provenance** label:

  | label | meaning | examples |
  |---|---|---|
  | `verified` | read directly from the file | `qwen2.block_count`, tensor shapes/dtypes, `config.json:num_key_value_heads` |
  | `derived` | computed deterministically from verified values | parameter count, head_dim = hidden / heads, GQA ratio |
  | `inferred` | heuristic | layer grouping & component roles from tensor names, KV heads from `k_proj` width when metadata is missing |

  Mismatches between metadata and tensors (e.g. `block_count` ≠ the number of layers actually present) are reported, not hidden.

### 3. Memory laboratory
- **Weight storage** for FP32, FP16/BF16, INT8 (Q8_0) and 4-bit (Q4_K_M and Q4_0). The estimator **replays llama.cpp's per-tensor type selection** over the model's real tensor list: block-scale overhead, 1-D tensors kept in F32, fallback types when a row length isn't a multiple of the block size, and higher-precision `output`/`attn_v`/`ffn_down` tensors in K-quant mixes.
  - Validated against a real `qwen2.5-0.5b-instruct-q4_k_m.gguf`: the Q4_K_M replay predicts the type of **all 291 tensors** and the total size **byte-for-byte**, and the Q4_0 plan estimate matched the file llama-quantize actually produced exactly (428,730,208 bytes).
- **Inference memory** is kept separate: weights + KV cache (`layers × tokens × kv_heads × (k_dim + v_dim) × bytes`, with F16/BF16/Q8_0/Q4_0/F32 cache types) + compute buffers (logits, activations, and the attention-score matrix when flash attention is off) + runtime overhead. Each term shows its formula and provenance.
- **Device profiles:** laptop CPU (16 GB), laptop with an 8 GB GPU (with partial-offload `-ngl` estimate), **Jetson Orin Nano 8GB** (desktop and headless), and a custom profile. Each profile lists its assumptions, and the lab shows a fit verdict, headroom and a memory-bandwidth decode ceiling (labelled as a theoretical upper bound).
- Every screen and export labels these figures as **estimates, never measurements**.

### 4. Quantization workflow
1. **Detects** `llama-quantize` (parses the quantization types *your build* supports from `--help`, the build number from `llama-cli --version`, and `--dry-run` support). It also detects `convert_hf_to_gguf.py` by running its `--help` with the configured Python, which proves its prerequisites import.
2. **Offers only what the tool and input support.** GGUF inputs get `llama-quantize` types. Types that need an importance matrix are disabled, and requantizing an already-quantized model needs an explicit opt-in. HF inputs get the converter's `--outtype` choices and are checked against its supported-model list.
3. **Dry-run plan.** Shows the exact argv, the output path, the size estimate and free disk space, with warnings and blockers. For quantization it can also ask `llama-quantize --dry-run` for the tool's own size figure.
4. **Explicit confirmation.** A single-use, expiring token is returned with the plan, and the job starts only after the user ticks the confirmation and presses Start.
5. **Safe execution.** The tool runs with a fixed argv and no shell. Output goes to a separate directory (never the source's directory), is written as `*.partial`, and existing files are never overwritten. Source files are fingerprinted and must not change between planning and execution. Only one heavy job runs at a time, and jobs can be cancelled.
6. **Validation before success.** Checks the exit status, re-parses the output GGUF, checks tensor-range integrity, confirms the tensor set and element counts are preserved (quantize) or the parameter count is within ±5% (convert), confirms the architecture is preserved, and checks that `general.file_type` matches the requested type. Only then is `*.partial` renamed into place and the output scanned into history. Failed or cancelled outputs are deleted.

### 5. Reports
- Exports: JSON (full scan), CSV (tensor table, with spreadsheet formula injection neutralised), and a GitHub-ready **Markdown report** with provenance and estimates labelled.
- **Compare** two scans: architecture, size, quantization, weight/inference estimates, device fit, tensor-level dtype changes (for example `Q5_0 → Q4_0 × 133`). Exportable as Markdown.
- Scan and job **history** stored locally in SQLite.

---

## Quick start (Windows 11)

Prerequisites: Python ≥ 3.11 and Node.js ≥ 18.

```powershell
git clone <this repo> model-surgery; cd model-surgery
.\scripts\setup.ps1                     # venv + backend + frontend build
.\.venv\Scripts\model-surgery serve --open
```

This opens <http://localhost:8765>. You can also run `scripts\setup.ps1` steps by hand:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e "backend[dev]"
cd frontend; npm ci; npm run build; cd ..
```

### Enable quantization (optional)

Download a llama.cpp release (for example `llama-bXXXX-bin-win-cpu-x64.zip`), unzip it, and point Model Surgery at the folder containing `llama-quantize.exe`:

```powershell
.\.venv\Scripts\model-surgery --llama-cpp-dir C:\tools\llamacpp\bin serve
```

For HF → GGUF conversion you also need a llama.cpp **source checkout** and its Python requirements:

```powershell
git clone https://github.com/ggml-org/llama.cpp C:\src\llama.cpp
py -3.11 -m venv C:\src\llama.cpp\.venv
C:\src\llama.cpp\.venv\Scripts\pip install -r C:\src\llama.cpp\requirements\requirements-convert_hf_to_gguf.txt
.\.venv\Scripts\model-surgery --llama-cpp-dir C:\tools\llamacpp\bin `
    --convert-script C:\src\llama.cpp\convert_hf_to_gguf.py `
    --convert-python C:\src\llama.cpp\.venv\Scripts\python.exe serve
```

These settings can also be given as environment variables (`MODEL_SURGERY_LLAMA_CPP_DIR`, `MODEL_SURGERY_CONVERT_SCRIPT`, `MODEL_SURGERY_CONVERT_PYTHON`, `MODEL_SURGERY_OUTPUT_DIR`, `MODEL_SURGERY_ALLOWED_ROOTS`) or as keys in `~/.model_surgery/config.json`:

```json
{
  "llama_cpp_bin_dir": "C:/tools/llamacpp/bin",
  "convert_script": "C:/src/llama.cpp/convert_hf_to_gguf.py",
  "convert_python": "C:/src/llama.cpp/.venv/Scripts/python.exe",
  "allowed_roots": ["D:/models"]
}
```

Tool paths are **server-side configuration only**. No API endpoint accepts an executable path, command or tool argument.

### Command line

```text
model-surgery inspect <path> [--format summary|json|markdown]   # scan without the UI
model-surgery tools                                              # show detected tooling
model-surgery serve [--host 127.0.0.1] [--port 8765] [--open]
```

---

## Jetson Orin Nano 8GB

The backend is pure Python with three dependencies (FastAPI, Uvicorn, Pydantic), so it runs unchanged on aarch64.

```bash
# JetPack 6 ships Python 3.10; install 3.11+ (e.g. via uv or deadsnakes)
uv venv --python 3.11 .venv && source .venv/bin/activate
pip install -e backend
# build the frontend on any machine and copy frontend/dist, or build on-device with Node 18+
cmake -B build -DGGML_CUDA=ON && cmake --build build -j4 --target llama-quantize   # in llama.cpp
model-surgery --llama-cpp-dir ~/llama.cpp/build/bin serve
```

The memory lab includes Jetson Orin Nano 8GB profiles: 7.4 GiB visible unified memory, with ~1.8 GiB reserved for the GNOME desktop or ~0.8 GiB headless, a ~350 MB CUDA context, and 68 GB/s bandwidth (102 GB/s in JetPack 6.2 "Super" MAXN mode). These are planning assumptions. Measure with `tegrastats` on your unit.

To reach the UI from another machine, prefer an SSH tunnel (`ssh -L 8765:localhost:8765 jetson`) over `--host 0.0.0.0 --allow-remote`. The app has **no authentication**.

---

## Security model

| Threat | Mitigation |
|---|---|
| Malicious model file | Never executed or deserialised. Pickle formats refused. GGUF/safetensors headers parsed with hard limits on counts, string/array lengths, dims and alignment. Every offset/size is bounds-checked against the real file size, and overlaps are detected. Duplicate keys are rejected. Metadata is rendered as plain text. |
| Path traversal via shard index | Shard names from `model.safetensors.index.json` must be plain sibling filenames that resolve inside the model directory. |
| Converter running model code | Conversion is **blocked** if the HF directory contains `*.py` files or `auto_map`, because `convert_hf_to_gguf.py` may use `trust_remote_code`. |
| Shell injection | No shell anywhere. Executables come from server config, their basenames are allowlisted, and argv is built from validated values. |
| Accidental data loss | Output dir must differ from the source dir, nothing is overwritten, work happens in `*.partial`, and the source is fingerprinted before execution. |
| Other websites calling the local API | Host-header allowlist (DNS rebinding), a required `X-Model-Surgery` header on state-changing requests (forces a CORS preflight that is never granted), no CORS headers, and a strict CSP plus `X-Frame-Options: DENY` on the UI. Binds to 127.0.0.1 unless `--allow-remote`. |
| Untrusted values in exports | CSV cells beginning with `= + - @` are prefixed. Markdown escapes `|` and `<`/`>`. |

---

## Development

```powershell
cd backend;  ..\.venv\Scripts\python -m pytest            # 147 tests (1 opt-in real-model test skipped by default)
cd frontend; npm run dev                                  # Vite on :5173, proxies /api to :8765
             npm run build                                # tsc type-check + production build
```

To run the optional tests against a real model, set `MODEL_SURGERY_TEST_GGUF=C:\path\to\model.gguf`.

### Browser edition

`frontend/src/engine/` is a TypeScript port of the Python engine (GGUF/safetensors/HF readers, architecture reconstruction, memory estimator, reports). `npm run build:browser` (Vite mode `browser`) swaps the HTTP API client for this in-page engine and writes `frontend/dist-browser/`, which is what `vercel.json` deploys. `npm run dev:browser` serves it locally.

The port was checked against the Python engine on real GGUFs (including llama.cpp-produced Q4_0 and Q4_K_M files), on HF folders (sharded, GPTQ, multimodal), and on deliberately corrupt files. Scan results, integrity errors and all memory-estimate scenarios match field for field. The one known difference is that JavaScript cannot distinguish `10000.0` from `10000` in `config.json`.

The test suite includes synthetic and deliberately corrupted GGUF and safetensors files (truncated, oversized counts, misaligned or overlapping offsets, unknown types, duplicate keys, path traversal), along with a fake `llama-quantize` that exercises success, tool failure, unreadable output, wrong output type, cancellation, concurrency limits, confirmation tokens and source-change detection.

### Layout

```text
backend/model_surgery/
  formats/gguf.py, safetensors_reader.py, hf.py, ggml_types.py   # untrusted-input parsers
  scanner.py        # path -> ScanResult (facts with provenance)
  architecture.py   # tensor-name -> layer/component reconstruction
  memory.py, devices.py
  tooling.py, jobs.py   # llama.cpp detection, plan/confirm/run/validate
  reports.py, history.py, api.py, cli.py
frontend/src/components/   # Import, Overview, Architecture (2D/3D), Tensors, Metadata, MemoryLab, Quantize, Reports, History
```

## Known limitations

- Memory figures are estimates. The compute-buffer term is a heuristic, and sliding-window attention, MLA KV compression and recurrent-state memory are reported as caveats rather than modelled.
- Per-tensor type replay covers llama.cpp's common presets (Q8_0, Q4_0/1, Q5_0/1, K-quants and their _S/_M/_L mixes). IQ-type mixes are approximated by their base type.
- Split (multi-file) GGUF inputs can be inspected but not quantized through the workflow. imatrix-based quantization is not offered.
- Packed GPTQ/AWQ/bitsandbytes checkpoints are inspected, but conversion is blocked and their parameter count is estimated from config.
- Architecture roles come from naming conventions. Unfamiliar architectures show tensors as `other` rather than guessing.

## License

MIT

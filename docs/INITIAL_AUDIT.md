# Initial audit — Model Surgery

Date: 2026-10-16 · Branch audited: `main` @ `c5964d2` · Machine: Windows 11, Python 3.11.0, Node 24.14.1, 32 GB RAM, NVIDIA RTX 2050 (4 GB), llama.cpp b8562 (CUDA build) at `C:\Users\YAFAI\tools\llamacpp\bin`.

This document records what the repository looked like **before** any change, what was verified, and how each finding was confirmed. "Reproduced" means a script or test demonstrated the failure; "code" means it is unambiguous from reading the source.

## 1. What the project is

| Area | Finding |
|---|---|
| Frontend | React 18.3 + TypeScript 5.6 + Vite 5.4, Three.js 0.169 (lazy-loaded). No router: a tab state in `App.tsx`. State is local `useState` per page. |
| Two editions | `vite --mode browser` swaps the HTTP client (`api.ts`) for an in-page TypeScript port of the Python engine (`src/engine/*`). The hosted Vercel site is this browser edition. |
| Backend | Python 3.11 FastAPI app (`backend/model_surgery`), SQLite history, pure-Python GGUF/safetensors header parsers, llama.cpp quantize/convert job runner. |
| Model loading | Header-only. GGUF (v2/v3, LE/BE, split), safetensors (single, sharded with index), HF folders, Hugging Face Hub links via HTTP range requests. Pickle formats refused. |
| Deployment | `vercel.json`: static build of `frontend/dist-browser` with a strict CSP (`connect-src 'self' huggingface.co`). No serverless functions. |
| Provenance | Every hyperparameter is a `Fact` labelled `verified` / `derived` / `inferred`; memory figures are labelled estimates (`is_measurement: false`). |

## 2. Baseline (before changes)

| Command | Result |
|---|---|
| `cd backend && ..\.venv\Scripts\python -m pytest` | **149 passed, 1 skipped** (4.8 s). The skip is the opt-in real-model test. |
| `cd frontend && npm run typecheck` | pass |
| `npm run build` | pass (Architecture3D chunk 494 kB) |
| `npm run build:browser` | pass |
| `model-surgery inspect gemma-3-1b-it-Q4_K_M.gguf` | correct: gemma3, 26 layers, 4 heads / 1 KV head (MQA), head_dim 256, SWA 512, Q4_K_M, 999.89 M params |
| `model-surgery inspect llama-3.2-3b-instruct-Q4_K_M.gguf` | correct: llama, 28 layers, GQA 24/8, head_dim 128, 3.21 B params |
| `model-surgery inspect gemma-3-1b-it-Q5_K_M.gguf` (a 5.7 MB truncated download) | honest error: `truncated file: metadata 'tokenizer.ggml.token_type' array needs 1048576 bytes` |

Frontend tests: **none existed**. The TypeScript engine that powers the public site had no automated tests at all.
Browser tests: none. Playwright browsers not installed; system Chrome and Edge are available.

## 3. Missing capabilities (relative to the requested scope)

| Capability | State before |
|---|---|
| Inference / token generation | **Not implemented anywhere.** No runtime integration, no UI. |
| Measured performance | None. Only estimates (correctly labelled). |
| Settings / runtime detection page | None (tool detection lived inside the Quantize page). |
| Architecture search / layer navigation / fit-to-view / reset camera | None. |
| Error boundary | None — any render error blanks the app. |
| Light theme by default | Dark by default; light only via `prefers-color-scheme`. |

## 4. Verified defects

### Backend

| # | Severity | Defect | Location | How verified |
|---|---|---|---|---|
| B1 | high | Any unexpected exception after `Popen` in a job thread (e.g. Windows `PermissionError` on `os.rename` while AV holds the file, sqlite "database is locked") kills the thread, leaves the job `validating` forever and the `.partial` on disk; every later job is refused with "another job is already running". | `jobs.py` `_run` | reproduced (patched `os.rename` to raise) |
| B2 | medium | GGUF hyperparameters are stored without type checks. A string `head_count` makes `/memory` multiply a string by `ubatch×ctx` (488 MiB allocated at ctx=100k, then HTTP 500); `bool` values pass as ints. | `scanner.py` `kv()`, `memory.py` | reproduced |
| B3 | medium | A crafted header-only GGUF with n_params > 2⁶³ makes `/api/scan` return 500 (`OverflowError` writing SQLite INTEGER). Same via `/api/scans/import`. | `history.py` | reproduced |
| B4 | medium | Export of any model whose name contains non-Latin-1 characters returns 500 (`Content-Disposition` header encoding). | `api.py` export | reproduced (`Qwen中文模型`) |
| B5 | medium | `/api/scans/import` trusts the client-supplied `id`: importing with an existing local scan's id overwrites it (`INSERT OR REPLACE`). Body read fully before the size check. | `api.py` | reproduced |
| B6 | medium | Convert validation requires output params within ±5 % of the source; multimodal HF checkpoints lose their vision tower in conversion (ratio ~0.72–0.9), so a successful conversion is deleted as "failed". | `jobs.py` `validate_output` | code |
| B7 | medium | `mixer.*` (NemotronH-style) names are classified `ssm` before the attention rule, so attention layers vanish and the KV cache estimate falls back to all layers (≈9–13× too high). | `architecture.py` | reproduced |
| B8 | medium | Layer count can be inferred from a vision tower in multimodal HF checkpoints (stack chosen by tensor count, not bytes). | `scanner.py` `_complete_facts` | reproduced (llava-style fixture: 3 vs 4) |
| B9 | medium | DeepSeek-V2/V3 HF configs: `qk_nope_head_dim`/`qk_rope_head_dim`/`v_head_dim` ignored → head_dim 56 instead of K=192, V=128 → wrong KV estimate. | `scanner.py` | reproduced |
| B10 | low | HF `mlp.shared_experts.*` classified as routed `experts` (GGUF `*_shexp` is correct). | `architecture.py` | reproduced |
| B11 | low | Jamba `mamba.A_log` / `mamba.D` / `mamba.in_proj` → `other` (pattern `A_log` tested against lower-cased text). | `architecture.py` | reproduced |
| B12 | low | With `allowed_roots` configured, the outputs folder is listed as a root but its job subfolders return 403 and produced models cannot be re-scanned. | `config.py` `is_allowed` | reproduced |
| B13 | low | Partial-offload verdict says "about 4 of 4 layers fit in VRAM; the rest run from system RAM" when all layers fit. | `memory.py` `_fit` | reproduced |
| B14 | low | `GET /api/tools?refresh=true` runs subprocesses (up to 180 s) and is reachable by any cross-site page (GET is exempt from the CSRF header). | `api.py` | code |
| B15 | low | `JobManager.list()` iterates `_jobs` without the lock (rare "dict changed size" → mapped to 409). | `jobs.py` | code |

### Frontend

| # | Severity | Defect | Location | How verified |
|---|---|---|---|---|
| F1 | high | No error boundary. A failed lazy chunk (after a redeploy) or a WebGL renderer exception blanks the whole app. | `main.tsx`, `App.tsx` | code |
| F2 | medium | Browser edition, split GGUF: selecting only shard 2 re-reads it as "shard 2" → duplicated tensors, bogus "appears in more than one shard" errors, missing shard 1 not reported. Python engine is correct. | `engine/scanner.ts` | reproduced (Python vs TS diff on fixtures) |
| F3 | medium | A slow Hugging Face scan that finishes after the user opened another model replaces it and forces the tab back to Overview; Hub scans cannot be cancelled. | `HubImport.tsx`, `App.tsx` | code |
| F4 | medium | Deleting the currently open scan in History leaves it loaded; Memory Lab / Reports / Quantize then 404. | `App.tsx`, `HistoryPage.tsx` | code |
| F5 | medium | Quantize page: a plan response for old inputs can repopulate a plan after the inputs changed (and then be confirmed); a stale job poll can overwrite a newly selected job. | `QuantizePage.tsx` | code |
| F6 | medium | 3D view renders every frame forever (even when idle) with one mesh + one transparent line object per block (~960 draw calls for an 80-layer model). | `Architecture3D.tsx` | code |
| F7 | medium | Hub scans of sharded HF repos swallow network errors (429/403) as integrity errors and save a partial scan with wrong parameter counts; Python only catches format errors. | `engine/hf.ts` | code |
| F8 | medium | Mouse-only controls: layer rows, file-browser rows, sortable headers; `role="tablist"` without tabs. | several | code |
| F9 | low | WebGL contexts are not released on unmount (`dispose()` does not free the context) — after ~16 toggles the browser refuses new contexts; no context-loss handling. | `Architecture3D.tsx` | code + three.js source |
| F10 | low | Camera fit clips models deeper than ~85 layers. | `Architecture3D.tsx` | computed |
| F11 | low | Reports: the comparison table keeps showing the previous pair after the selection changes while "Markdown" exports the new one; copy fails silently without the Clipboard API. | `ReportsPage.tsx` | code |
| F12 | low | Engine parity: uint64 metadata above 2⁵³ loses precision in TS; `config.json` containing `NaN` fails in TS but not Python; half-up vs half-even rounding differences in a few displayed values. | `engine/*` | reproduced |

### Checked and found correct

Header parsers (bounds, limits, duplicates, alignment, overlap), path-traversal guards for shard indexes and filesystem browsing (resolve-before-check), host-header allowlist, CSRF header, no CORS, fixed-argv subprocesses with allowlisted basenames, single-use expiring confirmation tokens, source fingerprinting, `.partial` handling, CSV formula-injection guard, header-only reads in the browser (8 MB GGUF prefix grown on demand, 1 GiB cap), HF token scope (huggingface.co only, `credentials: "omit"`, no referrer). Memory formulas (KV, GQA, overrides, llama.cpp per-tensor type replay) were checked against the code; the TS engine matched the Python engine field-for-field on synthetic fixtures apart from F2/F12.

## 5. Network dependencies

No CDN assets, web fonts, analytics or telemetry. The only outbound requests are to `huggingface.co` (and its CDN) and happen **only** when the user pastes a Hub link. The Vercel CSP enforces this. Local model files are never uploaded in either edition. Offline operation had not been tested.

## 6. Environment limitations for this audit

- GPU tests ran on a 4 GB RTX 2050; no Jetson hardware was available. Jetson figures remain planning assumptions.
- Conversion (`convert_hf_to_gguf.py`) is not configured on this machine, so conversion was only exercised through the existing fake-tool tests.
- OS-level network isolation was not available; offline behaviour is tested by blocking all non-local requests inside the browser (see `TEST_REPORT.md`).

# Test report

Date: 2026-10-16 · Branch: `audit-and-upgrade` · Windows 11, Python 3.11.0, Node 24.14.1, npm 11.5.1, Google Chrome (system), 32 GB RAM, NVIDIA RTX 2050 4 GB, llama.cpp b8562 (CUDA build).

Every result below was produced by the commands shown, in this session. Nothing is extrapolated.

## 1. Summary

| Suite | Command | Result |
|---|---|---|
| Backend unit + integration | `cd backend && ..\.venv\Scripts\python -m pytest` | **178 passed, 1 skipped** (the skip is the opt-in real-model test) |
| Backend incl. real model | `MODEL_SURGERY_TEST_GGUF=<gemma-3-1b-it-Q4_K_M.gguf> … pytest` | **179 passed** |
| Frontend type check (app + tests) | `cd frontend && npm run typecheck` | pass |
| Frontend unit/integration (Vitest) | `npm test` | **45 passed** (39 engine, 6 Python↔TS parity) |
| Production builds | `npm run build`, `npm run build:browser` | pass |
| Browser/UI (Playwright, Chrome) | `npm run test:e2e` | **12 passed, 1 skipped** (the skip is real-model inference) |
| Browser/UI incl. real inference | `E2E_LLAMA_CPP_DIR=… E2E_GGUF=… npx playwright test` | **13 passed** |
| Dependency audit | `npm audit` (incl. dev) | **0 vulnerabilities** (baseline lockfile: 2 — esbuild via Vite 5.4.10) |
| Vite dev server smoke | `npx vite --port 5199` + `curl` | index and `/src/main.tsx` 200; `/api` proxied (500 with no backend running, as expected) |

Baseline before any change: 149 passed / 1 skipped backend; typecheck and both builds passing; no frontend tests of any kind.

## 2. What the tests cover

### Backend (`backend/tests`)

| File | Covers |
|---|---|
| `test_gguf.py`, `test_safetensors_hf.py`, `test_scanner.py` | metadata parsing, shards, corrupted/truncated/malicious headers, path traversal (pre-existing) |
| `test_architecture.py`, `test_memory.py` | classification, KV/weight estimates, llama.cpp type replay (pre-existing) |
| `test_api.py` | host guard, CSRF header, no CORS, export, import rules (updated for server-assigned ids, `server` tool key) |
| `test_jobs.py` | quantize plan/confirm/run/validate/cancel with a fake tool; **new**: worker crash frees the job slot (B1) |
| `test_audit_fixes.py` (**new**) | B2–B14: each test fails on the original code (verified by running against `git archive HEAD`) |
| `test_runtime.py` (**new**, 17 tests) | runtime lifecycle against a fake llama-server: log parsing (allocations, offload, n_ctx), capabilities, chat template and token streaming, raw prompts, exclusivity, cancel within 5 s, context overflow, crash during load, load timeout kills the process, model switching unloads the previous one, only local/valid/unchanged GGUF loadable, memory guard + override, NDJSON over HTTP with CSRF, API key never exposed, measured system memory, **orphan prevention** (hard-kill the host → the server child dies; Windows) |

### Frontend unit/integration (`frontend/src/engine/*.test.ts`)

GGUF facts and provenance, exact parameter counts, GGML shape order, header-only reads (bytes read bounded, never the whole file), non-numeric hyperparameters, truncated header, data past end of file, bad magic, pickle refusal without reading, unsupported formats, safetensors header length and duplicate keys, split GGUF (F2, fails on original code), sharded HF folder, DeepSeek MLA dims, 15 classification cases, KV formula, weight size equal to stored bytes, "never fits just because the weights are smaller than the device", all-layers-on-GPU verdict, architecture graph (pre-norm layout, residual edges, illustrative nodes, provenance of dims, tied head, unknown tensors listed, MoE router/experts/shared), CSV formula guard, comparison dtype changes.

`parity.test.ts` writes byte-identical fixtures (F16, Q8 tied, truncated MQA, unknown tensors, non-numeric hyperparameter, single safetensors), scans them with the Python engine (`../.venv`) and the TS engine, and requires identical scans and identical memory estimates for two request profiles.

### Browser (`frontend/e2e`, Chrome)

Browser edition (served by `vite preview` with the production CSP from `vercel.json`):
- startup in light theme;
- import → 2D → search/jump → keyboard layer navigation → dataflow → 3D (fit, reset, keyboard, orbit drag does not change selection, labels) → isolate → back to 2D (canvas disposed);
- graph JSON export;
- bad magic and pickle errors announced;
- inference shown as unavailable;
- **zero requests outside the page origin** across every page including 3D;
- phone width (390 px) navigation with no horizontal scroll;
- theme persistence.

Every test asserts no uncaught page errors or console errors.

Local app (real FastAPI server, throwaway data dir):
- bad path then good path;
- deleting the open model;
- Settings runtime detection and measured memory;
- capability matrix;
- **real inference** with gemma-3-1b on CPU: load → 16 deterministic tokens streamed → token alternatives → prompt tokens → stop a 2,000-token generation early → measured table on Performance → unload.

## 3. Real-hardware measurements recorded during testing

These are measurements from llama.cpp or the OS on this laptop, quoted only as evidence that the pipeline reports real data. They are not benchmarks.

| Model | Settings | Load | Prompt | Decode | First token (wall) | RSS (OS) |
|---|---|---|---|---|---|---|
| gemma-3-1b-it Q4_K_M | CPU only, ctx 2048 | 1.81 s | 13 tok in 124 ms (104.7 tok/s) | 24 tok in 1,110 ms (21.6 tok/s) | 178 ms | 1.10 GiB |
| gemma-3-1b-it Q4_K_M | auto offload (27/27 on CUDA0), ctx 2048 | 5.52 s | — | — | — | 1.16 GiB |
| llama-3.2-3b-instruct Q4_K_M | auto offload (24/29 layers), ctx 4096 | — | 45 tok, 223 tok/s | 64 tok, 12.5 tok/s | 309 ms | 2.37 GiB |

Estimate vs runtime-reported allocation:

| Model / ctx | Component | Model Surgery estimate | llama.cpp reported |
|---|---|---|---|
| llama-3.2-3b, 4096 | KV cache (F16) | 448.00 MiB | 448.00 MiB (exact) |
| llama-3.2-3b, 4096 | compute + output | 56.49 MiB (heuristic) | 301.00 MiB (CUDA graph buffers) |
| gemma-3-1b, 2048 | KV cache | 52.00 MiB (full context, every layer — upper bound) | 30 MiB (4 global layers × 2048 + 22 sliding-window layers × 1024 cells) |

The compute-buffer heuristic and the sliding-window KV upper bound are documented limitations and are explained in the UI next to these numbers.

## 4. Environment limits of this test run

- **Offline:** tested by blocking every non-origin request inside Chrome, not by disabling the OS network adapter.
- **Jetson Orin Nano:** no hardware was available. Jetson profiles remain planning assumptions; the orphan-prevention Job Object is Windows-only (see §5).
- **Conversion** (`convert_hf_to_gguf.py`) is not configured on this machine and was covered only by the existing fake-tool tests.
- **Browsers:** Firefox and Safari were not tested. Chrome was, through Playwright `channel: "chrome"`.
- **GPU:** VRAM usage is reported through llama.cpp's allocation log, not through NVML/nvidia-smi.

## 5. Unresolved issues, impact, next steps

| Issue | Impact | Next step |
|---|---|---|
| On Linux/Jetson, a hard kill of Model Surgery can still orphan llama-server (Windows uses a kill-on-close Job Object; normal shutdown and Ctrl+C clean up on every OS) | memory stays allocated until the process is killed by hand | run under systemd (`KillMode=control-group`) or add a supervisor/`PR_SET_PDEATHSIG` spawned from a dedicated long-lived thread |
| No layer-by-layer trace, attention maps or activations | none hidden: the UI states the runtime cannot provide them | would need llama.cpp's eval callback (`llama-eval-callback`) or a PyTorch backend with hooks |
| Compute-buffer estimate is a heuristic, about 5× too low on CUDA | estimate total understates GPU use | model llama.cpp's graph-reserve sizing per backend, or calibrate from the reported allocations |
| KV estimate ignores sliding-window caches (Gemma 2/3, Cohere2) | upper bound (labelled) | add per-architecture SWA patterns matching llama.cpp (`n_swa + n_ubatch`, padded) |
| Generation measurements are kept in memory only | lost on server restart | persist in the SQLite history |
| Browser edition cannot run inference | by design | optional WebGPU runtime (e.g. wllama) for small GGUFs, clearly separated |
| uint64 metadata above 2^53 loses precision in the browser (also when the local API's JSON is parsed by the browser) | display of rare huge integers only | serialise large ints as strings in both engines |
| npm 11.5.1/10.9.2 crash resolving vitest ≥ 4 (optional-peer cycle) | `npm install vitest@…` needs `--legacy-peer-deps`; `npm ci` is unaffected | revisit when npm/vitest fix it |
| `Architecture3D` chunk is 504 kB (three.js) | lazy-loaded; only fetched when 3D is opened | tree-shake via `three/src` imports if size matters |
| Accessibility audited by hand, not with an automated axe scan | possible remaining issues | add `@axe-core/playwright` to the e2e suite |

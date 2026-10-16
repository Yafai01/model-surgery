# Implementation report

Branch: `audit-and-upgrade` (from `main` @ `c5964d2`). Nothing has been committed or deployed. Defect numbers (B…, F…) refer to [`INITIAL_AUDIT.md`](INITIAL_AUDIT.md). Commands and results are in [`TEST_REPORT.md`](TEST_REPORT.md).

## Stage B/C — defects fixed

### Backend

| # | Fix | Files | Regression test |
|---|---|---|---|
| B1 | Everything after `Popen` in a job runs inside a supervisor that, on any unexpected exception, kills the tool, removes the `.partial`, marks the job failed and frees the slot; the post-validation scan catches all exceptions | `jobs.py` | `test_jobs.py::test_unexpected_worker_error_fails_job_and_frees_slot` |
| B2 | GGUF hyperparameters are accepted only when numeric and not `bool`; others are ignored with a note | `scanner.py` (+ TS port) | `test_audit_fixes.py::test_b2…`, `engine.test.ts` |
| B3 | Integers outside SQLite's 64-bit range are stored as NULL in the summary columns (the full scan blob keeps the exact value) | `history.py` | `test_b3…` |
| B4 | Export filename is ASCII-only with an RFC 5987 `filename*` for the real name | `api.py` | `test_b4…` |
| B5 | `/api/scans/import` checks `Content-Length` and streams with a cap; the scan id is assigned by the server | `api.py` | `test_b5…`, updated `test_api.py` |
| B6 | Convert validation compares against language-model tensors when the source also has vision/audio towers (they go to a separate mmproj) | `jobs.py` | `test_b6…` |
| B7, B10, B11 | Classification: `mixer.{q,k,v,o,up,down,gate}_proj` are attention/FFN, `shared_experts.*` is FFN, `mamba.*` and `A_log` are SSM | `architecture.py` (+ TS port) | `test_b7_b10_b11…`, `engine.test.ts` |
| B8 | The layer count is inferred from the stack with the most bytes (same rule as the architecture view), not the most tensors | `scanner.py` (+ TS port) | `test_b8…` |
| B9 | DeepSeek MLA: head_dim K = `qk_nope_head_dim + qk_rope_head_dim`, V = `v_head_dim` | `scanner.py` (+ TS port) | `test_b9…`, `engine.test.ts` |
| B12 | The outputs folder is always allowed, so produced models can be re-scanned when `allowed_roots` is set | `config.py` | `test_b12…` |
| B13 | When every layer fits on the GPU, the verdict is "tight" (embedding stays in host RAM), not "partial offload" | `memory.py` (+ TS port) | `test_b13…`, `engine.test.ts` |
| B14 | Tool re-detection (runs subprocesses) moved to `POST /api/tools/refresh` | `api.py`, `api.ts` | `test_b14…` |
| B15 | `JobManager.list()` copies the job ids under the lock | `jobs.py` | — |

All 11 tests in `test_audit_fixes.py` were run against the original code and **all fail there**; they pass on the fixed code.

### Frontend

| # | Fix | Files |
|---|---|---|
| F1 | Error boundaries around the app, every page (keyed by tab and scan) and the 3D view (falls back to 2D) | `ErrorBoundary.tsx`, `main.tsx`, `App.tsx`, `ArchitecturePage.tsx` |
| F2 | Split GGUF: walk shards 1..N like the Python engine; a lone second shard no longer duplicates itself, and the missing first shard is reported. Regression test fails on the old code. | `engine/scanner.ts`, `engine.test.ts` |
| F3 | Navigation tickets in `App` ignore late results; Hub scans take an `AbortSignal`, have a Cancel button and abort when the user leaves Import; a scan finishing in the background is saved with a notice instead of hijacking the view | `App.tsx`, `HubImport.tsx`, `engine/hfhub.ts`, `api.ts` |
| F4 | Deleting the open scan clears it and leaves pages that need a model | `App.tsx`, `HistoryPage.tsx` |
| F5 | Quantize: plan responses for outdated inputs are discarded; stale job polls are ignored | `QuantizePage.tsx` |
| F6, F9, F10 | 3D rewrite (see Stage D) | `Architecture3D.tsx` |
| F7 | Hub shard read failures (429/403/network) abort the scan instead of becoming integrity errors | `engine/hf.ts`, `engine/scanner.ts` |
| F8 | Layer rows, sortable headers, file-browser rows and selectable table rows are keyboard-operable; error callouts are announced (`role="alert"`); spinner has `role="status"`; search inputs labelled | several |
| F11 | Comparison table clears when its inputs change and ignores stale responses; copy reports a missing Clipboard API | `ReportsPage.tsx` |
| new | Below 760 px the sidebar was hidden with **no** replacement, so phones had no navigation. Added a compact navigation bar. | `App.tsx`, `styles.css` |
| new | The WebGL probe ran on every Architecture visit and leaked a context; it now runs once and releases the context | `ArchitecturePage.tsx` |

## Stage D — 3D visualization

- One `InstancedMesh` for all blocks and one merged outline geometry: an 80-layer model drops from about 960 draw calls to 2–4.
- Rendering happens on demand only; it is idle when the camera is still.
- Fit selection / fit all / reset, using a field-of-view-correct bounding-sphere fit (deep models no longer clip).
- Keyboard control: ↑/↓/J/K step through layers, F fits, R resets. Pan and zoom go through OrbitControls.
- Search and isolate: matching layers stay coloured while others fade; component chips ghost the hidden components.
- Sparse layer labels and a hover tooltip, positioned imperatively.
- Quality presets (high/balanced/low) in Settings; colours follow the light/dark theme.
- `forceContextLoss()` on unmount, a context-loss restart prompt, and a 2D fallback through the error boundary.
- The caption states what is schematic: widths are proportional to bytes; heights, depth and spacing are illustrative.

## Stage C — normalized architecture schema

New `engine/graph.ts` (described in ARCHITECTURE.md §3) builds a per-layer dataflow graph with provenance on every node and dimension, marks illustrative nodes, recognises pre-norm, parallel, SSM and MoE layouts, and lists unplaceable tensors instead of guessing. It drives the new **Layer dataflow** view for the selected layer (replacing the old layer-0-only diagram, which drew a single chain with no residual connections) and the **Graph JSON** export.

Also on the Architecture page: tensor search with match counts, Enter to jump, prev/next/go-to-layer navigation, and stack switching that resets the selection.

## Stage E — genuine inference

Before this work there was no inference capability. Added:

- `backend/model_surgery/runtime.py`: llama-server lifecycle, a memory guard, NDJSON streaming, cancellation, log parsing and measurements. `system.py` reads OS memory with no new dependencies. `tooling.py` detects `llama-server`.
- Endpoints: `GET /api/runtime`, `POST /api/runtime/{start,stop,cancel,tokenize,generate}`, `GET /api/runtime/measurements` and `GET /api/system`.
- **Inference page:**
  - A capability matrix states plainly that layer traces, attention maps and activations are not available.
  - Load options: context length, GPU offload (auto / CPU / custom) and threads, with the estimate shown next to the RAM the OS reports as available. A "start anyway" path is offered only after an explicit warning.
  - Chat-template or raw prompts.
  - Output: a streamed token view showing token boundaries, ids and the runtime's top-k alternatives. The prompt is shown exactly as evaluated, as token chips.
  - Stats: llama.cpp timings labelled as such, and wall-clock time to first token labelled as wall clock.
  - Controls: Stop and Unload.
- **Performance page:** a new "Measured on this machine" panel shows estimate vs llama.cpp-reported allocation per component. Where a memory-mapped range overlaps GPU copies, it shows the split instead of a misleading percentage. It also lists the generation runs recorded this session.
- The browser edition reports the runtime as honestly unavailable.

Why llama-server: it is the llama.cpp ecosystem the project already integrates, it ships with the user's existing build, and it exposes token ids, probabilities and timings over HTTP. Nothing is installed into the frontend.

Two defects found and fixed while building this (both covered by tests):
- **Generation cancel did nothing.** When the server replies with `Connection: close`, `http.client` detaches the socket from the connection object; cancel now shuts down a retained socket reference.
- **Orphaned llama-server after a hard kill.** A forced kill or crash of Model Surgery left llama-server running with the model in memory (reproduced). On Windows the child is now in a kill-on-close Job Object. A test hard-kills a host process and asserts the child dies; the same scenario without the Job Object was confirmed to leave the child alive.

## Stage F — UI

- **Theme:** light by default, with neutral Apple/Linear-style tokens (white surfaces, `#0071e3` accent, subtle shadows, no gradients or glow). Dark and system themes are selectable in Settings. Every former hard-coded dark surface (3D canvas, tooltip, log, progress bar) is now a token.
- **Navigation:** regrouped into Model · Architecture (Overview, Architecture, Tensors, Metadata) · Run & measure (Inference, Performance, Quantize) · Records (Compare & export, History, Settings).
- **Settings page:** theme, 3D quality, labels, WebGL renderer, runtime and tool detection with re-detect, measured system memory, data/output/allowed-root paths, and a network/privacy statement.
- **Accessibility:** `:focus-visible` outlines, `prefers-reduced-motion`, and phone-width layout with no horizontal scroll (tested).

## Stage G — estimates, comparison, export

The existing estimator (llama.cpp per-tensor type replay) and the comparison/export features were kept. They gain:
- measured-vs-estimate comparison;
- the graph JSON export;
- stale-comparison fixes;
- Python/TS parity coverage for the estimator.

Real-model check on this machine: for llama-3.2-3b Q4_K_M at 4,096 context, the KV-cache estimate (448 MiB) equals llama.cpp's reported allocation exactly.

## Stage H — offline and security

- **Network:** no external assets were present. A browser test now blocks every request outside the page's origin and walks through all pages, including 3D. Zero external requests were made.
- **Security additions:** loopback-only llama-server with a random API key (llama-server otherwise answers cross-origin requests from any website), plus the B4/B5/B14 fixes.

## Tooling changes

| Change | Why |
|---|---|
| Added `vitest` 4.1.11 + `src/engine/*.test.ts` + `src/test/fixtures.ts` | The TypeScript engine behind the public site had no tests |
| Added `@playwright/test` 1.63.0 + `e2e/` (runs on the installed Google Chrome via `channel: "chrome"`; no browser download) | Browser/UI coverage for both editions |
| `vite` 5.4.10 → **6.4.3**, `@vitejs/plugin-react` 4.3.3 → 4.7.0 | Vitest versions free of the tinypool RCE and `@vitest/mocker` advisories (≥ 4.1.11) require Vite ≥ 6; Vite 6.4.3 also clears the pre-existing esbuild dev-server advisory (GHSA-67mh-4wv8-2f99). `npm audit` (including dev deps): 2 → **0**. |
| `@types/node` 20.17.6 (dev) with a separate `tsconfig.test.json` | Node types for tests without leaking into app code (`"types": []`) |

**npm note:** npm 11.5.1 (and 10.9.2) crash with `Cannot read properties of null (reading 'edgesOut')` when resolving `vitest@4.1.11` because of a cycle of optional peers (`@vitejs/devtools-vitest` → `vitest@*` → `@vitest/browser-*`). The lockfile was therefore written with `npm install --legacy-peer-deps --package-lock-only`. A plain `npm ci` (what `vercel.json` runs) installs from it without flags; this was verified on a clean `node_modules`. Future `npm install vitest@…` upgrades will need the same flag until npm fixes this.

## Unresolved items

See TEST_REPORT.md §5 for the full list with impact and next steps.

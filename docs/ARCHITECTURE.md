# Architecture

Model Surgery is one React UI with two interchangeable back ends, plus an optional local inference runtime.

```text
                    ┌────────────────────────── React UI (frontend/src) ──────────────────────────┐
                    │ Import · Overview · Architecture (2D/3D/dataflow) · Tensors · Metadata       │
                    │ Inference · Performance · Quantize · Compare & export · History · Settings   │
                    └───────────────┬─────────────────────────────────────────┬───────────────────┘
                                    │ api.ts (Api interface)                  │
              local edition (default build)                      browser edition (vite --mode browser)
                                    │                                         │
              HTTP to FastAPI on 127.0.0.1:8765               in-page engine: src/engine/*.ts
                                    │                         (File API / HTTP range requests,
      ┌─────────────────────────────┼───────────────────────┐  IndexedDB history; no runtime)
      │ backend/model_surgery       │                       │
      │  formats/ gguf, safetensors, hf  (untrusted-input parsers, header only)
      │  scanner.py    path → ScanResult (facts with provenance)
      │  architecture.py  tensor names → layers / components (inferred)
      │  memory.py, devices.py   estimates (never measurements)
      │  runtime.py   llama-server lifecycle, streaming, measurements   ──► llama-server (child process,
      │  system.py    OS memory readings (measured)                         127.0.0.1, random port + key)
      │  tooling.py, jobs.py   llama.cpp detection, quantize/convert jobs ──► llama-quantize / convert script
      │  history.py (SQLite) · reports.py · api.py · cli.py
      └─────────────────────────────────────────────────────┘
```

## 1. Two editions, one UI

`src/api.ts` defines an `Api` interface with two implementations:

| | Local edition (`npm run build`, served by `model-surgery serve`) | Browser edition (`npm run build:browser`, deployed to Vercel) |
|---|---|---|
| Model files | read by the Python server from a local path | read by the page with the File API (header slices only) |
| Engine | `backend/model_surgery` (Python) | `frontend/src/engine` (TypeScript port) |
| History | SQLite in the data directory | IndexedDB in the browser |
| Inference, measurements | llama-server via `runtime.py` | not available (stated in the UI) |
| Quantize/convert | llama.cpp tools via `jobs.py` | not available |

The two engines are kept in step by `src/engine/parity.test.ts`, which feeds byte-identical synthetic files to both and requires identical scans and memory estimates.

## 2. Model data flow

```text
file bytes (header only) ──► format parser ──► tensor descriptors + metadata      [verified]
                                   │
                                   ├─► hyperparameter facts (key, value, provenance, source)
                                   │      verified = read from the file
                                   │      derived  = computed from verified values (n_params, head_dim)
                                   │      inferred = heuristic (layer count from tensor names, …)
                                   ├─► architecture summary: stacks → layers → components   [inferred roles]
                                   └─► ScanResult (stored in history)
ScanResult ──► engine/graph.ts ──► ArchitectureGraph (normalized, see §3)
ScanResult + MemoryRequest ──► memory.py / memory.ts ──► MemoryEstimate (is_measurement: false)
ScanResult + RuntimeStartRequest ──► runtime.py ──► llama-server ──► token stream, timings, allocations [measured]
```

Parsers never read tensor data. GGUF readers grow a header window on demand (8 MB initial, 1 GiB cap); safetensors readers read 8 bytes plus the JSON header (100 MB cap). Every offset and size is bounds-checked against the real file size.

## 3. Normalized architecture schema

`frontend/src/engine/graph.ts` turns a scan into `model-surgery/architecture-graph@1` (exported from the Architecture page as JSON):

```ts
ArchitectureGraph {
  schema, model, family, stack, n_layers,
  head: GraphNode[]          // token ids → token embedding
  layers: LayerGraph[]       // one dataflow graph per layer
  tail: GraphNode[]          // final norm → output head → logits
  unrepresented: string[]    // tensors that could not be placed — listed, never forced into a slot
  notes: string[]
}
LayerGraph { index, layout: "sequential_prenorm" | "parallel" | "ssm" | "unknown", nodes, edges, notes }
GraphNode  { id, kind, label, layer, tensors[], n_elements, n_bytes, dims: Dim[], provenance, illustrative }
GraphEdge  { from, to, kind: "data" | "residual" }
Dim        { label, value, provenance }   // e.g. "KV heads: 8 (verified)", or a tensor shape (verified)
```

- Nodes backed by tensors carry the tensor names (verified); their *role* is inferred from naming conventions.
- Residual adds, the input/output hidden state, `logits` and a tied output head hold no tensors: `illustrative: true`.
- Layouts are recognised from which norms exist: pre-norm sequential (Llama, Qwen, Gemma with post-norms), parallel attention+MLP (GPT-J/Falcon), SSM-only, or `unknown` (components listed without edges).
- MoE layers get router → routed experts, plus a shared-expert FFN summed in parallel when present.

## 4. Local inference (`backend/model_surgery/runtime.py`)

```text
UI ──POST /api/runtime/start──► RuntimeManager.start
        checks: llama-server detected · scan is a local, valid GGUF · file unchanged since scan ·
                inside allowed roots · estimate ≤ RAM the OS reports available (or user override)
        spawns: llama-server -m <file> --host 127.0.0.1 --port <random> -c <ctx> -np 1 --no-webui
                --api-key <random, never leaves the process> [-ngl N] [-t N]
        Windows: process assigned to a kill-on-close Job Object (dies if Model Surgery dies)
        threads: log reader (parses buffer allocations, offload, n_ctx) · health poller · exit watcher
UI ──POST /api/runtime/generate──► NDJSON stream
        /apply-template (chat) → /tokenize (ids + pieces, exactly what is evaluated) →
        /completion stream with n_probs + return_tokens → events:
        prompt · token{content, ids, top-k logprobs, t_ms} · done{llama.cpp timings, wall clock} · cancelled · error
        client disconnect or POST /api/runtime/cancel → socket shutdown → llama-server stops the request
```

What is shown, and where it comes from:

| Value | Source |
|---|---|
| tokens, ids, pieces, top-k probabilities | llama-server |
| prompt/decode tokens per second, prompt ms | llama.cpp `timings` |
| time to first token, inter-token gaps | wall clock in Model Surgery (includes HTTP and tokenizing) |
| buffer allocations (model, KV, compute) per device | llama.cpp load log |
| process resident memory | the OS (`GetProcessMemoryInfo` / `/proc/<pid>/status`) |
| system total/available RAM | the OS (`GlobalMemoryStatusEx` / `/proc/meminfo`) |
| layer-by-layer trace, attention maps, activations | **not available** — llama-server has no API for them; the UI says so |

The Vercel deployment is static. It never runs inference and contains no serverless functions.

## 5. 3D view (`Architecture3D.tsx`)

- One `InstancedMesh` for every block plus one merged `LineSegments` for outlines (2–4 draw calls regardless of depth; previously one mesh and one line object per block).
- Render on demand: frames are drawn on camera changes, resize and restyle only; damping keeps requesting frames until it settles.
- Fit-to-view uses the bounding sphere and both fields of view; keyboard: ↑/↓ (or J/K) step layers, F fits the selection, R resets.
- Search matches and component isolation recolour instances; selection is a single outline box.
- Labels and tooltip are positioned imperatively (no React re-render per frame or pointer move).
- Quality presets (Settings): antialiasing, pixel-ratio cap and outlines.
- Cleanup disposes geometry/materials and calls `forceContextLoss()`; context loss shows a restart prompt; an error boundary falls back to 2D.

## 6. Security boundaries

| Boundary | Control |
|---|---|
| Browser → local API | Host-header allowlist; `X-Model-Surgery` header required on every non-GET (CORS preflight never granted); no CORS headers; subprocess-triggering refresh is POST-only |
| Untrusted model files | header-only parsers with limits and bounds checks; pickle formats refused; no model code executed |
| Imported Hub scans | only `hf://` scans accepted; size checked before reading; id assigned by the server |
| Subprocesses | executables from server config only, basename allowlisted, fixed argv, no shell |
| llama-server | loopback only, random port, random API key, one instance, killed on stop/exit (Job Object on Windows) |
| Exports | CSV formula-injection guard; Markdown escaping; RFC 5987 filenames |

## 7. Frontend state

There is no router or global store. `App.tsx` owns the open scan, the tab, history and preferences:

- Every load that changes navigation takes a ticket (`loadSeq`), so a slow result never replaces something the user opened since. A scan that finishes after the user left Import is saved to History with a notice.
- Deleting the open scan clears it.
- Each page is wrapped in an error boundary keyed by tab and scan.
- Preferences (theme, 3D quality, labels) live in `prefs.ts` (guarded `localStorage`).

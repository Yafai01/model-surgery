import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Tab } from "../App";
import { ApiError, BROWSER_EDITION, api } from "../api";
import { fmtBytes, fmtInt } from "../format";
import type { GenerateRequest, GenerationEvent, MemoryEstimate, RuntimeStatus, ScanResult, SystemInfo, TokenProb } from "../types";
import { Callout, EstimateBanner, Spinner } from "./common";

interface Props {
  scan: ScanResult;
  onError: (m: string) => void;
  onNavigate: (t: Tab) => void;
}

const CAPS: [key: string, label: string][] = [
  ["token_stream", "Streamed text"],
  ["token_ids", "Token ids & pieces"],
  ["top_k_logprobs", "Top-k token probabilities"],
  ["runtime_timings", "Prompt / decode timings"],
  ["runtime_allocations", "Buffer allocations"],
  ["layer_trace", "Layer-by-layer forward trace"],
  ["attention_maps", "Attention maps"],
  ["activations", "Intermediate activations"],
];

interface OutToken {
  content: string;
  ids: number[];
  prob: TokenProb | null;
  t_ms: number;
}

type Done = Extract<GenerationEvent, { type: "done" }>;
type PromptEv = Extract<GenerationEvent, { type: "prompt" }>;

const pct = (logprob: number) => `${(Math.exp(logprob) * 100).toFixed(logprob > -0.0005 ? 1 : 2)}%`;
const visible = (s: string) => (s === "" ? "∅" : s.replace(/\n/g, "↵\n"));

export default function InferencePage({ scan, onError, onNavigate }: Props) {
  const [rt, setRt] = useState<RuntimeStatus | null>(null);
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const trainCtx = scan.hyperparameters.find((f) => f.key === "context_length")?.value;
  const [ctx, setCtx] = useState(typeof trainCtx === "number" ? Math.min(4096, trainCtx) : 4096);
  const [gpuMode, setGpuMode] = useState<"auto" | "cpu" | "custom">("auto");
  const [gpuLayers, setGpuLayers] = useState(99);
  const [threads, setThreads] = useState("");
  const [starting, setStarting] = useState(false);
  const [memBlock, setMemBlock] = useState<string | null>(null);
  const [est, setEst] = useState<MemoryEstimate | null>(null);

  const [prompt, setPrompt] = useState("Explain in two sentences what a KV cache is.");
  const [system, setSystem] = useState("");
  const [chat, setChat] = useState(true);
  const [nPredict, setNPredict] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.95);
  const [seed, setSeed] = useState("");
  const [nProbs, setNProbs] = useState(5);

  const [promptEv, setPromptEv] = useState<PromptEv | null>(null);
  const [tokens, setTokens] = useState<OutToken[]>([]);
  const [done, setDone] = useState<Done | null>(null);
  const [stopped, setStopped] = useState<string | null>(null);
  const [notices, setNotices] = useState<string[]>([]);
  const [generating, setGenerating] = useState(false);
  const [boundaries, setBoundaries] = useState(true);
  const [picked, setPicked] = useState<number | null>(null);
  const ctrl = useRef<AbortController | null>(null);

  const isLocalGguf = scan.kind === "gguf" && !scan.path.startsWith("hf://");

  const refresh = useCallback(async () => {
    try {
      setRt(await api.runtime());
    } catch (e) {
      onError((e as Error).message);
    }
  }, [onError]);

  useEffect(() => {
    refresh();
    api.system().then(setSys).catch(() => setSys(null));
    return () => ctrl.current?.abort(); // leaving the page stops a running generation
  }, [refresh]);

  // Poll quickly while loading, slowly (for the measured RSS) while ready.
  useEffect(() => {
    if (!rt || (rt.state !== "starting" && rt.state !== "ready")) return;
    let cancelled = false;
    const t = setInterval(
      async () => {
        try {
          const s = await api.runtime();
          if (!cancelled) setRt(s);
        } catch {
          /* transient */
        }
      },
      rt.state === "starting" ? 700 : 5000,
    );
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [rt?.state]); // re-armed only when the state changes, not on every poll result

  // Memory estimate for the chosen context, shown next to what the OS reports as available.
  useEffect(() => {
    let cancelled = false;
    const t = setTimeout(() => {
      api
        .memory(scan.id, { weight_format: "current", context_length: ctx, batch_size: 1, kv_cache_type: "f16", flash_attention: true, n_ubatch: 512, device_id: null, custom_device: null })
        .then((m) => !cancelled && setEst(m))
        .catch(() => !cancelled && setEst(null));
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [scan.id, ctx]);

  const loadedHere = rt?.scan_id === scan.id && (rt.state === "ready" || rt.state === "starting");
  const ready = rt?.state === "ready" && rt.scan_id === scan.id;

  const start = async (allowRisk: boolean) => {
    setStarting(true);
    setMemBlock(null);
    try {
      setRt(
        await api.runtimeStart({
          scan_id: scan.id,
          context_length: ctx,
          n_gpu_layers: gpuMode === "auto" ? null : gpuMode === "cpu" ? 0 : gpuLayers,
          threads: threads ? Number(threads) : null,
          allow_memory_risk: allowRisk,
        }),
      );
    } catch (e) {
      const msg = (e as Error).message;
      if (e instanceof ApiError && e.status === 409 && /exceeds the .* available/.test(msg)) setMemBlock(msg);
      else onError(msg);
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    ctrl.current?.abort();
    try {
      setRt(await api.runtimeStop());
    } catch (e) {
      onError((e as Error).message);
    }
  };

  const generate = async () => {
    ctrl.current?.abort();
    const c = (ctrl.current = new AbortController());
    setGenerating(true);
    setPromptEv(null);
    setTokens([]);
    setDone(null);
    setStopped(null);
    setNotices([]);
    setPicked(null);
    const req: GenerateRequest = {
      prompt,
      chat,
      system: chat && system.trim() ? system : null,
      n_predict: nPredict,
      temperature,
      top_k: topK,
      top_p: topP,
      seed: seed.trim() ? Number(seed) : null,
      n_probs: nProbs,
    };
    const buffered: OutToken[] = [];
    let raf = 0;
    const flush = () => {
      raf = 0;
      setTokens([...buffered]);
    };
    try {
      await api.generate(
        req,
        (ev) => {
          if (c.signal.aborted) return;
          if (ev.type === "token") {
            buffered.push({ content: ev.content, ids: ev.ids, prob: ev.probs[0] ?? null, t_ms: ev.t_ms });
            if (!raf) raf = requestAnimationFrame(flush); // batch renders to the display rate
          } else if (ev.type === "prompt") setPromptEv(ev);
          else if (ev.type === "done") setDone(ev);
          else if (ev.type === "cancelled") setStopped("stopped by user");
          else if (ev.type === "notice") setNotices((n) => [...n, ev.message]);
          else if (ev.type === "error") onError(ev.message);
        },
        c.signal,
      );
    } catch (e) {
      onError((e as Error).message);
    } finally {
      if (raf) cancelAnimationFrame(raf);
      setTokens([...buffered]);
      if (c.signal.aborted) setStopped("stopped by user");
      if (ctrl.current === c) {
        ctrl.current = null;
        setGenerating(false);
      }
      refresh();
    }
  };

  const stopGenerating = () => {
    ctrl.current?.abort(); // closes the stream; the server cancels llama-server's request
    api.runtimeCancel().catch(() => undefined);
  };

  const gaps = useMemo(() => tokens.map((t, i) => (i === 0 ? t.t_ms : t.t_ms - tokens[i - 1].t_ms)), [tokens]);

  // ------------------------------------------------------------------ render
  const unavailable = BROWSER_EDITION || rt?.state === "unavailable";
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Inference</h2>
          <p>Runs this GGUF on your machine with llama.cpp's llama-server. Everything on this page is reported by the runtime or measured here — nothing is simulated.</p>
        </div>
        {rt && <span className={`badge ${rt.state === "ready" ? "ok" : rt.state === "failed" ? "bad" : rt.state === "starting" ? "warn" : ""}`}>runtime: {rt.state}</span>}
      </div>

      <div className="card">
        <div className="card-head">
          <h3>What this runtime can show</h3>
          {rt?.tool.version && <span className="small faint">llama-server {rt.tool.version}</span>}
        </div>
        <div className="cap-grid">
          {CAPS.map(([k, label]) => {
            const on = !!rt?.capabilities[k];
            return (
              <div key={k} className={`cap ${on ? "on" : "off"}`}>
                <span aria-hidden>{on ? "✓" : "—"}</span> {label}
                <span className="sr-only">{on ? " available" : " not available"}</span>
              </div>
            );
          })}
        </div>
        <p className="small dim" style={{ marginTop: 10 }}>
          {rt?.trace_note ?? "Checking the runtime…"} The 3D architecture view shows the model's structure; it is not animated during generation, because the runtime does not report which layer is executing.
        </p>
      </div>

      {unavailable ? (
        <Callout kind="warn" title="No runtime available.">
          {BROWSER_EDITION ? (
            <>The browser edition only reads file headers; it cannot execute GGUF or safetensors models. Install the local app and point it at a llama.cpp build to generate text.</>
          ) : (
            <>
              {rt?.tool.detail} Set <code>MODEL_SURGERY_LLAMA_CPP_DIR</code> (or <code>--llama-cpp-dir</code>) to a folder containing <code>llama-server</code>, then restart Model Surgery. See{" "}
              <button className="linklike" onClick={() => onNavigate("settings")}>
                Settings
              </button>
              .
            </>
          )}
        </Callout>
      ) : !isLocalGguf ? (
        <Callout kind="warn" title="This model cannot be run here.">
          Inference needs a local GGUF file. {scan.path.startsWith("hf://") ? "This scan was read remotely from the Hugging Face Hub (headers only)." : "Safetensors checkpoints can be inspected, and converted to GGUF on the Quantize page."}
        </Callout>
      ) : (
        <div className="grid grid-2" style={{ alignItems: "start" }}>
          <div className="card">
            <div className="card-head">
              <h3>Runtime</h3>
              {rt && rt.state !== "stopped" && rt.model_name && <span className="small faint">{rt.scan_id === scan.id ? "this model" : `loaded: ${rt.model_name}`}</span>}
            </div>
            {!loadedHere && (
              <div className="col">
                <div className="grid grid-3">
                  <label className="field">
                    Context length
                    <input type="number" min={128} max={262144} step={128} value={ctx} onChange={(e) => setCtx(Math.max(128, Number(e.target.value) || 128))} />
                  </label>
                  <label className="field">
                    GPU offload
                    <select value={gpuMode} onChange={(e) => setGpuMode(e.target.value as typeof gpuMode)}>
                      <option value="auto">auto (llama.cpp fits to free VRAM)</option>
                      <option value="cpu">CPU only (-ngl 0)</option>
                      <option value="custom">custom layer count</option>
                    </select>
                  </label>
                  {gpuMode === "custom" ? (
                    <label className="field">
                      GPU layers
                      <input type="number" min={0} max={1000} value={gpuLayers} onChange={(e) => setGpuLayers(Math.max(0, Number(e.target.value) || 0))} />
                    </label>
                  ) : (
                    <label className="field">
                      CPU threads
                      <input type="number" min={1} max={256} placeholder="auto" value={threads} onChange={(e) => setThreads(e.target.value)} />
                    </label>
                  )}
                </div>
                {est && (
                  <EstimateBanner>
                    Estimated need at {fmtInt(ctx)} tokens: <b>{fmtBytes(est.total_bytes)}</b> (weights {fmtBytes(est.selected_weights.bytes)} + KV cache + buffers).{" "}
                    {sys?.memory.available_bytes != null && (
                      <>
                        The OS reports <b>{fmtBytes(sys.memory.available_bytes)}</b> of RAM available now (measured).
                      </>
                    )}
                  </EstimateBanner>
                )}
                {rt?.state === "ready" && rt.scan_id !== scan.id && <p className="small dim">Starting will unload {rt.model_name} first.</p>}
                {memBlock && (
                  <Callout kind="warn" title="Probably too large for available memory.">
                    {memBlock}
                    <div className="row" style={{ marginTop: 8 }}>
                      <button className="btn sm" onClick={() => start(true)} disabled={starting}>
                        Start anyway
                      </button>
                    </div>
                  </Callout>
                )}
                {rt?.state === "failed" && rt.error && <Callout kind="bad" title="Last start failed:">{rt.error}</Callout>}
                <div className="row">
                  <button className="btn primary" onClick={() => start(false)} disabled={starting || !scan.integrity.valid}>
                    {starting ? <Spinner /> : "Load model"}
                  </button>
                  {!scan.integrity.valid && <span className="small faint">integrity errors — this file will not be loaded</span>}
                </div>
              </div>
            )}
            {loadedHere && rt && (
              <div className="col">
                {rt.state === "starting" ? (
                  <div className="row">
                    <Spinner /> <span className="dim">loading {rt.model_name}…</span>
                  </div>
                ) : (
                  <div className="kv-grid">
                    <span>Context (runtime)</span>
                    <b>{fmtInt(rt.n_ctx_runtime ?? rt.context_length)} tokens</b>
                    <span>GPU layers</span>
                    <b>{rt.gpu_layers_offloaded !== null ? `${rt.gpu_layers_offloaded} / ${rt.total_layers}` : "—"}</b>
                    <span>Load time</span>
                    <b>
                      {rt.load_seconds !== null ? `${rt.load_seconds.toFixed(2)} s` : "—"} <span className="badge prov verified">measured</span>
                    </b>
                    <span>Process resident memory</span>
                    <b>
                      {fmtBytes(rt.process_rss_bytes)} <span className="badge prov verified">OS</span>
                    </b>
                  </div>
                )}
                {rt.allocations.length > 0 && (
                  <details open>
                    <summary className="small">Buffers allocated by llama.cpp ({fmtBytes(rt.allocations.reduce((a, x) => a + x.bytes, 0))})</summary>
                    <table className="small" style={{ marginTop: 6 }}>
                      <tbody>
                        {rt.allocations.map((a, i) => (
                          <tr key={i}>
                            <td>{a.kind}</td>
                            <td className="mono">{a.device}</td>
                            <td className="num">{fmtBytes(a.bytes)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="small faint">Reported by llama.cpp in its load log. Compare with the estimates on the Performance page.</p>
                  </details>
                )}
                <div className="row">
                  <button className="btn" onClick={stop}>
                    Unload model
                  </button>
                  <button className="btn ghost sm" onClick={() => onNavigate("memory")}>
                    Compare with estimates →
                  </button>
                </div>
              </div>
            )}
            {rt && rt.log_tail.length > 0 && (rt.state === "starting" || rt.state === "failed") && (
              <details style={{ marginTop: 10 }}>
                <summary className="small">llama-server log</summary>
                <pre className="log mono small">{rt.log_tail.slice(-40).join("\n")}</pre>
              </details>
            )}
          </div>

          <div className="card">
            <div className="card-head">
              <h3>Prompt</h3>
              <label className="check small">
                <input type="checkbox" checked={chat} onChange={(e) => setChat(e.target.checked)} /> use the model's chat template
              </label>
            </div>
            <div className="col">
              {chat && (
                <label className="field">
                  System message (optional)
                  <input type="text" value={system} onChange={(e) => setSystem(e.target.value)} />
                </label>
              )}
              <label className="field">
                {chat ? "User message" : "Raw prompt"}
                <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
              </label>
              <div className="grid grid-3">
                <label className="field">
                  Max new tokens
                  <input type="number" min={1} max={4096} value={nPredict} onChange={(e) => setNPredict(Math.min(4096, Math.max(1, Number(e.target.value) || 1)))} />
                </label>
                <label className="field">
                  Temperature
                  <input type="number" min={0} max={2} step={0.05} value={temperature} onChange={(e) => setTemperature(Math.min(2, Math.max(0, Number(e.target.value))))} />
                </label>
                <label className="field">
                  Seed
                  <input type="number" min={0} placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value)} />
                </label>
                <label className="field">
                  Top-k
                  <input type="number" min={0} max={1000} value={topK} onChange={(e) => setTopK(Math.max(0, Number(e.target.value) || 0))} />
                </label>
                <label className="field">
                  Top-p
                  <input type="number" min={0.01} max={1} step={0.01} value={topP} onChange={(e) => setTopP(Math.min(1, Math.max(0.01, Number(e.target.value) || 1)))} />
                </label>
                <label className="field">
                  Alternatives shown
                  <input type="number" min={0} max={10} value={nProbs} onChange={(e) => setNProbs(Math.min(10, Math.max(0, Number(e.target.value) || 0)))} />
                </label>
              </div>
              <div className="row">
                {generating ? (
                  <button className="btn" onClick={stopGenerating}>
                    ■ Stop
                  </button>
                ) : (
                  <button className="btn primary" onClick={generate} disabled={!ready || !prompt.trim()}>
                    Generate
                  </button>
                )}
                {!ready && <span className="small faint">load the model first</span>}
              </div>
            </div>
          </div>
        </div>
      )}

      {(promptEv || tokens.length > 0 || done) && (
        <div className="card">
          <div className="card-head">
            <h3>Output</h3>
            <label className="check small">
              <input type="checkbox" checked={boundaries} onChange={(e) => setBoundaries(e.target.checked)} /> show token boundaries
            </label>
          </div>
          {notices.map((n, i) => (
            <Callout key={i} kind="warn">
              {n}
            </Callout>
          ))}
          <div className={`gen-out ${boundaries ? "bounded" : ""}`} aria-live="polite">
            {tokens.map((t, i) => (
              <button
                key={i}
                className={`tok ${picked === i ? "picked" : ""}`}
                onClick={() => setPicked(picked === i ? null : i)}
                title={t.prob ? `id ${t.ids.join(",")} · p=${pct(t.prob.logprob)}` : `id ${t.ids.join(",")}`}
                style={t.prob && boundaries ? { background: `color-mix(in srgb, var(--accent) ${Math.round(Math.max(0, 1 - Math.exp(t.prob.logprob)) * 35)}%, transparent)` } : undefined}
              >
                {boundaries ? visible(t.content) : t.content}
              </button>
            ))}
            {generating && <span className="cursor" aria-hidden />}
          </div>
          {boundaries && <p className="small faint">Shading = how unlikely the sampled token was (1 − p). Click a token for its id and the runtime's top alternatives.</p>}
          {picked !== null && tokens[picked] && <TokenDetail t={tokens[picked]} index={picked} />}

          <Stats tokens={tokens} gaps={gaps} done={done} stopped={stopped} generating={generating} />

          {promptEv && (
            <details>
              <summary className="small">
                Prompt as evaluated: {fmtInt(promptEv.n_tokens)} tokens{promptEv.chat_template_applied ? " (after the chat template)" : ""}
              </summary>
              <div className="prompt-toks">
                {promptEv.tokens.map((t, i) => (
                  <span key={i} className="ptok" title={`id ${t.id}`}>
                    <span className="mono">{t.piece === null ? "?" : visible(t.piece)}</span>
                    <small>{t.id}</small>
                  </span>
                ))}
                {promptEv.n_tokens > promptEv.tokens.length && <span className="small faint">… {promptEv.n_tokens - promptEv.tokens.length} more</span>}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function TokenDetail({ t, index }: { t: OutToken; index: number }) {
  return (
    <div className="token-detail">
      <div className="small">
        Token #{index + 1} · id <b className="mono">{t.ids.join(", ") || "—"}</b> · arrived at {t.t_ms.toFixed(0)} ms (wall clock)
      </div>
      {t.prob ? (
        <table className="small">
          <thead>
            <tr>
              <th>Candidate</th>
              <th className="num">id</th>
              <th className="num">probability</th>
              <th className="num">log-prob</th>
            </tr>
          </thead>
          <tbody>
            {t.prob.top.map((a) => (
              <tr key={a.id} className={a.id === t.prob!.id ? "diff" : ""}>
                <td className="mono">{JSON.stringify(a.token)}</td>
                <td className="num mono">{a.id}</td>
                <td className="num">{pct(a.logprob)}</td>
                <td className="num mono">{a.logprob.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="small faint">No probabilities were requested for this run.</p>
      )}
      <p className="small faint">Probabilities are the runtime's, before sampling (llama-server n_probs).</p>
    </div>
  );
}

function Stats({ tokens, gaps, done, stopped, generating }: { tokens: OutToken[]; gaps: number[]; done: Done | null; stopped: string | null; generating: boolean }) {
  const t = done?.timings ?? {};
  const max = Math.max(1, ...gaps.slice(1));
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="grid grid-4">
        <div className="stat">
          <span className="stat-label">Generated tokens</span>
          <span className="stat-value">{done?.tokens_predicted ?? tokens.length}</span>
          <span className="stat-sub">{done ? `stop: ${done.stop_type}` : stopped ?? (generating ? "streaming…" : "")}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Decode speed</span>
          <span className="stat-value">{t.predicted_per_second != null ? `${t.predicted_per_second.toFixed(1)} tok/s` : "—"}</span>
          <span className="stat-sub">measured by llama.cpp</span>
        </div>
        <div className="stat">
          <span className="stat-label">Prompt processing</span>
          <span className="stat-value">{t.prompt_per_second != null ? `${t.prompt_per_second.toFixed(0)} tok/s` : "—"}</span>
          <span className="stat-sub">{t.prompt_n != null ? `${t.prompt_n} tokens in ${t.prompt_ms?.toFixed(0)} ms · llama.cpp` : "measured by llama.cpp"}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Time to first token</span>
          <span className="stat-value">{tokens[0] ? `${tokens[0].t_ms.toFixed(0)} ms` : "—"}</span>
          <span className="stat-sub">wall clock in Model Surgery (incl. tokenizing & HTTP)</span>
        </div>
      </div>
      {gaps.length > 2 && (
        <div>
          <div className="small dim">Inter-token latency, wall clock (each bar = one token)</div>
          <svg className="spark" viewBox={`0 0 ${gaps.length - 1} 30`} preserveAspectRatio="none" role="img" aria-label="inter-token latency">
            {gaps.slice(1).map((g, i) => (
              <rect key={i} x={i + 0.1} width={0.8} y={30 - (g / max) * 30} height={(g / max) * 30} />
            ))}
          </svg>
          <div className="small faint">max {max.toFixed(0)} ms · includes network and browser scheduling; use the llama.cpp figures above for throughput.</div>
        </div>
      )}
      {done?.truncated && <Callout kind="warn">The runtime truncated the prompt to fit the context.</Callout>}
    </div>
  );
}

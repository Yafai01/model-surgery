import { useCallback, useEffect, useRef, useState } from "react";
import type { Tab } from "../App";
import { BROWSER_EDITION, api } from "../api";
import { fmtBytes, fmtTime } from "../format";
import type { Job, Plan, ScanResult, ToolDryRun, ToolsReport } from "../types";
import { Callout, List, Spinner } from "./common";

// Mirrors backend jobs.IMATRIX_REQUIRED: llama-quantize refuses these without an importance matrix.
const IMATRIX_REQUIRED = new Set(["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K_S"]);
const ACTIVE = new Set(["running", "validating"]);

interface Props {
  scan: ScanResult;
  onError: (m: string) => void;
  onOpenScan: (id: string, tab?: Tab) => void;
  onHistoryChanged: () => void;
}

export default function QuantizePage(props: Props) {
  return BROWSER_EDITION ? <LocalAppOnly scan={props.scan} onOpenScan={props.onOpenScan} /> : <QuantizeWorkflow {...props} />;
}

function LocalAppOnly({ scan, onOpenScan }: Pick<Props, "scan" | "onOpenScan">) {
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Quantize / convert</h2>
          <p>Quantization runs llama.cpp on your own machine, which a website cannot (and should not) do.</p>
        </div>
      </div>
      <Callout kind="warn" title="Available in the local app.">
        The hosted edition inspects models and estimates memory entirely in your browser. To quantize {scan.kind === "gguf" ? "this GGUF" : "or convert this model"}{" "}
        with a dry-run plan, explicit confirmation and output validation, run Model Surgery locally:
      </Callout>
      <div className="card">
        <pre className="argv mono" style={{ whiteSpace: "pre-wrap" }}>
          {`git clone https://github.com/Yafai01/model-surgery
cd model-surgery
.\\scripts\\setup.ps1          # or ./scripts/setup.sh on Linux / Jetson
.\\.venv\\Scripts\\model-surgery --llama-cpp-dir <folder with llama-quantize> serve --open`}
        </pre>
        <div className="row" style={{ marginTop: 12 }}>
          <a className="btn primary" href="https://github.com/Yafai01/model-surgery#quick-start-windows-11" target="_blank" rel="noreferrer">
            Setup guide on GitHub
          </a>
          <button className="btn" onClick={() => onOpenScan(scan.id, "memory")}>
            See size estimates per precision →
          </button>
        </div>
      </div>
    </div>
  );
}

function QuantizeWorkflow({ scan, onError, onOpenScan, onHistoryChanged }: Props) {
  const operation = scan.kind === "gguf" ? "quantize" : scan.kind === "hf_safetensors" ? "convert" : null;
  const [tools, setTools] = useState<ToolsReport | null>(null);
  const [checking, setChecking] = useState(false);
  const [target, setTarget] = useState("");
  const [allowRequant, setAllowRequant] = useState(false);
  const [outputDir, setOutputDir] = useState("");
  const [threads, setThreads] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [dry, setDry] = useState<ToolDryRun | null>(null);
  const [dryBusy, setDryBusy] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const logRef = useRef<HTMLPreElement>(null);

  const loadTools = useCallback(
    async (refresh = false) => {
      setChecking(true);
      try {
        setTools(await api.tools(refresh));
      } catch (e) {
        onError((e as Error).message);
      } finally {
        setChecking(false);
      }
    },
    [onError],
  );

  const loadJobs = useCallback(async () => {
    try {
      const js = await api.jobs();
      setJobs(js);
      return js;
    } catch (e) {
      onError((e as Error).message);
      return [];
    }
  }, [onError]);

  useEffect(() => {
    loadTools();
    loadJobs().then((js) => {
      const running = js.find((j) => ACTIVE.has(j.status));
      if (running) api.job(running.id).then(setJob).catch(() => undefined);
    });
  }, [loadTools, loadJobs]);

  // default target
  useEffect(() => {
    if (!tools || target) return;
    if (operation === "quantize") {
      const names = tools.quantize.types.map((t) => t.name);
      setTarget(names.includes("Q4_K_M") ? "Q4_K_M" : names[0] ?? "");
    } else if (operation === "convert") {
      setTarget(tools.convert.outtypes.includes("f16") ? "f16" : tools.convert.outtypes[0] ?? "");
    }
  }, [tools, operation, target]);

  // any change to inputs invalidates the plan and the confirmation
  useEffect(() => {
    setPlan(null);
    setDry(null);
    setConfirmed(false);
  }, [target, allowRequant, outputDir, threads, scan.id]);

  // poll the active job
  useEffect(() => {
    if (!job || !ACTIVE.has(job.status)) return;
    const t = setInterval(async () => {
      try {
        const j = await api.job(job.id);
        setJob(j);
        if (!ACTIVE.has(j.status)) {
          loadJobs();
          onHistoryChanged();
        }
      } catch (e) {
        onError((e as Error).message);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [job, loadJobs, onError, onHistoryChanged]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log_tail]);

  if (!operation) {
    return (
      <div className="page">
        <h2>Quantize / convert</h2>
        <Callout kind="warn">
          Single safetensors files without a config.json cannot be converted. Scan the full Hugging Face model directory instead.
        </Callout>
      </div>
    );
  }

  const makePlan = async () => {
    setPlanning(true);
    setDry(null);
    setConfirmed(false);
    try {
      setPlan(
        await api.plan({
          scan_id: scan.id,
          operation,
          target,
          output_dir: outputDir.trim() || null,
          allow_requantize: allowRequant,
          threads: threads ? Number(threads) : null,
        }),
      );
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setPlanning(false);
    }
  };

  const runToolDry = async () => {
    if (!plan) return;
    setDryBusy(true);
    try {
      setDry(await api.toolDryRun(plan.plan_id));
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setDryBusy(false);
    }
  };

  const start = async () => {
    if (!plan?.confirm_token) return;
    try {
      const j = await api.execute(plan.plan_id, plan.confirm_token);
      setJob(j);
      setPlan(null);
      setConfirmed(false);
      loadJobs();
    } catch (e) {
      onError((e as Error).message);
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      setJob(await api.cancel(job.id));
    } catch (e) {
      onError((e as Error).message);
    }
  };

  const tool = operation === "quantize" ? tools?.quantize : tools?.convert;
  const available = !!tool?.available;
  const srcQuantized = scan.quantization.is_quantized;
  const busyJob = job && ACTIVE.has(job.status);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>{operation === "quantize" ? "Quantize GGUF" : "Convert Hugging Face → GGUF"}</h2>
          <p>
            Dry-run plan → explicit confirmation → fixed-argument subprocess → output validation. The source is never modified and existing files are never
            overwritten.
          </p>
        </div>
      </div>

      {/* ---- tooling ---- */}
      <div className="card">
        <div className="card-head">
          <h3>Detected tooling</h3>
          <button className="btn sm" onClick={() => loadTools(true)} disabled={checking}>
            {checking ? <Spinner /> : "Re-detect"}
          </button>
        </div>
        {!tools ? (
          <Spinner />
        ) : (
          <div className="grid grid-2">
            <ToolCard
              title="llama-quantize"
              ok={tools.quantize.available}
              lines={[
                tools.quantize.path ?? "",
                tools.quantize.version ? `llama.cpp ${tools.quantize.version}` : "",
                tools.quantize.available ? `${tools.quantize.types.length} quantization types · --dry-run ${tools.quantize.supports_dry_run ? "supported" : "not supported"}` : "",
                tools.quantize.detail ?? "",
              ]}
            />
            <ToolCard
              title="convert_hf_to_gguf.py"
              ok={tools.convert.available}
              lines={[
                tools.convert.script ?? "",
                tools.convert.python ? `python: ${tools.convert.python}` : "",
                tools.convert.available ? `outtypes: ${tools.convert.outtypes.join(", ")}` : "",
                tools.convert.supported_architectures ? `${tools.convert.supported_architectures.length} supported model classes` : "",
                tools.convert.detail ?? "",
              ]}
            />
          </div>
        )}
        {tools && !available && (
          <div style={{ marginTop: 12 }}>
            <Callout kind="warn" title="Tool not available.">
              <pre className="mono small" style={{ margin: "6px 0 0", whiteSpace: "pre-wrap" }}>
                {tools.configuration_help.join("\n")}
              </pre>
            </Callout>
          </div>
        )}
      </div>

      {/* ---- configure ---- */}
      {available && !busyJob && (
        <div className="card">
          <div className="card-head">
            <h3>1 · Configure</h3>
            <span className="small faint">
              source: {scan.quantization.label ?? scan.kind} · {fmtBytes(scan.total_size_bytes)}
            </span>
          </div>
          <div className="grid grid-2">
            <label className="field">
              {operation === "quantize" ? "Target quantization (offered by your llama-quantize build)" : "Output type (offered by your converter)"}
              <select value={target} onChange={(e) => setTarget(e.target.value)}>
                {operation === "quantize"
                  ? tools!.quantize.types.map((t) => {
                      const blocked = IMATRIX_REQUIRED.has(t.name);
                      const same = (scan.quantization.label ?? "").split(" ")[0] === t.name;
                      return (
                        <option key={t.name} value={t.name} disabled={blocked || same}>
                          {t.name} — {blocked ? "needs an importance matrix (not supported here)" : same ? "same as source" : t.description}
                        </option>
                      );
                    })
                  : tools!.convert.outtypes.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
              </select>
            </label>
            <label className="field">
              Output directory (optional — must differ from the source directory)
              <input type="text" className="mono" placeholder="default: Model Surgery outputs folder" value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
            </label>
            {operation === "quantize" && (
              <label className="field">
                Threads (optional)
                <input type="number" min={1} max={256} value={threads} onChange={(e) => setThreads(e.target.value)} placeholder="tool default" />
              </label>
            )}
            {operation === "quantize" && srcQuantized && (
              <label className="check" style={{ alignSelf: "end" }}>
                <input type="checkbox" checked={allowRequant} onChange={(e) => setAllowRequant(e.target.checked)} />
                <span>
                  Allow requantizing an already-quantized model
                  <div className="small faint">Quality is worse than quantizing from F16/BF16. Passes --allow-requantize.</div>
                </span>
              </label>
            )}
          </div>
          <div className="row" style={{ marginTop: 14 }}>
            <button className="btn primary" onClick={makePlan} disabled={!target || planning}>
              {planning ? <Spinner /> : "Create dry-run plan"}
            </button>
            <span className="small faint">Nothing is executed at this step.</span>
          </div>
        </div>
      )}

      {/* ---- plan ---- */}
      {plan && !busyJob && <PlanView plan={plan} dry={dry} dryBusy={dryBusy} onToolDry={runToolDry} confirmed={confirmed} setConfirmed={setConfirmed} onStart={start} />}

      {/* ---- job ---- */}
      {job && (
        <div className="card">
          <div className="card-head">
            <h3>{busyJob ? "3 · Running" : "Result"}</h3>
            <div className="row">
              <StatusBadge status={job.status} />
              {job.status === "running" && (
                <button className="btn sm danger" onClick={cancel}>
                  Cancel
                </button>
              )}
            </div>
          </div>
          <div className="col" style={{ gap: 10 }}>
            <div className="mono small dim break">{job.plan.argv.join(" ")}</div>
            {busyJob && (
              <>
                <div className={`progress ${job.progress === null ? "indeterminate" : ""}`}>
                  <div style={{ width: `${(job.progress ?? 0) * 100}%` }} />
                </div>
                <div className="small dim">
                  {job.progress !== null ? `${Math.round(job.progress * 100)}% · ` : ""}
                  {job.progress_label ?? "starting…"}
                </div>
              </>
            )}
            {job.log_tail.length > 0 && (
              <pre className="log mono" ref={logRef}>
                {job.log_tail.join("\n")}
              </pre>
            )}
            {job.validation && (
              <div>
                <h4 style={{ marginBottom: 8 }}>Output validation</h4>
                <ul className="checklist">
                  {job.validation.checks.map((c) => (
                    <li key={c.name}>
                      <span className="check-icon" style={{ color: c.passed ? "var(--ok)" : "var(--bad)" }}>
                        {c.passed ? "✓" : "✗"}
                      </span>
                      <span>
                        <strong>{c.name}</strong> <span className="dim small mono">{c.detail}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {job.status === "succeeded" && (
              <Callout kind="ok" title="Succeeded and validated.">
                Wrote <span className="mono">{job.output_path}</span> ({fmtBytes(job.output_size_bytes)}). The output was re-parsed and checked against the source
                before being reported as successful.
                <div className="row" style={{ marginTop: 8 }}>
                  {job.output_scan_id && (
                    <>
                      <button className="btn sm" onClick={() => onOpenScan(job.output_scan_id!, "overview")}>
                        Open output scan
                      </button>
                      <button className="btn sm" onClick={() => onOpenScan(job.output_scan_id!, "memory")}>
                        Memory estimate for output
                      </button>
                    </>
                  )}
                </div>
              </Callout>
            )}
            {job.error && (
              <Callout kind={job.status === "cancelled" ? "warn" : "bad"} title={job.status === "cancelled" ? "Cancelled." : "Failed — no output was kept."}>
                {job.error}
              </Callout>
            )}
            {!busyJob && (
              <div>
                <button className="btn sm" onClick={() => setJob(null)}>
                  Close
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---- history ---- */}
      {jobs.length > 0 && (
        <div className="card">
          <div className="card-head">
            <h3>Job history</h3>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Operation</th>
                  <th>Source</th>
                  <th>Output</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} onClick={() => api.job(j.id).then(setJob).catch(() => undefined)} style={{ cursor: "pointer" }}>
                    <td className="nowrap small">{fmtTime(j.created_at)}</td>
                    <td className="nowrap">
                      {j.plan.operation} → <span className="mono">{j.plan.target}</span>
                    </td>
                    <td className="mono small break">{j.plan.source_path}</td>
                    <td className="mono small break">
                      {j.output_path ?? "—"}
                      {j.output_size_bytes ? <div className="faint">{fmtBytes(j.output_size_bytes)}</div> : null}
                    </td>
                    <td>
                      <StatusBadge status={j.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function ToolCard({ title, ok, lines }: { title: string; ok: boolean; lines: string[] }) {
  return (
    <div className="col" style={{ gap: 4, padding: 12, border: "1px solid var(--border)", borderRadius: 8 }}>
      <div className="row">
        <strong className="mono">{title}</strong>
        <span className={`badge ${ok ? "ok" : "warn"}`}>{ok ? "available" : "not available"}</span>
      </div>
      {lines.filter(Boolean).map((l, i) => (
        <div key={i} className="small dim mono break" style={{ whiteSpace: "pre-wrap" }}>
          {l}
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ status }: { status: Job["status"] }) {
  const cls = status === "succeeded" ? "ok" : status === "failed" || status === "interrupted" ? "bad" : status === "cancelled" ? "warn" : "info";
  return <span className={`badge ${cls}`}>{status}</span>;
}

function PlanView({
  plan,
  dry,
  dryBusy,
  onToolDry,
  confirmed,
  setConfirmed,
  onStart,
}: {
  plan: Plan;
  dry: ToolDryRun | null;
  dryBusy: boolean;
  onToolDry: () => void;
  confirmed: boolean;
  setConfirmed: (b: boolean) => void;
  onStart: () => void;
}) {
  const toolName = plan.operation === "quantize" ? "llama-quantize" : "convert_hf_to_gguf.py";
  return (
    <div className="card">
      <div className="card-head">
        <h3>2 · Dry-run summary</h3>
        <span className={`badge ${plan.can_execute ? "ok" : "bad"}`}>{plan.can_execute ? "ready" : "blocked"}</span>
      </div>
      <div className="grid grid-2">
        <dl className="kv">
          <dt>Source</dt>
          <dd className="mono small">{plan.source_path}</dd>
          <dt>Source format</dt>
          <dd>{plan.source_label ?? "—"}</dd>
          <dt>Target</dt>
          <dd>
            <strong>{plan.target}</strong>
          </dd>
          <dt>Output file</dt>
          <dd className="mono small">{plan.output_path ?? "—"}</dd>
          <dt>Written as</dt>
          <dd className="mono small faint">{plan.partial_path ?? "—"} (renamed only after validation)</dd>
          <dt>Tool</dt>
          <dd className="mono small">
            {plan.tool ?? "—"} {plan.tool_version && <span className="faint">({plan.tool_version})</span>}
          </dd>
        </dl>
        <dl className="kv">
          <dt>Estimated output</dt>
          <dd>
            <strong>{fmtBytes(plan.estimated_output_bytes)}</strong>
            <div className="small faint">{plan.estimate_note}</div>
          </dd>
          <dt>Free disk space</dt>
          <dd>{fmtBytes(plan.free_disk_bytes)}</dd>
          {dry && (
            <>
              <dt>Tool dry-run</dt>
              <dd>
                {dry.quant_size_mib !== null ? (
                  <>
                    <strong>{dry.quant_size_mib.toFixed(2)} MiB</strong> at {dry.quant_bpw} BPW (reported by llama-quantize --dry-run)
                    <div className="small faint">tensor data only; the file also contains the metadata header</div>
                    {dry.fallback_tensors > 0 && <div className="small faint">{dry.fallback_tensors} tensor(s) need fallback types</div>}
                  </>
                ) : (
                  <span className="dim">no size reported (exit {String(dry.exit_code)}): {dry.tail.slice(-2).join(" ")}</span>
                )}
              </dd>
            </>
          )}
        </dl>
      </div>
      <h4 style={{ margin: "14px 0 6px" }}>Exact command (argument list, no shell)</h4>
      <pre className="argv mono">
        {plan.argv.map((a, i) => (
          <span key={i}>{a}</span>
        ))}
      </pre>
      {plan.blockers.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <Callout kind="bad" title="Blocked:">
            <List items={plan.blockers} />
          </Callout>
        </div>
      )}
      {plan.warnings.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <Callout kind="warn" title="Warnings:">
            <List items={plan.warnings} />
          </Callout>
        </div>
      )}
      {plan.can_execute && (
        <div className="col" style={{ marginTop: 16, gap: 12 }}>
          {plan.supports_tool_dry_run && (
            <div className="row">
              <button className="btn sm" onClick={onToolDry} disabled={dryBusy}>
                {dryBusy ? <Spinner /> : "Ask llama-quantize for its own --dry-run size"}
              </button>
              <span className="small faint">reads the source; writes nothing</span>
            </div>
          )}
          <label className="check">
            <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
            <span>
              I confirm: run <strong className="mono">{toolName}</strong> with the arguments above and write about{" "}
              <strong>{fmtBytes(plan.estimated_output_bytes)}</strong> into <span className="mono">{plan.output_dir}</span>. This is CPU- and disk-intensive and may
              take several minutes.
            </span>
          </label>
          <div className="row">
            <button className="btn primary" disabled={!confirmed} onClick={onStart}>
              Start {plan.operation}
            </button>
            <span className="small faint">plan expires {new Date(plan.expires_at * 1000).toLocaleTimeString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}

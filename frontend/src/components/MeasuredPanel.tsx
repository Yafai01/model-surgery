import { useEffect, useState } from "react";
import { BROWSER_EDITION, api } from "../api";
import { fmtBytes, fmtInt, fmtTime } from "../format";
import type { Measurement, MemoryEstimate, RuntimeStatus, ScanResult } from "../types";

/**
 * Measurements for this model on this machine, next to Model Surgery's estimates for the same settings.
 * Estimates come from the scan; "reported" values come from llama.cpp's load log; "OS" values from the operating system.
 */
export default function MeasuredPanel({ scan }: { scan: ScanResult }) {
  const [rt, setRt] = useState<RuntimeStatus | null>(null);
  const [runs, setRuns] = useState<Measurement[]>([]);
  const [est, setEst] = useState<MemoryEstimate | null>(null);

  useEffect(() => {
    if (BROWSER_EDITION) return;
    let cancelled = false;
    Promise.all([api.runtime(), api.runtimeMeasurements()])
      .then(([r, m]) => {
        if (cancelled) return;
        setRt(r);
        setRuns(m.filter((x) => x.scan_id === scan.id));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [scan.id]);

  const loaded = rt?.state === "ready" && rt.scan_id === scan.id;
  const nCtx = rt?.n_ctx_runtime ?? rt?.context_length ?? null;
  useEffect(() => {
    if (!loaded || !nCtx) return;
    let cancelled = false;
    api
      .memory(scan.id, { weight_format: "current", context_length: nCtx, batch_size: 1, kv_cache_type: "f16", flash_attention: true, n_ubatch: 512, device_id: null, custom_device: null })
      .then((m) => !cancelled && setEst(m))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [loaded, nCtx, scan.id]);

  if (BROWSER_EDITION)
    return (
      <div className="card">
        <h3>Measured on this machine</h3>
        <p className="small dim">The browser edition cannot run models, so it has no measurements. Everything on this page is an estimate.</p>
      </div>
    );

  const sum = (pred: (k: string) => boolean) => (rt?.allocations ?? []).filter((a) => pred(a.kind)).reduce((a, x) => a + x.bytes, 0);
  const estOf = (key: string) => est?.inference.find((c) => c.key === key)?.bytes ?? null;
  const modelBufs = (rt?.allocations ?? []).filter((a) => a.kind === "model");
  // A "*_Mapped" buffer is a memory-mapped file range; with partial GPU offload it can span tensors that were also
  // copied to the GPU, so adding it to the device buffers would double count. Show the split instead of a %.
  const mappedOverlap = modelBufs.some((a) => /mapped/i.test(a.device)) && modelBufs.some((a) => !/mapped|^cpu$/i.test(a.device));
  const rows: [label: string, est: number | null, reported: number | null, detail?: string][] = loaded
    ? [
        [
          "Weights",
          estOf("weights"),
          sum((k) => k === "model"),
          mappedOverlap ? modelBufs.map((a) => `${a.device} ${fmtBytes(a.bytes)}`).join(" + ") : undefined,
        ],
        ["KV cache", estOf("kv_cache"), sum((k) => k === "KV" || k === "RS")],
        ["Compute + output buffers", estOf("compute"), sum((k) => k === "compute" || k === "output")],
      ]
    : [];
  const swa = scan.hyperparameters.find((f) => f.key === "sliding_window")?.value;

  return (
    <div className="card">
      <div className="card-head">
        <h3>Measured on this machine</h3>
        {loaded && <span className="badge ok">runtime loaded · {fmtInt(nCtx)} ctx</span>}
      </div>
      {loaded ? (
        <>
          <table className="small">
            <thead>
              <tr>
                <th />
                <th className="num">Estimate (Model Surgery)</th>
                <th className="num">Allocated (reported by llama.cpp)</th>
                <th className="num">Difference</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, e, r, detail]) => (
                <tr key={label}>
                  <td>{label}</td>
                  <td className="num">{fmtBytes(e)}</td>
                  <td className="num">{detail ?? (r ? fmtBytes(r) : "—")}</td>
                  <td className="num">
                    {detail ? <span className="faint">mapped range overlaps GPU copies</span> : e && r ? `${r >= e ? "+" : "−"}${fmtBytes(Math.abs(r - e))} (${(((r - e) / e) * 100).toFixed(0)}%)` : "—"}
                  </td>
                </tr>
              ))}
              <tr>
                <td>Process resident memory</td>
                <td className="num">{fmtBytes(est?.total_bytes)} total</td>
                <td className="num">
                  {fmtBytes(rt?.process_rss_bytes)} <span className="badge prov verified">OS</span>
                </td>
                <td className="num faint">not comparable 1:1</td>
              </tr>
            </tbody>
          </table>
          <ul className="small dim" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
            <li>Allocations come from llama.cpp's own load log, per device ({[...new Set(rt!.allocations.map((a) => a.device))].join(", ")}). GPU buffers are included; GPU utilisation is not measured.</li>
            {typeof swa === "number" && <li>This model uses sliding-window attention ({swa} tokens): llama.cpp keeps a smaller cache for those layers, so the estimate (full-context KV for every layer) is an upper bound.</li>}
            <li>Resident memory counts the weight pages the OS currently has mapped; with mmap it can be lower than the weight size until every page has been touched.</li>
            {mappedOverlap && <li>With partial GPU offload, llama.cpp reports the memory-mapped file range (*_Mapped) separately from the GPU copies; the two overlap, so they are not added up or compared with the file size.</li>}
            <li>The compute-buffer estimate is a heuristic. GPU backends reserve larger graph buffers (llama.cpp sizes them for the worst-case batch), so a large difference there is expected.</li>
          </ul>
        </>
      ) : (
        <p className="small dim">
          No runtime is loaded for this model. Load it on the Inference page to see the buffers llama.cpp actually allocates next to these estimates.
        </p>
      )}

      <h4 style={{ marginTop: 14 }}>Generation runs</h4>
      {runs.length === 0 ? (
        <p className="small dim">No measured runs for this model in this session. Generate text on the Inference page to record prompt and decode throughput.</p>
      ) : (
        <div className="table-wrap">
          <table className="small">
            <thead>
              <tr>
                <th>When</th>
                <th className="num">ctx</th>
                <th className="num">GPU layers</th>
                <th className="num">Prompt</th>
                <th className="num">Prompt tok/s</th>
                <th className="num">Generated</th>
                <th className="num">Decode tok/s</th>
                <th className="num">First token</th>
                <th className="num">Resident</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((m, i) => (
                <tr key={i}>
                  <td className="nowrap">{fmtTime(m.at)}</td>
                  <td className="num">{fmtInt(m.n_ctx)}</td>
                  <td className="num">{m.gpu_layers_offloaded ?? "—"}</td>
                  <td className="num">{fmtInt(m.prompt_tokens)}</td>
                  <td className="num">{m.prompt_tokens_per_s?.toFixed(1) ?? "—"}</td>
                  <td className="num">{fmtInt(m.generated_tokens)}</td>
                  <td className="num">
                    <b>{m.decode_tokens_per_s?.toFixed(1) ?? "—"}</b>
                  </td>
                  <td className="num">{m.wall_first_token_ms != null ? `${m.wall_first_token_ms.toFixed(0)} ms` : "—"}</td>
                  <td className="num">{fmtBytes(m.process_rss_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small faint">Throughput: llama.cpp timings. First token: wall clock in Model Surgery. Resident memory: OS reading after the run. Kept in memory for this server session.</p>
        </div>
      )}
    </div>
  );
}

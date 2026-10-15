import { type ChangeEvent, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { fmtBytes, fmtInt } from "../format";
import type { DeviceProfile, FitResult, KvType, MemoryEstimate, MemoryRequest, ScanResult } from "../types";
import { Callout, EstimateBanner, List, Prov, Spinner } from "./common";

const CTX_PRESETS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const GiB = 1024 ** 3;
const MiB = 1024 ** 2;
const SEG_COLORS: Record<string, string> = { weights: "#2dd4bf", kv_cache: "#38bdf8", compute: "#a78bfa", runtime: "#94a3b8" };
const VERDICT: Record<FitResult["verdict"], string> = {
  fits: "Fits",
  tight: "Tight fit",
  partial_offload: "Partial GPU offload",
  does_not_fit: "Does not fit",
};

const DEFAULT_CUSTOM: DeviceProfile = {
  id: "custom",
  name: "Custom device",
  memory_kind: "unified",
  capacity_bytes: 16 * GiB,
  reserved_bytes: 2 * GiB,
  runtime_overhead_bytes: 300 * MiB,
  host_ram_bytes: null,
  bandwidth_gbps: 100,
  assumptions: ["user-defined profile"],
};

export default function MemoryLab({ scan, onError }: { scan: ScanResult; onError: (m: string) => void }) {
  const trainCtx = scan.hyperparameters.find((f) => f.key === "context_length")?.value as number | undefined;
  const [devices, setDevices] = useState<DeviceProfile[]>([]);
  const [custom, setCustom] = useState<DeviceProfile>(DEFAULT_CUSTOM);
  const [req, setReq] = useState<MemoryRequest>({
    weight_format: "current",
    context_length: Math.min(trainCtx ?? 4096, 4096),
    batch_size: 1,
    kv_cache_type: "f16",
    flash_attention: true,
    n_ubatch: 512,
    device_id: "jetson-orin-nano-8gb",
    custom_device: null,
  });
  const [est, setEst] = useState<MemoryEstimate | null>(null);
  const [allFits, setAllFits] = useState<FitResult[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.devices().then(setDevices).catch((e) => onError(e.message));
  }, [onError]);

  const effective: MemoryRequest = useMemo(
    () => (req.device_id === "custom" ? { ...req, device_id: null, custom_device: custom } : { ...req, custom_device: null }),
    [req, custom],
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const t = setTimeout(async () => {
      try {
        const main = await api.memory(scan.id, effective);
        if (cancelled) return;
        setEst(main);
        const others = await Promise.all(devices.map((d) => api.memory(scan.id, { ...effective, device_id: d.id, custom_device: null })));
        if (!cancelled) setAllFits(others.map((o) => o.fit!).filter(Boolean));
      } catch (e) {
        if (!cancelled) onError((e as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [scan.id, effective, devices, onError]);

  const set = <K extends keyof MemoryRequest>(k: K, v: MemoryRequest[K]) => setReq((r) => ({ ...r, [k]: v }));
  const ctxIdx = CTX_PRESETS.reduce((best, v, i) => (Math.abs(v - req.context_length) < Math.abs(CTX_PRESETS[best] - req.context_length) ? i : best), 0);
  const fit = est?.fit ?? null;
  const scale = Math.max(est?.total_bytes ?? 1, fit?.usable_bytes ?? 0) * 1.08;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Memory laboratory</h2>
          <p>Weight storage and inference memory for different precisions, context lengths and target devices.</p>
        </div>
        {loading && <Spinner />}
      </div>
      <EstimateBanner />

      <div className="grid" style={{ gridTemplateColumns: "minmax(280px, 340px) minmax(0, 1fr)" }}>
        {/* ---- controls ---- */}
        <div className="card col" style={{ gap: 14, alignSelf: "start" }}>
          <h3>Assumptions</h3>
          <label className="field">
            Weight format
            <select value={req.weight_format} onChange={(e) => set("weight_format", e.target.value)}>
              {est?.weights.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.label}
                </option>
              )) ?? <option value="current">As stored</option>}
            </select>
          </label>
          <label className="field">
            Context length per sequence: <span className="mono" style={{ color: "var(--text)" }}>{fmtInt(req.context_length)}</span>
            <input type="range" min={0} max={CTX_PRESETS.length - 1} value={ctxIdx} onChange={(e) => set("context_length", CTX_PRESETS[Number(e.target.value)])} />
            <input type="number" min={1} max={10_000_000} value={req.context_length} onChange={(e) => set("context_length", Math.max(1, Number(e.target.value) || 1))} />
            {trainCtx && <span className="faint">training context: {fmtInt(trainCtx)}</span>}
          </label>
          <label className="field">
            Parallel sequences (batch)
            <input type="number" min={1} max={4096} value={req.batch_size} onChange={(e) => set("batch_size", Math.max(1, Number(e.target.value) || 1))} />
          </label>
          <label className="field">
            KV cache type
            <select value={req.kv_cache_type} onChange={(e) => set("kv_cache_type", e.target.value as KvType)}>
              <option value="f16">F16 (llama.cpp default)</option>
              <option value="bf16">BF16</option>
              <option value="q8_0">Q8_0</option>
              <option value="q4_0">Q4_0</option>
              <option value="f32">F32</option>
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={req.flash_attention} onChange={(e) => set("flash_attention", e.target.checked)} />
            <span>Flash attention (no materialised attention-score matrix)</span>
          </label>
          <label className="field">
            Prompt micro-batch (n_ubatch)
            <input type="number" min={1} max={65536} value={req.n_ubatch} onChange={(e) => set("n_ubatch", Math.max(1, Number(e.target.value) || 1))} />
          </label>
          <label className="field">
            Target device
            <select value={req.device_id ?? "custom"} onChange={(e) => set("device_id", e.target.value)}>
              {devices.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
              <option value="custom">Custom…</option>
            </select>
          </label>
          {req.device_id === "custom" && <CustomDevice value={custom} onChange={setCustom} />}
        </div>

        {/* ---- results ---- */}
        <div className="col" style={{ gap: 18 }}>
          {est && (
            <>
              <div className="card">
                <div className="card-head">
                  <h3>Inference memory — {est.selected_weights.label}</h3>
                  <span className="stat-value">{fmtBytes(est.total_bytes)}</span>
                </div>
                <div className="stack-bar" aria-label="Memory breakdown">
                  {est.inference.map((c) => (
                    <span
                      key={c.key}
                      title={`${c.label}: ${fmtBytes(c.bytes)}`}
                      style={{ width: `${(c.bytes / scale) * 100}%`, background: SEG_COLORS[c.key] ?? "#64748b" }}
                    >
                      {c.bytes / scale > 0.1 ? c.label.split(" (")[0] : ""}
                    </span>
                  ))}
                  {fit && <div className="capacity-line" style={{ left: `${(fit.usable_bytes / scale) * 100}%` }} title={`usable on ${fit.device_name}: ${fmtBytes(fit.usable_bytes)}`} />}
                </div>
                {fit && (
                  <div className="small faint" style={{ marginTop: 6 }}>
                    vertical line = usable memory on {fit.device_name} ({fmtBytes(fit.usable_bytes)} of {fmtBytes(fit.capacity_bytes)})
                  </div>
                )}
                <div className="table-wrap" style={{ marginTop: 14 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Component</th>
                        <th className="num">Size</th>
                        <th>Provenance</th>
                        <th>How it is computed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {est.inference.map((c) => (
                        <tr key={c.key}>
                          <td className="nowrap">
                            <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 3, background: SEG_COLORS[c.key], marginRight: 8 }} />
                            {c.label}
                          </td>
                          <td className="num">{fmtBytes(c.bytes)}</td>
                          <td>
                            <Prov p={c.provenance} />
                          </td>
                          <td className="small dim mono">{c.formula}</td>
                        </tr>
                      ))}
                      <tr>
                        <td>
                          <strong>Total (estimate)</strong>
                        </td>
                        <td className="num">
                          <strong>{fmtBytes(est.total_bytes)}</strong>
                        </td>
                        <td colSpan={2} />
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {fit && (
                <div className="card">
                  <div className="card-head">
                    <h3>{fit.device_name}</h3>
                    <span className={`verdict ${fit.verdict}`}>{VERDICT[fit.verdict]} (estimate)</span>
                  </div>
                  <div className="grid grid-3">
                    <div className="stat">
                      <span className="stat-label">Required</span>
                      <span className="stat-value">{fmtBytes(fit.required_bytes)}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">Usable</span>
                      <span className="stat-value">{fmtBytes(fit.usable_bytes)}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">Headroom</span>
                      <span className="stat-value" style={{ color: fit.headroom_bytes < 0 ? "var(--bad)" : undefined }}>
                        {fmtBytes(fit.headroom_bytes)}
                      </span>
                    </div>
                  </div>
                  {fit.gpu_layers !== null && (
                    <p>
                      ≈ <strong>{fit.gpu_layers}</strong> of {fit.total_layers} layers on GPU (<code>-ngl {fit.gpu_layers}</code>), remainder in system RAM.
                    </p>
                  )}
                  {fit.decode_ceiling_tokens_per_s !== null && (
                    <p className="small">
                      Memory-bandwidth decode ceiling: <strong>≤ {fit.decode_ceiling_tokens_per_s} tokens/s</strong> — theoretical upper bound, not a benchmark.
                    </p>
                  )}
                  <div className="small dim">
                    <List items={[...fit.notes, ...fit.assumptions]} />
                  </div>
                </div>
              )}

              {allFits.length > 0 && (
                <div className="card">
                  <div className="card-head">
                    <h3>All device profiles</h3>
                    <span className="small faint">same settings</span>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Device</th>
                          <th className="num">Usable</th>
                          <th className="num">Required</th>
                          <th className="num">Headroom</th>
                          <th>Verdict (estimate)</th>
                          <th className="num">Decode ceiling</th>
                        </tr>
                      </thead>
                      <tbody>
                        {allFits.map((f) => (
                          <tr key={f.device_id}>
                            <td>{f.device_name}</td>
                            <td className="num">{fmtBytes(f.usable_bytes)}</td>
                            <td className="num">{fmtBytes(f.required_bytes)}</td>
                            <td className="num">{fmtBytes(f.headroom_bytes)}</td>
                            <td>
                              <span className={`verdict ${f.verdict}`} style={{ fontSize: 13 }}>
                                {VERDICT[f.verdict]}
                                {f.gpu_layers !== null ? ` (${f.gpu_layers}/${f.total_layers} layers)` : ""}
                              </span>
                            </td>
                            <td className="num">{f.decode_ceiling_tokens_per_s ? `≤ ${f.decode_ceiling_tokens_per_s} tok/s` : "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="card">
                <div className="card-head">
                  <h3>Weight storage by precision</h3>
                  <span className="small faint">click a row to use it above</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Format</th>
                        <th className="num">Size</th>
                        <th className="num">Bits / weight</th>
                        <th>Provenance</th>
                        <th>Method</th>
                      </tr>
                    </thead>
                    <tbody>
                      {est.weights.map((w) => (
                        <tr key={w.id} className={w.id === est.selected_weights.id ? "selected" : ""} onClick={() => set("weight_format", w.id)} style={{ cursor: "pointer" }}>
                          <td className="nowrap">
                            {w.label}
                            {Object.keys(w.type_mix).length > 0 && (
                              <div className="small faint mono">
                                {Object.entries(w.type_mix)
                                  .map(([k, v]) => `${k}×${v}`)
                                  .join(" ")}
                              </div>
                            )}
                          </td>
                          <td className="num">{fmtBytes(w.bytes)}</td>
                          <td className="num">{w.bits_per_weight}</td>
                          <td>
                            <Prov p={w.provenance} />
                          </td>
                          <td className="small dim">{w.method}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {(est.warnings.length > 0 || est.assumptions.length > 0) && (
                <div className="grid grid-2">
                  {est.warnings.length > 0 && (
                    <Callout kind="warn" title="Warnings">
                      <List items={est.warnings} />
                    </Callout>
                  )}
                  <Callout title="Assumptions">
                    <List items={est.assumptions} />
                  </Callout>
                </div>
              )}
              <p className="small faint">{est.disclaimer}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function CustomDevice({ value, onChange }: { value: DeviceProfile; onChange: (d: DeviceProfile) => void }) {
  const num = (k: keyof DeviceProfile, unit: number) => (e: ChangeEvent<HTMLInputElement>) =>
    onChange({ ...value, [k]: Math.max(0, Math.round((Number(e.target.value) || 0) * unit)) });
  return (
    <div className="col" style={{ gap: 10, padding: 12, border: "1px dashed var(--border-strong)", borderRadius: 8 }}>
      <label className="field">
        Memory kind
        <select value={value.memory_kind} onChange={(e) => onChange({ ...value, memory_kind: e.target.value as DeviceProfile["memory_kind"] })}>
          <option value="unified">Unified (Jetson / Apple-style)</option>
          <option value="system_ram">System RAM (CPU)</option>
          <option value="discrete_gpu">Discrete GPU VRAM</option>
        </select>
      </label>
      <label className="field">
        Capacity (GiB)
        <input type="number" step="0.5" min={0.5} value={value.capacity_bytes / GiB} onChange={num("capacity_bytes", GiB)} />
      </label>
      <label className="field">
        Reserved by OS / other apps (GiB)
        <input type="number" step="0.25" min={0} value={value.reserved_bytes / GiB} onChange={num("reserved_bytes", GiB)} />
      </label>
      <label className="field">
        Runtime overhead (MiB)
        <input type="number" step="50" min={0} value={value.runtime_overhead_bytes / MiB} onChange={num("runtime_overhead_bytes", MiB)} />
      </label>
      {value.memory_kind === "discrete_gpu" && (
        <label className="field">
          Host RAM available for offload (GiB)
          <input type="number" step="1" min={0} value={(value.host_ram_bytes ?? 0) / GiB} onChange={num("host_ram_bytes", GiB)} />
        </label>
      )}
      <label className="field">
        Memory bandwidth (GB/s)
        <input type="number" step="1" min={1} value={value.bandwidth_gbps ?? 0} onChange={(e) => onChange({ ...value, bandwidth_gbps: Math.max(1, Number(e.target.value) || 1) })} />
      </label>
    </div>
  );
}

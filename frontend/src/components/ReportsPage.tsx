import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { download, fmtBytes, fmtTime } from "../format";
import type { Comparison, ScanResult, ScanSummary } from "../types";
import { Callout, EstimateBanner, Spinner } from "./common";

interface Props {
  scan: ScanResult | null;
  history: ScanSummary[];
  onError: (m: string) => void;
}

const DEVICES = [
  ["laptop-cpu-16gb", "Laptop · CPU 16 GB"],
  ["laptop-gpu-8gb", "Laptop · 8 GB GPU"],
  ["jetson-orin-nano-8gb", "Jetson Orin Nano 8GB"],
  ["jetson-orin-nano-8gb-headless", "Jetson Orin Nano 8GB headless"],
];

export default function ReportsPage({ scan, history, onError }: Props) {
  const [md, setMd] = useState<string | null>(null);
  const [mdBusy, setMdBusy] = useState(false);
  const [a, setA] = useState(scan?.id ?? history[0]?.id ?? "");
  const [b, setB] = useState(history.find((h) => h.id !== (scan?.id ?? history[0]?.id))?.id ?? "");
  const [ctx, setCtx] = useState(4096);
  const [device, setDevice] = useState("jetson-orin-nano-8gb");
  const [cmp, setCmp] = useState<Comparison | null>(null);
  const [cmpBusy, setCmpBusy] = useState(false);
  const [onlyDiff, setOnlyDiff] = useState(false);
  const selRef = useRef("");
  selRef.current = `${a}|${b}|${ctx}|${device}`;

  useEffect(() => setMd(null), [scan?.id]);
  // A shown comparison must always match the current selection (the Markdown export uses the selection).
  useEffect(() => setCmp(null), [a, b, ctx, device]);

  const buildMd = async () => {
    if (!scan) return;
    setMdBusy(true);
    try {
      setMd(
        await api.reportMd(scan.id, {
          weight_format: "current",
          context_length: 4096,
          batch_size: 1,
          kv_cache_type: "f16",
          flash_attention: true,
          n_ubatch: 512,
          device_id: device,
          custom_device: null,
        }),
      );
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setMdBusy(false);
    }
  };

  const runCompare = async () => {
    if (!a || !b) return;
    setCmpBusy(true);
    const want = `${a}|${b}|${ctx}|${device}`;
    try {
      const c = await api.compare(a, b, ctx, device);
      if (want === selRef.current) setCmp(c);
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setCmpBusy(false);
    }
  };

  const exportAs = async (format: "json" | "csv") => {
    if (!scan) return;
    try {
      const text = await api.exportText(scan.id, format);
      const stem = scan.name.replace(/[^A-Za-z0-9._-]+/g, "-").slice(0, 60) || "model";
      download(format === "json" ? `${stem}.scan.json` : `${stem}.tensors.csv`, text, format === "json" ? "application/json" : "text/csv");
    } catch (e) {
      onError((e as Error).message);
    }
  };

  const copy = (text: string) => {
    if (!navigator.clipboard) return onError("clipboard unavailable on this origin; use Download instead");
    navigator.clipboard.writeText(text).catch(() => onError("clipboard unavailable; use Download instead"));
  };

  let section = "";
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Reports & comparison</h2>
          <p>Export scan results, generate a GitHub-ready Markdown report, and compare two scanned models.</p>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Export current scan</h3>
          {scan && <span className="small faint">{scan.name}</span>}
        </div>
        {!scan ? (
          <p className="dim">Open a scan (Import or History) to export it.</p>
        ) : (
          <>
            <div className="row">
              <button className="btn" onClick={() => exportAs("json")}>
                ⤓ JSON (full scan)
              </button>
              <button className="btn" onClick={() => exportAs("csv")}>
                ⤓ CSV (tensor table)
              </button>
              <select value={device} onChange={(e) => setDevice(e.target.value)} aria-label="Device for report">
                {DEVICES.map(([id, label]) => (
                  <option key={id} value={id}>
                    {label}
                  </option>
                ))}
              </select>
              <button className="btn primary" onClick={buildMd} disabled={mdBusy}>
                {mdBusy ? <Spinner /> : "Generate Markdown report"}
              </button>
            </div>
            {md && (
              <div className="col" style={{ marginTop: 14 }}>
                <div className="row">
                  <button className="btn sm" onClick={() => copy(md)}>
                    Copy Markdown
                  </button>
                  <button className="btn sm" onClick={() => download(`${scan.name}.report.md`, md, "text/markdown")}>
                    Download .md
                  </button>
                  <span className="small faint">GitHub-flavoured Markdown · provenance and estimates labelled</span>
                </div>
                <pre className="md-preview mono">{md}</pre>
              </div>
            )}
          </>
        )}
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Compare two models</h3>
        </div>
        {history.length < 2 ? (
          <p className="dim">Scan at least two models to compare them (for example a source model and its quantized output).</p>
        ) : (
          <>
            <div className="grid grid-2">
              <label className="field">
                Model A
                <ScanSelect value={a} onChange={setA} history={history} />
              </label>
              <label className="field">
                Model B
                <ScanSelect value={b} onChange={setB} history={history} />
              </label>
              <label className="field">
                Context length for inference estimates
                <input type="number" min={1} value={ctx} onChange={(e) => setCtx(Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <label className="field">
                Device profile
                <select value={device} onChange={(e) => setDevice(e.target.value)}>
                  {DEVICES.map(([id, label]) => (
                    <option key={id} value={id}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn primary" onClick={runCompare} disabled={!a || !b || a === b || cmpBusy}>
                {cmpBusy ? <Spinner /> : "Compare"}
              </button>
              {a === b && <span className="small faint">pick two different scans</span>}
              {cmp && (
                <>
                  <label className="check small">
                    <input type="checkbox" checked={onlyDiff} onChange={(e) => setOnlyDiff(e.target.checked)} />
                    only differences
                  </label>
                  <span className="spacer" />
                  <button
                    className="btn sm"
                    onClick={async () => {
                      try {
                        const text = await api.compareMd(a, b, ctx, device);
                        download("model-comparison.md", text, "text/markdown");
                      } catch (e) {
                        onError((e as Error).message);
                      }
                    }}
                  >
                    ⤓ Markdown
                  </button>
                </>
              )}
            </div>
          </>
        )}
        {cmp && (
          <div className="col" style={{ marginTop: 16, gap: 12 }}>
            <EstimateBanner>Size and memory rows are estimates unless labelled “As stored”. Architecture and quantization rows come from the scans.</EstimateBanner>
            {cmp.note && <Callout kind="warn">{cmp.note}</Callout>}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th style={{ width: "30%" }} />
                    <th>A · {cmp.a_name}</th>
                    <th>B · {cmp.b_name}</th>
                  </tr>
                </thead>
                <tbody>
                  {cmp.rows
                    .filter((r) => !onlyDiff || r.differs)
                    .flatMap((r, i) => {
                      const out = [];
                      if (r.section !== section) {
                        section = r.section;
                        out.push(
                          <tr key={`s${i}`}>
                            <td colSpan={3} style={{ background: "var(--bg-elev-2)" }}>
                              <h4>{r.section}</h4>
                            </td>
                          </tr>,
                        );
                      }
                      out.push(
                        <tr key={i} className={r.differs ? "diff" : ""}>
                          <td>
                            {r.label} {r.differs && <span className="badge warn">≠</span>}
                          </td>
                          <td className="mono small break">{r.a}</td>
                          <td className="mono small break">{r.b}</td>
                        </tr>,
                      );
                      return out;
                    })}
                </tbody>
              </table>
            </div>
            <div className="grid grid-4">
              <div className="stat">
                <span className="stat-label">Only in A</span>
                <span className="stat-value">{cmp.tensors_only_in_a}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Only in B</span>
                <span className="stat-value">{cmp.tensors_only_in_b}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Element count changed</span>
                <span className="stat-value">{cmp.tensors_shape_changed}</span>
              </div>
              <div className="stat">
                <span className="stat-label">dtype changed</span>
                <span className="stat-value">{cmp.tensors_dtype_changed}</span>
              </div>
            </div>
            {cmp.dtype_changes.length > 0 && (
              <div className="row" style={{ gap: 6 }}>
                {cmp.dtype_changes.map((d) => (
                  <span key={`${d.from}-${d.to}`} className="chip">
                    {d.from} → {d.to} × {d.count}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ScanSelect({ value, onChange, history }: { value: string; onChange: (v: string) => void; history: ScanSummary[] }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">— choose —</option>
      {history.map((h) => (
        <option key={h.id} value={h.id}>
          {h.name} · {h.quant_label ?? h.kind} · {fmtBytes(h.total_size_bytes)} · {fmtTime(h.scanned_at)}
        </option>
      ))}
    </select>
  );
}

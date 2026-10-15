import type { Tab } from "../App";
import { fmtBytes, fmtCount, fmtInt, fmtValue } from "../format";
import type { ScanResult } from "../types";
import { Callout, List, Prov, ProvLegend, Stat } from "./common";

export default function Overview({ scan, onNavigate }: { scan: ScanResult; onNavigate: (t: Tab) => void }) {
  const params = scan.hyperparameters.find((f) => f.key === "n_params");
  const logical = scan.hyperparameters.find((f) => f.key === "n_params_logical");
  const layers = scan.hyperparameters.find((f) => f.key === "n_layers");
  const ctx = scan.hyperparameters.find((f) => f.key === "context_length");
  const totalTensorBytes = scan.dtype_breakdown.reduce((a, d) => a + d.n_bytes, 0);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Overview</h2>
          <p>
            {scan.architecture.family ?? "unknown architecture"} · scanned {new Date(scan.scanned_at).toLocaleString()}
          </p>
        </div>
        <ProvLegend />
      </div>

      {!scan.integrity.valid && (
        <Callout kind="bad" title={`Integrity check failed (${scan.integrity.errors.length}).`}>
          The file is truncated, corrupted or malformed. Values below come from the readable parts of the header and may be incomplete.
          <List items={scan.integrity.errors} max={12} />
        </Callout>
      )}
      {scan.quantization.packed_checkpoint && (
        <Callout kind="warn" title="Packed quantized checkpoint.">
          Tensor dtypes describe storage containers ({scan.quantization.label}); element counts are not the logical parameter count.
        </Callout>
      )}

      <div className="grid grid-4">
        <Stat
          label="Parameters"
          value={fmtCount((logical?.value as number) ?? (params?.value as number))}
          sub={
            <span className="row" style={{ gap: 6 }}>
              {fmtInt((logical?.value as number) ?? (params?.value as number))} <Prov p={(logical ?? params)?.provenance} />
            </span>
          }
        />
        <Stat
          label="Quantization"
          value={scan.quantization.label ?? "—"}
          sub={
            <span className="row" style={{ gap: 6 }}>
              {scan.quantization.effective_bits_per_weight ? `${scan.quantization.effective_bits_per_weight} bits/weight` : ""}
              <Prov p={scan.quantization.label_provenance} />
            </span>
          }
        />
        <Stat label="Layers" value={fmtValue(layers?.value)} sub={<Prov p={layers?.provenance} />} />
        <Stat label="On disk" value={fmtBytes(scan.total_size_bytes)} sub={`${scan.tensors.length.toLocaleString()} tensors · ${scan.files.length} file(s)`} />
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="card-head">
            <h3>Hyperparameters</h3>
            {ctx && <span className="small faint">training context {fmtValue(ctx.value)}</span>}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Property</th>
                  <th>Value</th>
                  <th>Provenance</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {scan.hyperparameters.map((f) => (
                  <tr key={f.key}>
                    <td>{f.label}</td>
                    <td className="mono">{fmtValue(f.value)}</td>
                    <td>
                      <Prov p={f.provenance} />
                    </td>
                    <td className="small faint mono break">{f.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="col" style={{ gap: 18 }}>
          <div className="card">
            <div className="card-head">
              <h3>Tensor storage by dtype</h3>
              <span className="small faint">from tensor descriptors</span>
            </div>
            <div className="stack-bar" style={{ marginBottom: 12 }}>
              {scan.dtype_breakdown.map((d, i) => (
                <span
                  key={d.dtype}
                  title={`${d.dtype}: ${fmtBytes(d.n_bytes)}`}
                  style={{ width: `${(d.n_bytes / Math.max(totalTensorBytes, 1)) * 100}%`, background: `hsl(${170 + i * 47} 65% 60%)` }}
                >
                  {d.n_bytes / Math.max(totalTensorBytes, 1) > 0.08 ? d.dtype : ""}
                </span>
              ))}
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>dtype</th>
                    <th className="num">tensors</th>
                    <th className="num">elements</th>
                    <th className="num">bytes</th>
                  </tr>
                </thead>
                <tbody>
                  {scan.dtype_breakdown.map((d) => (
                    <tr key={d.dtype}>
                      <td className="mono">{d.dtype}</td>
                      <td className="num">{fmtInt(d.n_tensors)}</td>
                      <td className="num">{fmtInt(d.n_elements)}</td>
                      <td className="num">{fmtBytes(d.n_bytes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {scan.quantization.notes.length > 0 && (
              <div className="small dim" style={{ marginTop: 10 }}>
                <List items={scan.quantization.notes} />
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-head">
              <h3>Format</h3>
              <span className={`badge ${scan.integrity.valid ? "ok" : "bad"}`}>{scan.integrity.valid ? "all tensor ranges verified" : "integrity errors"}</span>
            </div>
            <dl className="kv">
              <dt>Container</dt>
              <dd>
                {scan.format.name} {scan.format.version}
              </dd>
              {Object.entries(scan.format.details).map(([k, v]) => (
                <FragmentKV key={k} k={k} v={v} />
              ))}
            </dl>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Files</h3>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Path</th>
                <th>Role</th>
                <th>Parsed</th>
                <th className="num">Size</th>
              </tr>
            </thead>
            <tbody>
              {scan.files.map((f) => (
                <tr key={f.path}>
                  <td className="mono small break">
                    {f.path}
                    {f.note && <div className="faint">{f.note}</div>}
                  </td>
                  <td className="nowrap">{f.role}</td>
                  <td>{f.parsed ? <span className="badge ok">header</span> : <span className="badge">not opened</span>}</td>
                  <td className="num">{fmtBytes(f.size_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {(scan.integrity.warnings.length > 0 || scan.architecture.notes.length > 0) && (
        <Callout kind="warn" title="Notes">
          <List items={[...scan.architecture.notes, ...scan.integrity.warnings]} max={20} />
        </Callout>
      )}

      <div className="row">
        <button className="btn" onClick={() => onNavigate("architecture")}>
          Explore architecture →
        </button>
        <button className="btn" onClick={() => onNavigate("memory")}>
          Estimate memory →
        </button>
        <button className="btn" onClick={() => onNavigate("quantize")}>
          Quantize / convert →
        </button>
      </div>
    </div>
  );
}

function FragmentKV({ k, v }: { k: string; v: unknown }) {
  return (
    <>
      <dt>{k.replace(/_/g, " ")}</dt>
      <dd className="mono small">{Array.isArray(v) ? v.join(", ") || "—" : fmtValue(v)}</dd>
    </>
  );
}

import { useState } from "react";
import { fmtBytes, fmtCount, fmtInt } from "../format";
import type { HubInfo, ScanResult } from "../types";
import { Prov } from "./common";

const hubLink = (path: string) => `https://huggingface.co/${path}`;

function date(s: string | null) {
  if (!s) return "—";
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export default function HubCard({ scan, onInspect, busy }: { scan: ScanResult; onInspect: (file: string) => void; busy: string | null }) {
  const h = scan.hub as HubInfo;
  const [showFiles, setShowFiles] = useState(false);
  const [showCard, setShowCard] = useState(false);
  const ours = scan.hyperparameters.find((f) => f.key === "n_params")?.value as number | undefined;
  const hubTotal = h.api_parameters?.total ?? h.api_gguf?.total ?? null;
  const match = hubTotal !== null && ours !== undefined ? hubTotal === ours : null;

  return (
    <div className="card">
      <div className="card-head">
        <div style={{ minWidth: 0 }}>
          <h3>
            <span aria-hidden>🤗</span>{" "}
            <a href={h.url} target="_blank" rel="noreferrer noopener">
              {h.repo}
            </a>
          </h3>
          <div className="small faint mono">
            {h.revision} @ {h.sha.slice(0, 12)} · inspected: {h.scanned_target}
          </div>
        </div>
        <div className="row" style={{ gap: 6 }}>
          {h.pipeline_tag && <span className="badge info">{h.pipeline_tag}</span>}
          {h.library_name && <span className="badge">{h.library_name}</span>}
          {h.license && <span className="badge">license: {h.license}</span>}
          {h.gated && <span className="badge warn">gated ({h.gated})</span>}
          {h.private && <span className="badge warn">private</span>}
        </div>
      </div>

      <p className="small dim" style={{ marginTop: 0 }}>
        Repository details below are <strong>reported by the Hugging Face Hub API</strong>, not read from the weights. Everything else on this page comes from
        the model files' own headers ({h.requests} requests, {fmtBytes(h.bytes_fetched)} fetched — the weights were not downloaded).
      </p>

      <div className="grid grid-4" style={{ marginBottom: 14 }}>
        <div className="stat">
          <span className="stat-label">Downloads (30 days)</span>
          <span className="stat-value">{fmtCount(h.downloads)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Likes</span>
          <span className="stat-value">{fmtInt(h.likes)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Repository size</span>
          <span className="stat-value">{fmtBytes(h.total_file_bytes)}</span>
          <span className="stat-sub">{h.file_count} files</span>
        </div>
        <div className="stat">
          <span className="stat-label">Updated</span>
          <span className="stat-value" style={{ fontSize: 18 }}>
            {date(h.last_modified)}
          </span>
          <span className="stat-sub">created {date(h.created_at)}</span>
        </div>
      </div>

      <dl className="kv">
        <dt>Parameters (Hub)</dt>
        <dd>
          {hubTotal !== null ? (
            <>
              {fmtInt(hubTotal)}
              {h.api_parameters && Object.keys(h.api_parameters.by_dtype).length > 0 && (
                <span className="faint">
                  {" "}
                  ({Object.entries(h.api_parameters.by_dtype)
                    .map(([k, v]) => `${k}: ${fmtCount(v)}`)
                    .join(", ")})
                </span>
              )}{" "}
              {match === true && <span className="badge ok">matches the tensors read from the files</span>}
              {match === false && (
                <span className="badge warn">
                  tensors sum to {fmtInt(ours ?? null)} <Prov p="derived" />
                </span>
              )}
            </>
          ) : (
            "not reported"
          )}
        </dd>
        {h.base_models.length > 0 && (
          <>
            <dt>Base model</dt>
            <dd>
              {h.base_models.map((b) => (
                <a key={b} className="mono small" href={hubLink(b)} target="_blank" rel="noreferrer noopener" style={{ marginRight: 10 }}>
                  {b}
                </a>
              ))}
            </dd>
          </>
        )}
        {h.datasets.length > 0 && (
          <>
            <dt>Datasets</dt>
            <dd className="small">{h.datasets.join(", ")}</dd>
          </>
        )}
        {h.languages.length > 0 && (
          <>
            <dt>Languages</dt>
            <dd className="small">{h.languages.join(", ")}</dd>
          </>
        )}
        {h.tags.length > 0 && (
          <>
            <dt>Tags</dt>
            <dd>
              {h.tags.map((t) => (
                <span key={t} className="chip">
                  {t}
                </span>
              ))}
            </dd>
          </>
        )}
      </dl>

      {h.gguf_variants.length > 0 && (
        <>
          <h4 style={{ margin: "18px 0 8px" }}>GGUF variants in this repository ({h.gguf_variants.length})</h4>
          <div className="table-wrap" style={{ maxHeight: 320 }}>
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Quant</th>
                  <th className="num">Size</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {h.gguf_variants.map((v) => {
                  const current = v.path === h.scanned_target;
                  return (
                    <tr key={v.path} className={current ? "selected" : ""}>
                      <td className="mono small break">
                        {v.path}
                        {v.shards > 1 && <span className="faint"> · {v.shards} shards</span>}
                      </td>
                      <td className="mono">{v.label ?? "—"}</td>
                      <td className="num">{fmtBytes(v.size)}</td>
                      <td className="right nowrap">
                        {current ? (
                          <span className="badge ok">inspected</span>
                        ) : (
                          <button className="btn sm" disabled={!!busy} onClick={() => onInspect(v.path)}>
                            {busy === v.path ? "…" : "Inspect"}
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="row" style={{ marginTop: 14 }}>
        <button className="btn sm" onClick={() => setShowFiles(!showFiles)}>
          {showFiles ? "Hide" : "Show"} files ({h.file_count})
        </button>
        {h.readme_excerpt && (
          <button className="btn sm" onClick={() => setShowCard(!showCard)}>
            {showCard ? "Hide" : "Show"} model card
          </button>
        )}
        <a className="btn sm ghost" href={h.url} target="_blank" rel="noreferrer noopener">
          Open on huggingface.co ↗
        </a>
      </div>
      {showFiles && (
        <div className="table-wrap" style={{ maxHeight: 320, marginTop: 10 }}>
          <table>
            <tbody>
              {h.files.map((f) => (
                <tr key={f.path}>
                  <td className="mono small break">{f.path}</td>
                  <td className="num small">{fmtBytes(f.size)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {showCard && h.readme_excerpt && (
        <pre className="md-preview" style={{ marginTop: 10, fontFamily: "var(--sans)", fontSize: 13 }}>
          {h.readme_excerpt}
        </pre>
      )}
    </div>
  );
}

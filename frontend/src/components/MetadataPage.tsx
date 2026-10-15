import { useMemo, useState } from "react";
import { fmtInt } from "../format";
import type { MetadataEntry, ScanResult } from "../types";
import { Prov } from "./common";

function render(e: MetadataEntry): string {
  if (Array.isArray(e.value)) {
    const shown = (e.value as unknown[]).slice(0, 16).map((v) => (typeof v === "string" ? JSON.stringify(v) : String(v)));
    const more = (e.length ?? e.value.length) - shown.length;
    return `[${shown.join(", ")}${more > 0 ? `, … ${fmtInt(more)} more` : ""}]`;
  }
  if (e.value !== null && typeof e.value === "object") return JSON.stringify(e.value, null, 1);
  return String(e.value);
}

export default function MetadataPage({ scan }: { scan: ScanResult }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const rows = useMemo(() => {
    const ql = q.toLowerCase();
    return scan.metadata.filter((m) => !ql || m.key.toLowerCase().includes(ql) || render(m).toLowerCase().includes(ql));
  }, [scan.metadata, q]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Metadata</h2>
          <p>
            Raw key/value metadata exactly as stored in the file <Prov p="verified" />. Long strings and arrays are truncated for display; lengths are exact.
            Values are untrusted data and are shown as plain text only.
          </p>
        </div>
      </div>
      <div className="card">
        <input type="text" placeholder="Filter keys or values…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: "100%", marginBottom: 12 }} />
        <div className="table-wrap" style={{ maxHeight: "calc(100vh - 300px)" }}>
          <table>
            <thead>
              <tr>
                <th>Key</th>
                <th>Type</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m) => {
                const text = render(m);
                const long = text.length > 300;
                return (
                  <tr key={m.key}>
                    <td className="mono small break" style={{ width: "28%" }}>
                      {m.key}
                    </td>
                    <td className="mono small nowrap">
                      {m.type}
                      {m.length !== null && <div className="faint">len {fmtInt(m.length)}</div>}
                    </td>
                    <td className="mono small break">
                      <span style={{ whiteSpace: "pre-wrap" }}>{long && open !== m.key ? text.slice(0, 300) + "…" : text}</span>
                      {long && (
                        <button className="btn ghost sm" onClick={() => setOpen(open === m.key ? null : m.key)}>
                          {open === m.key ? "collapse" : "expand"}
                        </button>
                      )}
                      {m.truncated && <span className="badge warn" style={{ marginLeft: 6 }}>truncated for display</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

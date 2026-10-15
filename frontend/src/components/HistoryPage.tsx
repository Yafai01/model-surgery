import { useState } from "react";
import { api } from "../api";
import { fmtBytes, fmtCount, fmtTime } from "../format";
import type { ScanSummary } from "../types";
import { Empty } from "./common";

interface Props {
  history: ScanSummary[];
  onOpen: (id: string) => void;
  onChanged: () => void;
  onError: (m: string) => void;
}

export default function HistoryPage({ history, onOpen, onChanged, onError }: Props) {
  const [q, setQ] = useState("");
  const rows = history.filter((h) => !q || `${h.name} ${h.path} ${h.quant_label}`.toLowerCase().includes(q.toLowerCase()));

  const remove = async (id: string) => {
    if (!window.confirm("Remove this scan from history? (The model file itself is not touched.)")) return;
    try {
      await api.deleteScan(id);
      onChanged();
    } catch (e) {
      onError((e as Error).message);
    }
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Scan history</h2>
          <p>Stored locally in SQLite. Removing an entry never touches model files.</p>
        </div>
        <input type="text" placeholder="Filter…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      {history.length === 0 ? (
        <Empty title="No scans yet">Import a model to start building a history.</Empty>
      ) : (
        <div className="card">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Scanned</th>
                  <th>Model</th>
                  <th>Format</th>
                  <th className="num">Params</th>
                  <th className="num">Size</th>
                  <th>Integrity</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((h) => (
                  <tr key={h.id}>
                    <td className="nowrap small">{fmtTime(h.scanned_at)}</td>
                    <td>
                      <div>{h.name}</div>
                      <div className="mono small faint break">{h.path}</div>
                    </td>
                    <td className="nowrap">
                      {h.kind === "gguf" ? "GGUF" : h.kind === "hf_safetensors" ? "HF safetensors" : "safetensors"}
                      {h.quant_label && <div className="small faint">{h.quant_label}</div>}
                    </td>
                    <td className="num">{fmtCount(h.n_params)}</td>
                    <td className="num">{fmtBytes(h.total_size_bytes)}</td>
                    <td>
                      <span className={`badge ${h.valid ? "ok" : "bad"}`}>{h.valid ? "ok" : "errors"}</span>
                    </td>
                    <td className="nowrap">
                      <button className="btn sm" onClick={() => onOpen(h.id)}>
                        Open
                      </button>{" "}
                      <button className="btn sm ghost" onClick={() => remove(h.id)} aria-label={`Remove ${h.name} from history`}>
                        ✕
                      </button>
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

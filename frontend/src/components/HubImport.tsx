import { useState } from "react";
import { api } from "../api";
import { fmtBytes } from "../format";
import { getHubToken, setHubToken } from "../hubSession";
import type { ScanResult } from "../types";
import { Callout, Spinner } from "./common";

const EXAMPLES = ["Qwen/Qwen2.5-0.5B-Instruct", "bartowski/Qwen2.5-0.5B-Instruct-GGUF", "microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen3-30B-A3B"];

export default function HubImport({ onScanned }: { onScanned: (r: ScanResult) => void }) {
  const [link, setLink] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ requests: number; bytes: number; stage: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showToken, setShowToken] = useState(!!getHubToken());
  const [token, setToken] = useState(getHubToken() ?? "");

  const inspect = async (value: string) => {
    if (!value.trim()) return;
    setLink(value);
    setBusy(true);
    setError(null);
    setProgress(null);
    setHubToken(token);
    try {
      onScanned(await api.scanHub(value, { token: getHubToken(), onProgress: setProgress }));
    } catch (e) {
      const msg = (e as Error).message;
      setError(msg);
      if (/gated|token|private/i.test(msg)) setShowToken(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="card-head">
        <h3>From Hugging Face</h3>
        <span className="small faint">reads only header bytes over HTTP range requests — weights are not downloaded</span>
      </div>
      <form
        className="row"
        onSubmit={(ev) => {
          ev.preventDefault();
          inspect(link);
        }}
      >
        <input
          type="text"
          className="mono"
          style={{ flex: 1, minWidth: 280 }}
          placeholder="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct   or   owner/model"
          value={link}
          onChange={(e) => setLink(e.target.value)}
          spellCheck={false}
          aria-label="Hugging Face model link"
        />
        <button className="btn primary" type="submit" disabled={busy || !link.trim()}>
          {busy ? <Spinner /> : "Inspect"}
        </button>
      </form>
      <div className="row small" style={{ marginTop: 10, gap: 6 }}>
        <span className="faint">Try:</span>
        {EXAMPLES.map((x) => (
          <button key={x} className="chip" style={{ cursor: "pointer" }} disabled={busy} onClick={() => inspect(x)}>
            {x}
          </button>
        ))}
        <span className="spacer" />
        <button className="btn ghost sm" onClick={() => setShowToken(!showToken)}>
          {showToken ? "hide token" : "gated or private model?"}
        </button>
      </div>
      {showToken && (
        <label className="field" style={{ marginTop: 10 }}>
          Hugging Face access token (read scope) — kept in this tab's memory only and sent only to huggingface.co
          <input
            type="password"
            className="mono"
            autoComplete="off"
            placeholder="hf_…"
            value={token}
            onChange={(e) => {
              setToken(e.target.value);
              setHubToken(e.target.value);
            }}
          />
        </label>
      )}
      {busy && progress && (
        <div className="small dim" style={{ marginTop: 10 }}>
          <Spinner /> {progress.stage} · {progress.requests} request{progress.requests === 1 ? "" : "s"} · {fmtBytes(progress.bytes)} fetched
        </div>
      )}
      {error && (
        <div style={{ marginTop: 12 }}>
          <Callout kind="bad" title="Could not inspect:">
            {error}
          </Callout>
        </div>
      )}
    </div>
  );
}

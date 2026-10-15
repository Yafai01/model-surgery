import { type DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { BROWSER_EDITION, api } from "../api";
import { fmtBytes, fmtCount, fmtTime } from "../format";
import type { FsEntry, FsListing, FsRoot, ScanResult, ScanSummary } from "../types";
import { Callout, Spinner } from "./common";

interface Props {
  onScanned: (r: ScanResult) => void;
  onError: (msg: string) => void;
  recent: ScanSummary[];
  onOpen: (id: string) => void;
}

const KIND_ICON: Record<string, string> = { gguf: "◆", safetensors: "◇", json: "{}", pickle: "⚠", onnx: "○" };

function crumbs(path: string): { label: string; path: string }[] {
  const sep = path.includes("\\") ? "\\" : "/";
  const parts = path.split(/[\\/]/).filter(Boolean);
  const out: { label: string; path: string }[] = [];
  let acc = path.startsWith("/") ? "/" : "";
  for (const p of parts) {
    acc = acc === "" ? p + sep : acc.endsWith(sep) ? acc + p : acc + sep + p;
    out.push({ label: p, path: acc });
  }
  return out;
}

export default function ImportPage(props: Props) {
  return BROWSER_EDITION ? <BrowserImport {...props} /> : <LocalImport {...props} />;
}

function RecentScans({ recent, onOpen }: Pick<Props, "recent" | "onOpen">) {
  return (
    <div className="card">
      <h3 style={{ marginBottom: 10 }}>Recent scans</h3>
      {recent.length === 0 && <p className="dim small">Nothing scanned yet.</p>}
      <div className="col">
        {recent.slice(0, 8).map((r) => (
          <button key={r.id} className="nav-item" onClick={() => onOpen(r.id)} style={{ flexDirection: "column", alignItems: "flex-start", gap: 0 }}>
            <span style={{ color: "var(--text)", fontWeight: 500 }}>{r.name}</span>
            <span className="small faint">
              {r.quant_label ?? r.kind} · {fmtCount(r.n_params)} params · {fmtBytes(r.total_size_bytes)} · {fmtTime(r.scanned_at)}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ---- browser edition: files are read locally with the File API; nothing is uploaded ----

function withPath(f: File, path: string): File {
  Object.defineProperty(f, "webkitRelativePath", { value: path, configurable: true });
  return f;
}

async function filesFromEntry(entry: FileSystemEntry, prefix: string, depth = 0): Promise<File[]> {
  if (entry.isFile) {
    const f = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
    return [withPath(f, prefix + f.name)];
  }
  if (!entry.isDirectory || depth > 2) return [];
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  const out: File[] = [];
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
    if (!batch.length) break;
    for (const e of batch) out.push(...(await filesFromEntry(e, `${prefix}${entry.name}/`, depth + 1)));
  }
  return out;
}

function BrowserImport({ onScanned, recent, onOpen }: Props) {
  const [busy, setBusy] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const dirInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    dirInput.current?.setAttribute("webkitdirectory", "");
  }, []);

  const scan = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setScanError(null);
    try {
      onScanned(await api.scanFiles(files));
    } catch (e) {
      setScanError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onDrop = async (ev: DragEvent) => {
    ev.preventDefault();
    setDrag(false);
    const items = [...ev.dataTransfer.items].map((i) => i.webkitGetAsEntry()).filter((e): e is FileSystemEntry => !!e);
    const files = (await Promise.all(items.map((e) => filesFromEntry(e, "")))).flat();
    scan(files.length ? files : [...ev.dataTransfer.files]);
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Import a model</h2>
          <p>Pick a GGUF file or a Hugging Face model folder. Everything runs in this tab: only the header bytes are read, and nothing is uploaded.</p>
        </div>
      </div>

      <div
        className="card"
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        style={{ borderStyle: "dashed", borderWidth: 2, borderColor: drag ? "var(--accent)" : "var(--border-strong)", textAlign: "center", padding: "40px 20px" }}
      >
        <h3 style={{ marginBottom: 6 }}>{busy ? "Reading headers…" : "Drop a .gguf file or a model folder here"}</h3>
        <p className="dim" style={{ marginTop: 0 }}>
          Multi-GB files are fine — only the first few megabytes are read.
        </p>
        <div className="row" style={{ justifyContent: "center", marginTop: 14 }}>
          <button className="btn primary" disabled={busy} onClick={() => fileInput.current?.click()}>
            {busy ? <Spinner /> : "Choose GGUF / safetensors file…"}
          </button>
          <button className="btn" disabled={busy} onClick={() => dirInput.current?.click()}>
            Choose Hugging Face folder…
          </button>
        </div>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".gguf,.safetensors"
          hidden
          onChange={(e) => {
            scan([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
        <input
          ref={dirInput}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            scan([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
        <p className="small faint" style={{ marginTop: 14 }}>
          Your browser may ask to “upload” the folder — that is its standard wording for granting this page read access. No file leaves your computer.
        </p>
        {scanError && (
          <div style={{ marginTop: 12, textAlign: "left" }}>
            <Callout kind="bad" title="Could not scan:">
              {scanError}
            </Callout>
          </div>
        )}
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>How this works</h3>
          <ul className="small dim" style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>
            <li>GGUF and safetensors headers are parsed in JavaScript with strict bounds checks.</li>
            <li>Tensor data is never read, so inspecting a 70B model is as fast as a 0.5B one.</li>
            <li>Pickle files (.bin, .pt, .ckpt) are never opened.</li>
            <li>Scan history stays in this browser (IndexedDB).</li>
            <li>
              Quantization needs llama.cpp on your machine — use the{" "}
              <a href="https://github.com/Yafai01/model-surgery" target="_blank" rel="noreferrer">
                local app
              </a>
              .
            </li>
          </ul>
        </div>
        <RecentScans recent={recent} onOpen={onOpen} />
      </div>
    </div>
  );
}

// ---- local edition: server-side path entry and file browser ----

function LocalImport({ onScanned, onError, recent, onOpen }: Props) {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [roots, setRoots] = useState<FsRoot[]>([]);
  const [listing, setListing] = useState<FsListing | null>(null);
  const [loadingDir, setLoadingDir] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  const browse = useCallback(
    async (p: string) => {
      setLoadingDir(true);
      try {
        setListing(await api.list(p));
      } catch (e) {
        onError((e as Error).message);
      } finally {
        setLoadingDir(false);
      }
    },
    [onError],
  );

  useEffect(() => {
    api
      .roots()
      .then((r) => {
        setRoots(r);
        if (r.length) browse(r[0].path);
      })
      .catch((e) => onError(e.message));
  }, [browse, onError]);

  const scan = async (target: string) => {
    if (!target.trim()) return;
    setBusy(true);
    setScanError(null);
    setPath(target);
    try {
      onScanned(await api.scan(target.trim()));
    } catch (e) {
      setScanError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const clickEntry = (e: FsEntry) => {
    if (e.type === "dir") browse(e.path);
    else setPath(e.path);
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Import a model</h2>
          <p>Select a GGUF file or a Hugging Face directory (config.json + safetensors). Only headers are read; tensor data is never loaded.</p>
        </div>
      </div>

      <div className="card">
        <form
          className="row"
          onSubmit={(ev) => {
            ev.preventDefault();
            scan(path);
          }}
        >
          <input
            type="text"
            className="mono"
            style={{ flex: 1, minWidth: 280 }}
            placeholder="C:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf   or   C:\models\Llama-3.2-1B"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            spellCheck={false}
            aria-label="Model path"
          />
          <button className="btn primary" type="submit" disabled={busy || !path.trim()}>
            {busy ? <Spinner /> : "Scan"}
          </button>
        </form>
        {scanError && (
          <div style={{ marginTop: 12 }}>
            <Callout kind="bad" title="Could not scan:">
              {scanError}
            </Callout>
          </div>
        )}
      </div>

      <div className="grid grid-2" style={{ gridTemplateColumns: "minmax(0, 1.6fr) minmax(0, 1fr)" }}>
        <div className="card">
          <div className="card-head">
            <h3>Browse this computer</h3>
            <div className="row">
              {roots.map((r) => (
                <button key={r.path} className="btn sm" onClick={() => browse(r.path)}>
                  {r.label}
                </button>
              ))}
            </div>
          </div>
          {listing && (
            <div className="row" style={{ marginBottom: 10 }}>
              {listing.parent && (
                <button className="btn sm" onClick={() => browse(listing.parent!)} aria-label="Up one directory">
                  ↑ Up
                </button>
              )}
              <div className="crumbs mono">
                {crumbs(listing.path).map((c, i) => (
                  <span key={c.path}>
                    {i > 0 && <span className="faint">›</span>}
                    <button onClick={() => browse(c.path)}>{c.label}</button>
                  </span>
                ))}
              </div>
              <span className="spacer" />
              {loadingDir && <Spinner />}
            </div>
          )}
          <div className="fs-list">
            {listing?.entries.length === 0 && <div className="empty small">No folders or model files here.</div>}
            {listing?.entries.map((e) => (
              <div key={e.path} className="fs-row" onClick={() => clickEntry(e)} onDoubleClick={() => e.type === "file" && scan(e.path)}>
                <span className="nav-icon" aria-hidden>
                  {e.type === "dir" ? (e.is_model_dir ? "▣" : "▸") : KIND_ICON[e.kind ?? ""] ?? "·"}
                </span>
                <span className="fs-name">{e.name}</span>
                {e.is_model_dir && <span className="badge info">HF model</span>}
                {e.kind === "pickle" && (
                  <span className="badge warn" title="Pickle files can execute code when loaded; Model Surgery never opens them">
                    pickle · not opened
                  </span>
                )}
                {e.size_bytes !== null && <span className="small faint nowrap">{fmtBytes(e.size_bytes)}</span>}
                {(e.is_model_dir || e.kind === "gguf" || e.kind === "safetensors") && (
                  <button
                    className="btn sm primary"
                    disabled={busy}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      scan(e.path);
                    }}
                  >
                    Scan
                  </button>
                )}
              </div>
            ))}
          </div>
          {listing?.truncated && <p className="small faint">Listing truncated to 2,000 entries.</p>}
          {listing && (
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn sm" onClick={() => scan(listing.path)} disabled={busy}>
                Scan this directory
              </button>
              <span className="small faint">for Hugging Face model folders</span>
            </div>
          )}
        </div>

        <div className="col" style={{ gap: 18 }}>
          <div className="card">
            <h3 style={{ marginBottom: 10 }}>Supported inputs</h3>
            <table>
              <tbody>
                <tr>
                  <td>
                    <span className="badge ok">GGUF</span>
                  </td>
                  <td className="small">v2/v3, little- or big-endian, split shards (-00001-of-N)</td>
                </tr>
                <tr>
                  <td>
                    <span className="badge ok">HF dir</span>
                  </td>
                  <td className="small">config.json + safetensors (single or sharded with index)</td>
                </tr>
                <tr>
                  <td>
                    <span className="badge ok">.safetensors</span>
                  </td>
                  <td className="small">single file; architecture inferred from tensor names only</td>
                </tr>
                <tr>
                  <td>
                    <span className="badge bad">.bin .pt .ckpt</span>
                  </td>
                  <td className="small">refused: pickle formats can execute code when loaded</td>
                </tr>
                <tr>
                  <td>
                    <span className="badge">ONNX, TFLite…</span>
                  </td>
                  <td className="small">reported as unsupported</td>
                </tr>
              </tbody>
            </table>
          </div>
          <RecentScans recent={recent} onOpen={onOpen} />
        </div>
      </div>
    </div>
  );
}

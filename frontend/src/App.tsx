import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Callout } from "./components/common";
import ArchitecturePage from "./components/ArchitecturePage";
import HistoryPage from "./components/HistoryPage";
import ImportPage from "./components/ImportPage";
import MemoryLab from "./components/MemoryLab";
import MetadataPage from "./components/MetadataPage";
import Overview from "./components/Overview";
import QuantizePage from "./components/QuantizePage";
import ReportsPage from "./components/ReportsPage";
import TensorTable from "./components/TensorTable";
import { fmtBytes } from "./format";
import type { ScanResult, ScanSummary } from "./types";

export type Tab = "import" | "overview" | "architecture" | "tensors" | "metadata" | "memory" | "quantize" | "reports" | "history";

const NAV: { group: string; items: { id: Tab; label: string; icon: string; needsScan?: boolean }[] }[] = [
  { group: "Model", items: [{ id: "import", label: "Import", icon: "⇪" }] },
  {
    group: "Inspect",
    items: [
      { id: "overview", label: "Overview", icon: "◉", needsScan: true },
      { id: "architecture", label: "Architecture", icon: "▤", needsScan: true },
      { id: "tensors", label: "Tensors", icon: "▦", needsScan: true },
      { id: "metadata", label: "Metadata", icon: "≡", needsScan: true },
    ],
  },
  {
    group: "Optimize",
    items: [
      { id: "memory", label: "Memory Lab", icon: "◫", needsScan: true },
      { id: "quantize", label: "Quantize", icon: "⚙", needsScan: true },
    ],
  },
  {
    group: "Records",
    items: [
      { id: "reports", label: "Reports & Compare", icon: "⎘" },
      { id: "history", label: "History", icon: "⟲" },
    ],
  },
];

function Logo() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="#0b1220" stroke="#23314a" />
      <g fill="none" strokeWidth="2.2" strokeLinecap="round">
        <path d="M8 22h16M8 16h16M8 10h16" stroke="#2dd4bf" />
        <path d="M12 7v18" stroke="#38bdf8" />
      </g>
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("import");
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [history, setHistory] = useState<ScanSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshHistory = useCallback(() => {
    api.scans().then(setHistory).catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(refreshHistory, [refreshHistory]);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 9000);
    return () => clearTimeout(t);
  }, [error]);

  const onScanned = useCallback(
    (r: ScanResult) => {
      setScan(r);
      setTab("overview");
      refreshHistory();
    },
    [refreshHistory],
  );

  const openScan = useCallback(
    async (id: string, nextTab: Tab = "overview") => {
      try {
        setScan(await api.getScan(id));
        setTab(nextTab);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [],
  );

  const body = (() => {
    if (tab === "import") return <ImportPage onScanned={onScanned} onError={setError} recent={history} onOpen={openScan} />;
    if (tab === "history") return <HistoryPage history={history} onOpen={openScan} onChanged={refreshHistory} onError={setError} />;
    if (tab === "reports") return <ReportsPage scan={scan} history={history} onError={setError} />;
    if (!scan) return null;
    switch (tab) {
      case "overview":
        return <Overview scan={scan} onNavigate={setTab} />;
      case "architecture":
        return <ArchitecturePage scan={scan} />;
      case "tensors":
        return <TensorTable scan={scan} />;
      case "metadata":
        return <MetadataPage scan={scan} />;
      case "memory":
        return <MemoryLab scan={scan} onError={setError} />;
      case "quantize":
        return <QuantizePage scan={scan} onError={setError} onOpenScan={openScan} onHistoryChanged={refreshHistory} />;
    }
    return null;
  })();

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Logo />
          <div>
            <div className="brand-name">Model Surgery</div>
            <div className="brand-sub">inspect · estimate · quantize</div>
          </div>
        </div>
        {NAV.map((g) => (
          <div key={g.group}>
            <div className="nav-group">{g.group}</div>
            {g.items.map((it) => (
              <button
                key={it.id}
                className={`nav-item ${tab === it.id ? "active" : ""}`}
                disabled={it.needsScan && !scan}
                onClick={() => setTab(it.id)}
                title={it.needsScan && !scan ? "Import a model first" : undefined}
              >
                <span className="nav-icon" aria-hidden>
                  {it.icon}
                </span>
                {it.label}
              </button>
            ))}
          </div>
        ))}
        <div className="sidebar-foot">
          Local-only · read-only inspection
          <br />
          Model files are never executed.
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          {scan ? (
            <>
              <div style={{ minWidth: 0 }}>
                <div className="topbar-title">{scan.name}</div>
                <div className="topbar-path mono">{scan.path}</div>
              </div>
              <span className="spacer" />
              <span className="badge info">
                {scan.format.name} {scan.format.version ?? ""}
              </span>
              {scan.quantization.label && <span className="badge">{scan.quantization.label}</span>}
              <span className="badge">{fmtBytes(scan.total_size_bytes)}</span>
              <span className={`badge ${scan.integrity.valid ? "ok" : "bad"}`}>{scan.integrity.valid ? "integrity ok" : "integrity errors"}</span>
            </>
          ) : (
            <div className="dim">No model loaded — import a GGUF file or a Hugging Face directory to begin.</div>
          )}
        </header>
        <div className="content">{body}</div>
      </main>
      {error && (
        <div className="toast" role="alert">
          <Callout kind="bad" title="Error:">
            {error}{" "}
            <button className="btn ghost sm" onClick={() => setError(null)}>
              dismiss
            </button>
          </Callout>
        </div>
      )}
    </div>
  );
}

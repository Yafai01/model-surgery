import { useCallback, useEffect, useRef, useState } from "react";
import { BROWSER_EDITION, api } from "./api";
import { Callout } from "./components/common";
import ArchitecturePage from "./components/ArchitecturePage";
import ErrorBoundary from "./components/ErrorBoundary";
import HistoryPage from "./components/HistoryPage";
import ImportPage from "./components/ImportPage";
import InferencePage from "./components/InferencePage";
import MemoryLab from "./components/MemoryLab";
import MetadataPage from "./components/MetadataPage";
import Overview from "./components/Overview";
import QuantizePage from "./components/QuantizePage";
import ReportsPage from "./components/ReportsPage";
import SettingsPage from "./components/SettingsPage";
import TensorTable from "./components/TensorTable";
import { fmtBytes } from "./format";
import { getHubToken } from "./hubSession";
import { type Prefs, applyTheme, loadPrefs, savePrefs } from "./prefs";
import type { ScanResult, ScanSummary } from "./types";

export type Tab =
  | "import"
  | "overview"
  | "architecture"
  | "tensors"
  | "metadata"
  | "inference"
  | "memory"
  | "quantize"
  | "reports"
  | "history"
  | "settings";

interface NavItem {
  id: Tab;
  label: string;
  icon: string;
  needsScan?: boolean;
}

const NAV: { group: string; items: NavItem[] }[] = [
  { group: "Model", items: [{ id: "import", label: "Import", icon: "⇪" }] },
  {
    group: "Architecture",
    items: [
      { id: "overview", label: "Overview", icon: "◉", needsScan: true },
      { id: "architecture", label: "Architecture", icon: "▤", needsScan: true },
      { id: "tensors", label: "Tensors", icon: "▦", needsScan: true },
      { id: "metadata", label: "Metadata", icon: "≡", needsScan: true },
    ],
  },
  {
    group: "Run & measure",
    items: [
      { id: "inference", label: "Inference", icon: "▷", needsScan: true },
      { id: "memory", label: "Performance", icon: "◫", needsScan: true },
      { id: "quantize", label: "Quantize", icon: "⚙", needsScan: true },
    ],
  },
  {
    group: "Records",
    items: [
      { id: "reports", label: "Compare & export", icon: "⎘" },
      { id: "history", label: "History", icon: "⟲" },
      { id: "settings", label: "Settings", icon: "⚙︎" },
    ],
  },
];
const ALL_ITEMS = NAV.flatMap((g) => g.items);

function Logo() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <g fill="none" strokeWidth="2.2" strokeLinecap="round" stroke="#fff">
        <path d="M8 22h16M8 16h16M8 10h16" opacity="0.9" />
        <path d="M12 7v18" />
      </g>
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("import");
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [history, setHistory] = useState<ScanSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [prefs, setPrefsState] = useState<Prefs>(loadPrefs);
  const [hubBusy, setHubBusy] = useState<string | null>(null);

  // Every navigation-changing load gets a ticket; a result whose ticket is stale (the user has opened something
  // else since) is ignored instead of replacing the current model.
  const loadSeq = useRef(0);
  const tabRef = useRef(tab);
  tabRef.current = tab;

  const setPrefs = useCallback((p: Prefs) => {
    setPrefsState(p);
    savePrefs(p);
    applyTheme(p.theme);
  }, []);

  const refreshHistory = useCallback(() => {
    api.scans().then(setHistory).catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(refreshHistory, [refreshHistory]);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 9000);
    return () => clearTimeout(t);
  }, [error]);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 6000);
    return () => clearTimeout(t);
  }, [notice]);

  const navigate = useCallback((t: Tab) => {
    loadSeq.current++; // explicit navigation wins over any load still in flight
    setTab(t);
  }, []);

  const onScanned = useCallback(
    (r: ScanResult) => {
      refreshHistory();
      if (tabRef.current !== "import") {
        // The user left the Import page while this scan was running: keep what they are looking at.
        setNotice(`Scan of ${r.name} finished and was saved to History.`);
        return;
      }
      loadSeq.current++;
      setScan(r);
      setTab("overview");
    },
    [refreshHistory],
  );

  const inspectHub = useCallback(
    async (repo: string, file: string) => {
      const ticket = ++loadSeq.current;
      setHubBusy(file);
      try {
        const r = await api.scanHub(repo, { file, token: getHubToken() });
        refreshHistory();
        if (ticket !== loadSeq.current) return setNotice(`Scan of ${r.name} finished and was saved to History.`);
        setScan(r);
        setTab("overview");
      } catch (e) {
        if (ticket === loadSeq.current) setError((e as Error).message);
      } finally {
        setHubBusy(null);
      }
    },
    [refreshHistory],
  );

  const openScan = useCallback(async (id: string, nextTab: Tab = "overview") => {
    const ticket = ++loadSeq.current;
    try {
      const r = await api.getScan(id);
      if (ticket !== loadSeq.current) return;
      setScan(r);
      setTab(nextTab);
    } catch (e) {
      if (ticket === loadSeq.current) setError((e as Error).message);
    }
  }, []);

  const onDeleted = useCallback((id: string) => {
    setScan((cur) => {
      if (cur?.id !== id) return cur;
      if (ALL_ITEMS.find((i) => i.id === tabRef.current)?.needsScan) setTab("history");
      return null;
    });
  }, []);

  const body = (() => {
    if (tab === "import") return <ImportPage onScanned={onScanned} onError={setError} recent={history} onOpen={openScan} />;
    if (tab === "history") return <HistoryPage history={history} onOpen={openScan} onChanged={refreshHistory} onDeleted={onDeleted} onError={setError} />;
    if (tab === "reports") return <ReportsPage scan={scan} history={history} onError={setError} />;
    if (tab === "settings") return <SettingsPage prefs={prefs} onPrefs={setPrefs} onError={setError} />;
    if (!scan) return null;
    switch (tab) {
      case "overview":
        return <Overview scan={scan} onNavigate={navigate} onHubInspect={inspectHub} hubBusy={hubBusy} />;
      case "architecture":
        return <ArchitecturePage key={scan.id} scan={scan} prefs={prefs} />;
      case "tensors":
        return <TensorTable scan={scan} />;
      case "metadata":
        return <MetadataPage scan={scan} />;
      case "inference":
        return <InferencePage scan={scan} onError={setError} onNavigate={navigate} />;
      case "memory":
        return <MemoryLab scan={scan} onError={setError} />;
      case "quantize":
        return <QuantizePage scan={scan} onError={setError} onOpenScan={openScan} onHistoryChanged={refreshHistory} />;
    }
    return null;
  })();

  const navButton = (it: NavItem, cls: string) => (
    <button
      key={it.id}
      className={`${cls} ${tab === it.id ? "active" : ""}`}
      disabled={it.needsScan && !scan}
      onClick={() => navigate(it.id)}
      aria-current={tab === it.id ? "page" : undefined}
      title={it.needsScan && !scan ? "Import a model first" : undefined}
    >
      {cls === "nav-item" && (
        <span className="nav-icon" aria-hidden>
          {it.icon}
        </span>
      )}
      {it.label}
    </button>
  );

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Logo />
          <div>
            <div className="brand-name">Model Surgery</div>
            <div className="brand-sub">inspect · run · measure</div>
          </div>
        </div>
        <nav aria-label="Main">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="nav-group">{g.group}</div>
              {g.items.map((it) => navButton(it, "nav-item"))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          {BROWSER_EDITION ? (
            <>
              Browser edition · files never leave your computer
              <br />
              <a href="https://github.com/Yafai01/model-surgery" target="_blank" rel="noreferrer">
                GitHub · local app with inference & quantization
              </a>
            </>
          ) : (
            <>
              Local app · nothing is uploaded
              <br />
              Model files are never executed by Model Surgery.
            </>
          )}
        </div>
      </aside>
      <main className="main">
        <nav className="mobile-nav" aria-label="Main (compact)">
          {ALL_ITEMS.map((it) => navButton(it, ""))}
        </nav>
        <header className="topbar">
          {scan ? (
            <>
              <div style={{ minWidth: 0 }}>
                <div className="topbar-title">{scan.name}</div>
                <div className="topbar-path mono">{scan.path}</div>
              </div>
              <span className="spacer" />
              {hubBusy && <span className="small dim">inspecting {hubBusy.split("/").pop()}…</span>}
              {scan.hub && <span className="badge warn">🤗 remote</span>}
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
        <div className="content">
          <ErrorBoundary key={`${tab}:${scan?.id ?? ""}`} area={ALL_ITEMS.find((i) => i.id === tab)?.label}>
            {body}
          </ErrorBoundary>
        </div>
      </main>
      {(error || notice) && (
        <div className="toast" role={error ? undefined : "status"}>
          {error ? (
            <Callout kind="bad" title="Error:">
              {error}{" "}
              <button className="btn ghost sm" onClick={() => setError(null)}>
                dismiss
              </button>
            </Callout>
          ) : (
            <Callout kind="ok">
              {notice}{" "}
              <button className="btn ghost sm" onClick={() => setNotice(null)}>
                dismiss
              </button>
            </Callout>
          )}
        </div>
      )}
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";
import { BROWSER_EDITION, api } from "../api";
import { fmtBytes, fmtTime } from "../format";
import type { Prefs, RenderQuality, Theme } from "../prefs";
import type { RuntimeStatus, SystemInfo, ToolsReport } from "../types";
import { Callout, Spinner } from "./common";

interface Props {
  prefs: Prefs;
  onPrefs: (p: Prefs) => void;
  onError: (m: string) => void;
}

function useOnline() {
  const [online, setOnline] = useState(typeof navigator === "undefined" ? true : navigator.onLine);
  useEffect(() => {
    const on = () => setOnline(true);
    const off = () => setOnline(false);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", off);
    };
  }, []);
  return online;
}

function webglInfo(): string {
  try {
    const c = document.createElement("canvas");
    const gl = (c.getContext("webgl2") || c.getContext("webgl")) as WebGLRenderingContext | null;
    if (!gl) return "unavailable (the 2D view is used)";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const r = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    gl.getExtension("WEBGL_lose_context")?.loseContext(); // release the probe context immediately
    return `${c.getContext("webgl2") ? "WebGL 2" : "WebGL"} · ${r}`;
  } catch {
    return "unavailable";
  }
}

export default function SettingsPage({ prefs, onPrefs, onError }: Props) {
  const [tools, setTools] = useState<ToolsReport | null>(null);
  const [rt, setRt] = useState<RuntimeStatus | null>(null);
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const online = useOnline();
  const [gl] = useState(webglInfo);

  const load = useCallback(
    async (refresh: boolean) => {
      if (BROWSER_EDITION) return;
      setBusy(true);
      try {
        const [t, r, s] = await Promise.all([api.tools(refresh), api.runtime(), api.system()]);
        setTools(t);
        setRt(r);
        setSys(s);
      } catch (e) {
        onError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [onError],
  );
  useEffect(() => {
    load(false);
  }, [load]);

  const set = <K extends keyof Prefs>(k: K, v: Prefs[K]) => onPrefs({ ...prefs, [k]: v });

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Settings</h2>
          <p>Display preferences are stored in this browser. Tool locations are server-side configuration and cannot be changed from the page.</p>
        </div>
      </div>

      <div className="grid grid-2" style={{ alignItems: "start" }}>
        <div className="card">
          <div className="card-head">
            <h3>Appearance</h3>
          </div>
          <div className="col">
            <label className="field">
              Theme
              <select value={prefs.theme} onChange={(e) => set("theme", e.target.value as Theme)}>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">Match the system</option>
              </select>
            </label>
            <label className="field">
              3D rendering quality
              <select value={prefs.renderQuality} onChange={(e) => set("renderQuality", e.target.value as RenderQuality)}>
                <option value="high">High — antialiasing, full pixel ratio, edge outlines</option>
                <option value="balanced">Balanced — antialiasing, pixel ratio ≤ 1.5</option>
                <option value="low">Low — no antialiasing, pixel ratio 1, no outlines (laptops on battery, Jetson)</option>
              </select>
            </label>
            <label className="check">
              <input type="checkbox" checked={prefs.showLayerLabels} onChange={(e) => set("showLayerLabels", e.target.checked)} /> Layer labels in the 3D view
            </label>
            <div className="small dim">Graphics: {gl}</div>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <h3>Network & privacy</h3>
            <span className={`badge ${online ? "ok" : "warn"}`}>{online ? "browser reports online" : "browser reports offline"}</span>
          </div>
          <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
            <li>Model files you pick are read in place {BROWSER_EDITION ? "by this page" : "by the local server"}; they are never uploaded.</li>
            <li>The only outbound requests go to huggingface.co, and only when you paste a Hub link.</li>
            <li>No analytics, web fonts or CDN assets are loaded.</li>
            <li>{BROWSER_EDITION ? "Scan history is stored in this browser (IndexedDB)." : "Scan history is stored in a local SQLite file."}</li>
            <li>Offline: inspection, the architecture views, estimates, comparison and {BROWSER_EDITION ? "exports" : "local inference"} work without a network connection once the app is loaded.</li>
          </ul>
        </div>
      </div>

      {BROWSER_EDITION ? (
        <Callout kind="info" title="Browser edition.">
          There is no runtime or tool detection in the hosted page: it can only read file headers. Run the local app (see the README) for llama.cpp inference, measurements and quantization.
        </Callout>
      ) : (
        <>
          <div className="card">
            <div className="card-head">
              <h3>Runtimes & tools</h3>
              <button className="btn sm" onClick={() => load(true)} disabled={busy}>
                {busy ? <Spinner /> : "Re-detect"}
              </button>
            </div>
            {!tools ? (
              <Spinner />
            ) : (
              <table className="small">
                <tbody>
                  <ToolRow name="llama-server (inference)" ok={!!tools.server?.available} detail={tools.server?.available ? `${tools.server.version} · ${tools.server.path}` : tools.server?.detail} />
                  <ToolRow name="llama-quantize" ok={tools.quantize.available} detail={tools.quantize.available ? `${tools.quantize.version ?? ""} · ${tools.quantize.path}` : tools.quantize.detail} />
                  <ToolRow name="convert_hf_to_gguf.py" ok={tools.convert.available} detail={tools.convert.available ? tools.convert.script : tools.convert.detail} />
                  <tr>
                    <td>Runtime state</td>
                    <td colSpan={2}>
                      {rt?.state}
                      {rt?.model_name ? ` · ${rt.model_name}` : ""}
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
            {tools && (
              <pre className="argv mono small" style={{ marginTop: 10 }}>
                {tools.configuration_help.join("\n")}
              </pre>
            )}
            {tools && <p className="small faint">Checked {fmtTime(new Date(tools.checked_at * 1000).toISOString())}.</p>}
          </div>

          {sys && (
            <div className="card">
              <div className="card-head">
                <h3>This machine</h3>
                <span className="small faint">Model Surgery {sys.version}</span>
              </div>
              <div className="kv-grid">
                <span>Platform</span>
                <b>
                  {sys.platform} · {sys.cpu_count ?? "?"} logical CPUs
                </b>
                <span>Physical memory</span>
                <b>
                  {fmtBytes(sys.memory.total_bytes)} total · {fmtBytes(sys.memory.available_bytes)} available <span className="badge prov verified">measured</span>
                </b>
                <span>Data directory</span>
                <b className="mono">{sys.data_dir}</b>
                <span>Outputs</span>
                <b className="mono">{sys.outputs_dir}</b>
                <span>Allowed model roots</span>
                <b className="mono">{sys.allowed_roots.length ? sys.allowed_roots.join(" · ") : "any local path (set MODEL_SURGERY_ALLOWED_ROOTS to restrict)"}</b>
                <span>llama.cpp directory</span>
                <b className="mono">{sys.llama_cpp_bin_dir ?? "not configured (PATH is searched)"}</b>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ToolRow({ name, ok, detail }: { name: string; ok: boolean; detail: string | null | undefined }) {
  return (
    <tr>
      <td className="nowrap">{name}</td>
      <td>
        <span className={`badge ${ok ? "ok" : ""}`}>{ok ? "available" : "not found"}</span>
      </td>
      <td className="mono break">{detail ?? ""}</td>
    </tr>
  );
}

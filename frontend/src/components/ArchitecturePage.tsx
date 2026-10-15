import { Suspense, lazy, useMemo, useState } from "react";
import { COMPONENT_COLORS, componentColor, fmtBytes, fmtCount, shapeStr } from "../format";
import type { ComponentSummary, LayerSummary, ScanResult, StackSummary, TensorRecord } from "../types";
import { Callout, Prov, Spinner } from "./common";

const Architecture3D = lazy(() => import("./Architecture3D"));

export type Selection = { kind: "layer"; index: number } | { kind: "global"; component: string } | null;

function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

const GLOBAL_ORDER_TOP = ["embedding"];
const GLOBAL_ORDER_BOTTOM = ["norm", "output", "other"];

export default function ArchitecturePage({ scan }: { scan: ScanResult }) {
  const webgl = useMemo(hasWebGL, []);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const stacks = scan.architecture.stacks;
  const [stackIdx, setStackIdx] = useState(0);
  const [sel, setSel] = useState<Selection>(stacks.length ? { kind: "layer", index: stacks[0].layers[0]?.index ?? 0 } : null);
  const stack: StackSummary | undefined = stacks[stackIdx];
  const tensorsByName = useMemo(() => new Map(scan.tensors.map((t) => [t.name, t])), [scan.tensors]);

  const selectedLayer = sel?.kind === "layer" ? stack?.layers.find((l) => l.index === sel.index) : undefined;
  const selectedGlobal = sel?.kind === "global" ? scan.architecture.globals.find((g) => g.component === sel.component) : undefined;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Architecture explorer</h2>
          <p>
            {scan.architecture.family ?? "Unknown family"} <Prov p={scan.architecture.family_provenance} /> ·{" "}
            {stacks.length ? `${stacks.length} layer stack(s)` : "no repeating layer pattern recognised"}
          </p>
        </div>
        <div className="row">
          {stacks.length > 1 && (
            <select value={stackIdx} onChange={(e) => setStackIdx(Number(e.target.value))} aria-label="Layer stack">
              {stacks.map((s, i) => (
                <option key={s.prefix} value={i}>
                  {s.prefix} · {s.n_layers} layers{s.primary ? " (primary)" : ""}
                </option>
              ))}
            </select>
          )}
          <div className="seg" role="tablist">
            <button className={view === "2d" ? "on" : ""} onClick={() => setView("2d")}>
              2D
            </button>
            <button className={view === "3d" ? "on" : ""} onClick={() => setView("3d")} disabled={!webgl} title={webgl ? "" : "WebGL is not available in this browser"}>
              3D
            </button>
          </div>
        </div>
      </div>

      <Callout>
        <strong>What is verified here:</strong> tensor names, shapes, dtypes and byte sizes <Prov p="verified" /> come straight from the file header.{" "}
        <strong>What is inferred:</strong> grouping tensors into layers and labelling them attention / FFN / norm <Prov p="inferred" /> relies on naming
        conventions (llama.cpp <code>blk.N.*</code>, Hugging Face <code>model.layers.N.*</code>). Unrecognised tensors are shown as “other”, not guessed.
      </Callout>

      {stack && <Anatomy layer={stack.layers[0]} uniform={scan.architecture.uniform_layers} />}

      <div className="arch-layout">
        <div className="card">
          <div className="card-head">
            <h3>{stack ? `${stack.prefix} — ${stack.n_layers} layers, ${fmtBytes(stack.n_bytes)}` : "Tensors"}</h3>
            <Legend />
          </div>
          {view === "3d" && webgl && stack ? (
            <Suspense
              fallback={
                <div className="canvas-3d row" style={{ justifyContent: "center" }}>
                  <Spinner />
                </div>
              }
            >
              <Architecture3D stack={stack} globals={scan.architecture.globals} selection={sel} onSelect={setSel} />
              <p className="small faint" style={{ marginTop: 8 }}>
                Drag to orbit · scroll to zoom · click a block to inspect. Block width is proportional to bytes (minimum width for visibility).
              </p>
            </Suspense>
          ) : (
            <Stack2D stack={stack} globals={scan.architecture.globals} selection={sel} onSelect={setSel} />
          )}
          {!webgl && <p className="small faint">WebGL unavailable — showing the 2D view.</p>}
        </div>

        <div className="card" style={{ alignSelf: "start", position: "sticky", top: 0 }}>
          {selectedLayer && <LayerDetails layer={selectedLayer} prefix={stack!.prefix} tensors={tensorsByName} />}
          {selectedGlobal && <GlobalDetails comp={selectedGlobal} tensors={tensorsByName} />}
          {!selectedLayer && !selectedGlobal && <p className="dim">Select a layer or block to see its tensors.</p>}
        </div>
      </div>

      {scan.architecture.notes.length > 0 && (
        <Callout kind="warn" title="Reconstruction notes:">
          <ul>
            {scan.architecture.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </Callout>
      )}
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      {Object.entries(COMPONENT_COLORS).map(([k, c]) => (
        <span key={k}>
          <i style={{ background: c }} />
          {k}
        </span>
      ))}
    </div>
  );
}

function Bar({ comps, scale }: { comps: ComponentSummary[]; scale: number }) {
  const total = comps.reduce((a, c) => a + c.n_bytes, 0) || 1;
  return (
    <div className="layer-bar" style={{ width: `${Math.max(scale * 100, 4)}%` }}>
      {comps.map((c) => (
        <span
          key={c.component}
          title={`${c.component}: ${fmtBytes(c.n_bytes)} (${c.tensors.length} tensors)`}
          style={{ width: `${(c.n_bytes / total) * 100}%`, background: componentColor(c.component) }}
        />
      ))}
    </div>
  );
}

function Stack2D({
  stack,
  globals,
  selection,
  onSelect,
}: {
  stack: StackSummary | undefined;
  globals: ComponentSummary[];
  selection: Selection;
  onSelect: (s: Selection) => void;
}) {
  const maxBytes = Math.max(1, ...(stack?.layers.map((l) => l.n_bytes) ?? []), ...globals.map((g) => g.n_bytes));
  const top = globals.filter((g) => GLOBAL_ORDER_TOP.includes(g.component));
  const bottom = globals.filter((g) => GLOBAL_ORDER_BOTTOM.includes(g.component)).sort((a, b) => GLOBAL_ORDER_BOTTOM.indexOf(a.component) - GLOBAL_ORDER_BOTTOM.indexOf(b.component));
  const globalRow = (g: ComponentSummary) => (
    <div
      key={g.component}
      className={`layer-row global-row ${selection?.kind === "global" && selection.component === g.component ? "selected" : ""}`}
      onClick={() => onSelect({ kind: "global", component: g.component })}
    >
      <span className="layer-label">{g.component}</span>
      <Bar comps={[g]} scale={g.n_bytes / maxBytes} />
      <span className="small faint right">{fmtBytes(g.n_bytes)}</span>
    </div>
  );
  return (
    <div className="layer-stack">
      <div className="small faint">input ↓</div>
      {top.map(globalRow)}
      {stack?.layers.map((l) => (
        <div
          key={l.index}
          className={`layer-row ${selection?.kind === "layer" && selection.index === l.index ? "selected" : ""}`}
          onClick={() => onSelect({ kind: "layer", index: l.index })}
        >
          <span className="layer-label">L{l.index}</span>
          <Bar comps={l.components} scale={l.n_bytes / maxBytes} />
          <span className="small faint right">{fmtBytes(l.n_bytes)}</span>
        </div>
      ))}
      {bottom.map(globalRow)}
      <div className="small faint">output ↓</div>
    </div>
  );
}

function Anatomy({ layer, uniform }: { layer: LayerSummary | undefined; uniform: boolean }) {
  if (!layer) return null;
  const by = new Map(layer.components.map((c) => [c.component, c]));
  const norm = by.get("norm");
  const nodes: { title: string; comp: string; sub: string }[] = [];
  const normRoles = norm?.roles ?? [];
  if (normRoles.includes("attn_norm")) nodes.push({ title: "Norm", comp: "norm", sub: "attn_norm" });
  if (by.get("attention")) nodes.push({ title: "Attention", comp: "attention", sub: by.get("attention")!.roles.join(" · ") || "—" });
  if (by.get("ssm")) nodes.push({ title: "SSM / recurrent", comp: "ssm", sub: `${by.get("ssm")!.tensors.length} tensors` });
  if (normRoles.includes("ffn_norm") || normRoles.includes("post_attn_norm")) nodes.push({ title: "Norm", comp: "norm", sub: "ffn_norm" });
  if (by.get("router")) nodes.push({ title: "Router", comp: "router", sub: "expert gating" });
  if (by.get("experts")) nodes.push({ title: "Experts", comp: "experts", sub: by.get("experts")!.roles.join(" · ") });
  if (by.get("ffn")) nodes.push({ title: by.get("experts") ? "Shared FFN" : "Feed-forward", comp: "ffn", sub: by.get("ffn")!.roles.join(" · ") || "—" });
  if (by.get("other")) nodes.push({ title: "Other", comp: "other", sub: `${by.get("other")!.tensors.length} unclassified` });
  return (
    <div className="card">
      <div className="card-head">
        <h3>
          Layer anatomy <Prov p="inferred" />
        </h3>
        <span className="small faint">
          {uniform ? "all layers share this structure" : "layers differ — this is layer " + layer.index}
        </span>
      </div>
      <div className="anatomy">
        <div className="anatomy-node">
          <b>hidden in</b>
          <span className="faint">residual</span>
        </div>
        {nodes.map((n, i) => (
          <div key={i} style={{ display: "contents" }}>
            <span className="anatomy-arrow">→</span>
            <div className="anatomy-node" style={{ borderColor: componentColor(n.comp) }}>
              <b style={{ color: componentColor(n.comp) }}>{n.title}</b>
              <span className="mono faint">{n.sub}</span>
            </div>
          </div>
        ))}
        <span className="anatomy-arrow">→</span>
        <div className="anatomy-node">
          <b>hidden out</b>
          <span className="faint">+ residual</span>
        </div>
      </div>
    </div>
  );
}

function TensorList({ names, tensors }: { names: string[]; tensors: Map<string, TensorRecord> }) {
  return (
    <div className="table-wrap" style={{ maxHeight: 260 }}>
      <table>
        <tbody>
          {names.map((n) => {
            const t = tensors.get(n);
            return (
              <tr key={n}>
                <td className="mono small break">
                  {n}
                  {t && (
                    <div className="faint">
                      {shapeStr(t.shape)} · {t.dtype}
                    </div>
                  )}
                </td>
                <td className="num small">{fmtBytes(t?.n_bytes)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function LayerDetails({ layer, prefix, tensors }: { layer: LayerSummary; prefix: string; tensors: Map<string, TensorRecord> }) {
  return (
    <div className="col" style={{ gap: 12 }}>
      <div>
        <h3>
          {prefix}.{layer.index}
        </h3>
        <div className="small dim">
          {fmtBytes(layer.n_bytes)} · {fmtCount(layer.n_elements)} params <Prov p="derived" />
        </div>
      </div>
      <div className="row" style={{ gap: 4 }}>
        {Object.entries(layer.dtypes).map(([d, n]) => (
          <span key={d} className="chip">
            {d} × {n}
          </span>
        ))}
      </div>
      {layer.components.map((c) => (
        <div key={c.component} className="col" style={{ gap: 6 }}>
          <div className="row" style={{ gap: 8 }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: componentColor(c.component) }} />
            <strong>{c.component}</strong>
            <Prov p="inferred" />
            <span className="spacer" />
            <span className="small faint">{fmtBytes(c.n_bytes)}</span>
          </div>
          <TensorList names={c.tensors} tensors={tensors} />
        </div>
      ))}
    </div>
  );
}

function GlobalDetails({ comp, tensors }: { comp: ComponentSummary; tensors: Map<string, TensorRecord> }) {
  return (
    <div className="col" style={{ gap: 12 }}>
      <div>
        <h3 style={{ color: componentColor(comp.component) }}>{comp.component}</h3>
        <div className="small dim">
          {fmtBytes(comp.n_bytes)} · {fmtCount(comp.n_elements)} params · roles: {comp.roles.join(", ") || "—"} <Prov p="inferred" />
        </div>
      </div>
      <TensorList names={comp.tensors} tensors={tensors} />
    </div>
  );
}

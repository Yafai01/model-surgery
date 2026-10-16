import { type KeyboardEvent, Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { buildArchitectureGraph, type GraphNode, type LayerGraph } from "../engine/graph";
import { COMPONENT_COLORS, componentColor, download, fmtBytes, fmtCount, shapeStr } from "../format";
import type { Prefs } from "../prefs";
import type { ComponentSummary, LayerSummary, ScanResult, StackSummary, TensorRecord } from "../types";
import { Callout, Prov, Spinner } from "./common";
import ErrorBoundary from "./ErrorBoundary";

const Architecture3D = lazy(() => import("./Architecture3D"));

export type Selection = { kind: "layer"; index: number } | { kind: "global"; component: string } | null;

// Probed once per page load; the probe context is released immediately so it does not count against the
// browser's limit on live WebGL contexts.
let webglCache: boolean | null = null;
function hasWebGL(): boolean {
  if (webglCache !== null) return webglCache;
  try {
    const c = document.createElement("canvas");
    const gl = (c.getContext("webgl2") || c.getContext("webgl")) as WebGLRenderingContext | null;
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    webglCache = !!gl;
  } catch {
    webglCache = false;
  }
  return webglCache;
}

const GLOBAL_ORDER_TOP = ["embedding"];
const GLOBAL_ORDER_BOTTOM = ["norm", "output", "other"];
const ALL_COMPONENTS = Object.keys(COMPONENT_COLORS);

export default function ArchitecturePage({ scan, prefs }: { scan: ScanResult; prefs: Prefs }) {
  const webgl = useMemo(hasWebGL, []);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const stacks = scan.architecture.stacks;
  const [stackIdx, setStackIdx] = useState(0);
  const stack: StackSummary | undefined = stacks[stackIdx];
  const [sel, setSel] = useState<Selection>(stack?.layers[0] ? { kind: "layer", index: stack.layers[0].index } : null);
  const [query, setQuery] = useState("");
  const [visible, setVisible] = useState<Set<string>>(() => new Set(ALL_COMPONENTS));
  const tensorsByName = useMemo(() => new Map(scan.tensors.map((t) => [t.name, t])), [scan.tensors]);
  const graph = useMemo(() => buildArchitectureGraph(scan, stack), [scan, stack]);

  // Search tensor names (and component/role) across the selected stack.
  const search = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q || !stack) return null;
    const hits = scan.tensors.filter((t) => t.stack === stack.prefix && `${t.name} ${t.component} ${t.role ?? ""} ${t.dtype}`.toLowerCase().includes(q));
    const globals = scan.tensors.filter((t) => t.stack === null && `${t.name} ${t.component} ${t.role ?? ""} ${t.dtype}`.toLowerCase().includes(q));
    return { layers: new Set(hits.map((t) => t.layer!)), n: hits.length + globals.length, globals };
  }, [query, scan.tensors, stack]);

  const changeStack = (i: number) => {
    setStackIdx(i);
    const first = stacks[i]?.layers[0];
    setSel(first ? { kind: "layer", index: first.index } : null); // a layer index from another stack would be meaningless
  };

  const layerIdx = stack?.layers.map((l) => l.index) ?? [];
  const step = (d: number) => {
    const cur = sel?.kind === "layer" ? layerIdx.indexOf(sel.index) : -1;
    const next = layerIdx[Math.min(layerIdx.length - 1, Math.max(0, cur + d))];
    if (next !== undefined) setSel({ kind: "layer", index: next });
  };
  const jumpToFirstMatch = () => {
    const first = layerIdx.find((i) => search?.layers.has(i));
    if (first !== undefined) setSel({ kind: "layer", index: first });
  };

  const selectedLayer = sel?.kind === "layer" ? stack?.layers.find((l) => l.index === sel.index) : undefined;
  const selectedGlobal = sel?.kind === "global" ? scan.architecture.globals.find((g) => g.component === sel.component) : undefined;
  const selectedGraph = selectedLayer ? graph.layers.find((l) => l.index === selectedLayer.index) : undefined;
  const toggle = (c: string) =>
    setVisible((v) => {
      const n = new Set(v);
      if (n.has(c)) n.delete(c);
      else n.add(c);
      return n.size ? n : new Set(ALL_COMPONENTS);
    });

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
            <select value={stackIdx} onChange={(e) => changeStack(Number(e.target.value))} aria-label="Layer stack">
              {stacks.map((s, i) => (
                <option key={s.prefix} value={i}>
                  {s.prefix} · {s.n_layers} layers{s.primary ? " (primary)" : ""}
                </option>
              ))}
            </select>
          )}
          <div className="seg" role="radiogroup" aria-label="View">
            <button role="radio" aria-checked={view === "2d"} className={view === "2d" ? "on" : ""} onClick={() => setView("2d")}>
              2D
            </button>
            <button role="radio" aria-checked={view === "3d"} className={view === "3d" ? "on" : ""} onClick={() => setView("3d")} disabled={!webgl} title={webgl ? "" : "WebGL is not available in this browser"}>
              3D
            </button>
          </div>
          <button className="btn sm" onClick={() => download(`${scan.name.replace(/[^\w.-]+/g, "-")}.architecture.json`, JSON.stringify(graph, null, 2), "application/json")}>
            ⤓ Graph JSON
          </button>
        </div>
      </div>

      <Callout>
        <strong>Verified:</strong> tensor names, shapes, dtypes and byte sizes <Prov p="verified" /> come straight from the file header. <strong>Inferred:</strong> grouping tensors
        into layers and labelling them attention / FFN / norm <Prov p="inferred" /> relies on naming conventions. Residual connections and the input/output nodes are{" "}
        <em>illustrative</em>: they hold no tensors. Unrecognised tensors are listed, not guessed.
      </Callout>

      {stack && (
        <div className="card arch-tools">
          <div className="row" style={{ gap: 8 }}>
            <input
              type="search"
              placeholder="Search tensors (e.g. attn_k, q_proj, Q4_K, experts)…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && jumpToFirstMatch()}
              aria-label="Search tensors"
              style={{ flex: "1 1 260px" }}
            />
            {search && (
              <span className="small dim" role="status">
                {search.n} tensor(s) in {search.layers.size} layer(s){search.globals.length ? ` + ${search.globals.length} global` : ""}
              </span>
            )}
            <span className="spacer" />
            <div className="row" style={{ gap: 4 }} aria-label="Layer navigation">
              <button className="btn sm" onClick={() => step(-1)} aria-label="Previous layer">
                ↑
              </button>
              <label className="small dim">
                layer{" "}
                <input
                  type="number"
                  className="narrow"
                  min={layerIdx[0]}
                  max={layerIdx[layerIdx.length - 1]}
                  value={sel?.kind === "layer" ? sel.index : ""}
                  onChange={(e) => {
                    const n = Number(e.target.value);
                    if (layerIdx.includes(n)) setSel({ kind: "layer", index: n });
                  }}
                  aria-label="Go to layer"
                />{" "}
                of {stack.n_layers}
              </label>
              <button className="btn sm" onClick={() => step(1)} aria-label="Next layer">
                ↓
              </button>
            </div>
          </div>
          <div className="row" style={{ gap: 6, marginTop: 8 }} role="group" aria-label="Show components">
            <span className="small faint">Show:</span>
            {ALL_COMPONENTS.map((c) => (
              <button key={c} className={`chip toggle ${visible.has(c) ? "on" : ""}`} aria-pressed={visible.has(c)} onClick={() => toggle(c)}>
                <i style={{ background: componentColor(c) }} />
                {c}
              </button>
            ))}
            {visible.size < ALL_COMPONENTS.length && (
              <button className="btn ghost sm" onClick={() => setVisible(new Set(ALL_COMPONENTS))}>
                show all
              </button>
            )}
          </div>
        </div>
      )}

      <div className="arch-layout">
        <div className="card">
          <div className="card-head">
            <h3>{stack ? `${stack.prefix} — ${stack.n_layers} layers, ${fmtBytes(stack.n_bytes)}` : "Tensors"}</h3>
          </div>
          {view === "3d" && webgl && stack ? (
            <ErrorBoundary
              area="3D view"
              fallback={(e) => (
                <>
                  <Callout kind="warn" title="3D view unavailable:">
                    {e.message}. Showing the 2D view.
                  </Callout>
                  <Stack2D stack={stack} globals={scan.architecture.globals} selection={sel} onSelect={setSel} matches={search?.layers ?? null} visible={visible} onStep={step} />
                </>
              )}
            >
              <Suspense
                fallback={
                  <div className="canvas-3d row" style={{ justifyContent: "center" }}>
                    <Spinner />
                  </div>
                }
              >
                <Architecture3D
                  stack={stack}
                  globals={scan.architecture.globals}
                  selection={sel}
                  onSelect={setSel}
                  matches={search?.layers ?? null}
                  visible={visible}
                  quality={prefs.renderQuality}
                  showLabels={prefs.showLayerLabels}
                />
                <p className="small faint" style={{ marginTop: 8 }}>
                  Schematic: block width is proportional to the bytes stored for that component (with a minimum width for visibility); heights, depth and spacing are illustrative.
                </p>
              </Suspense>
            </ErrorBoundary>
          ) : (
            <Stack2D stack={stack} globals={scan.architecture.globals} selection={sel} onSelect={setSel} matches={search?.layers ?? null} visible={visible} onStep={step} />
          )}
          {!webgl && <p className="small faint">WebGL unavailable — showing the 2D view.</p>}
        </div>

        <div className="card details-panel">
          {selectedLayer && <LayerDetails layer={selectedLayer} prefix={stack!.prefix} tensors={tensorsByName} query={query} />}
          {selectedGlobal && <GlobalDetails comp={selectedGlobal} tensors={tensorsByName} query={query} />}
          {!selectedLayer && !selectedGlobal && <p className="dim">Select a layer or block to see its tensors.</p>}
        </div>
      </div>

      {selectedGraph && <LayerFlow layer={selectedGraph} uniform={scan.architecture.uniform_layers} />}

      {(scan.architecture.notes.length > 0 || graph.unrepresented.length > 0) && (
        <Callout kind="warn" title="Reconstruction notes:">
          <ul>
            {scan.architecture.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
            {graph.unrepresented.length > 0 && (
              <li>
                {graph.unrepresented.length} tensor(s) are not placed in the dataflow graph: <span className="mono">{graph.unrepresented.slice(0, 8).join(", ")}</span>
                {graph.unrepresented.length > 8 ? " …" : ""}
              </li>
            )}
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

function Bar({ comps, scale, visible }: { comps: ComponentSummary[]; scale: number; visible: Set<string> }) {
  const total = comps.reduce((a, c) => a + c.n_bytes, 0) || 1;
  return (
    <div className="layer-bar" style={{ width: `${Math.max(scale * 100, 4)}%` }}>
      {comps.map((c) => (
        <span
          key={c.component}
          title={`${c.component}: ${fmtBytes(c.n_bytes)} (${c.tensors.length} tensors)`}
          style={{ width: `${(c.n_bytes / total) * 100}%`, background: visible.has(c.component) ? componentColor(c.component) : "var(--border)" }}
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
  matches,
  visible,
  onStep,
}: {
  stack: StackSummary | undefined;
  globals: ComponentSummary[];
  selection: Selection;
  onSelect: (s: Selection) => void;
  matches: Set<number> | null;
  visible: Set<string>;
  onStep: (d: number) => void;
}) {
  const maxBytes = Math.max(1, ...(stack?.layers.map((l) => l.n_bytes) ?? []), ...globals.map((g) => g.n_bytes));
  const top = globals.filter((g) => GLOBAL_ORDER_TOP.includes(g.component));
  const bottom = globals.filter((g) => GLOBAL_ORDER_BOTTOM.includes(g.component)).sort((a, b) => GLOBAL_ORDER_BOTTOM.indexOf(a.component) - GLOBAL_ORDER_BOTTOM.indexOf(b.component));
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Keep the selected row in view when the selection changes (keyboard stepping, search, 3D clicks).
    listRef.current?.querySelector<HTMLElement>(".layer-row.selected")?.scrollIntoView({ block: "nearest" });
  }, [selection]);
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      onStep(e.key === "ArrowDown" ? 1 : -1);
    }
  };
  const globalRow = (g: ComponentSummary) => (
    <button
      key={g.component}
      className={`layer-row global-row ${selection?.kind === "global" && selection.component === g.component ? "selected" : ""} ${matches ? "dim-row" : ""}`}
      onClick={() => onSelect({ kind: "global", component: g.component })}
      aria-pressed={selection?.kind === "global" && selection.component === g.component}
    >
      <span className="layer-label">{g.component}</span>
      <Bar comps={[g]} scale={g.n_bytes / maxBytes} visible={visible} />
      <span className="small faint right">{fmtBytes(g.n_bytes)}</span>
    </button>
  );
  return (
    <div className="col" style={{ gap: 8 }}>
      <Legend />
      <div className="layer-stack" onKeyDown={onKey} ref={listRef}>
        <div className="small faint">input ↓</div>
        {top.map(globalRow)}
        {stack?.layers.map((l) => {
          const isSel = selection?.kind === "layer" && selection.index === l.index;
          return (
            <button
              key={l.index}
              className={`layer-row ${isSel ? "selected" : ""} ${matches && !matches.has(l.index) ? "dim-row" : ""} ${matches?.has(l.index) ? "match-row" : ""}`}
              onClick={() => onSelect({ kind: "layer", index: l.index })}
              aria-pressed={isSel}
            >
              <span className="layer-label">L{l.index}</span>
              <Bar comps={l.components} scale={l.n_bytes / maxBytes} visible={visible} />
              <span className="small faint right">{fmtBytes(l.n_bytes)}</span>
            </button>
          );
        })}
        {bottom.map(globalRow)}
        <div className="small faint">output ↓</div>
      </div>
    </div>
  );
}

/** Focused view of one layer's dataflow, from the normalized architecture graph. */
function LayerFlow({ layer, uniform }: { layer: LayerGraph; uniform: boolean }) {
  const byId = new Map(layer.nodes.map((n) => [n.id, n]));
  const residualTargets = new Set(layer.edges.filter((e) => e.kind === "residual").map((e) => e.to));
  const order = layer.nodes; // build order follows the dataflow
  return (
    <div className="card">
      <div className="card-head">
        <h3>
          Layer {layer.index} dataflow <Prov p="inferred" />
        </h3>
        <span className="small faint">
          {layer.layout === "sequential_prenorm"
            ? "pre-norm residual block"
            : layer.layout === "parallel"
              ? "parallel attention + MLP"
              : layer.layout === "ssm"
                ? "recurrent (SSM) block"
                : "layout not recognised"}
          {uniform ? " · all layers share this structure" : " · layers differ"}
        </span>
      </div>
      <div className="flow" role="list">
        {order.map((n, i) => (
          <div key={n.id} className="flow-step" role="listitem">
            {i > 0 && <span className="flow-arrow" aria-hidden>{residualTargets.has(n.id) ? "⊕" : "→"}</span>}
            <FlowNode
              node={n}
              residualFrom={layer.edges
                .filter((e) => e.kind === "residual" && e.to === n.id)
                .map((e) => (byId.get(e.from)?.kind === "residual_add" ? "the previous sum" : "the layer input"))}
            />
          </div>
        ))}
      </div>
      {layer.notes.map((n, i) => (
        <p key={i} className="small dim">
          {n}
        </p>
      ))}
      <p className="small faint">Dashed nodes are illustrative (no tensors in the file). Dimensions show the provenance of the hyperparameter they come from.</p>
    </div>
  );
}

function FlowNode({ node, residualFrom }: { node: GraphNode; residualFrom: string[] }) {
  const color = node.illustrative ? "var(--text-faint)" : componentColor(node.kind === "residual_add" ? "other" : node.kind);
  return (
    <div className={`flow-node ${node.illustrative ? "illustrative" : ""}`} style={{ borderColor: color }}>
      <b style={{ color: node.illustrative ? undefined : color }}>{node.label}</b>
      {node.tensors.length > 0 && (
        <span className="small faint">
          {node.tensors.length} tensor{node.tensors.length > 1 ? "s" : ""} · {fmtBytes(node.n_bytes)}
        </span>
      )}
      {residualFrom.length > 0 && <span className="small faint">adds the skip connection from {residualFrom.join(", ")}</span>}
      {node.dims
        .filter((d) => d.provenance !== "verified" || !d.value.startsWith("["))
        .slice(0, 4)
        .map((d) => (
          <span key={d.label} className="small mono">
            {d.label}: {d.value} <Prov p={d.provenance} />
          </span>
        ))}
    </div>
  );
}

function TensorList({ names, tensors, query }: { names: string[]; tensors: Map<string, TensorRecord>; query: string }) {
  const q = query.trim().toLowerCase();
  return (
    <div className="table-wrap" style={{ maxHeight: 260 }}>
      <table>
        <tbody>
          {names.map((n) => {
            const t = tensors.get(n);
            const hit = q && `${n} ${t?.component} ${t?.role ?? ""} ${t?.dtype}`.toLowerCase().includes(q);
            return (
              <tr key={n} className={hit ? "diff" : ""}>
                <td className="mono small break">
                  {n}
                  {t && (
                    <div className="faint">
                      {shapeStr(t.shape)} · {t.dtype}
                      {t.role ? ` · ${t.role}` : ""}
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

function LayerDetails({ layer, prefix, tensors, query }: { layer: LayerSummary; prefix: string; tensors: Map<string, TensorRecord>; query: string }) {
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
          <TensorList names={c.tensors} tensors={tensors} query={query} />
        </div>
      ))}
    </div>
  );
}

function GlobalDetails({ comp, tensors, query }: { comp: ComponentSummary; tensors: Map<string, TensorRecord>; query: string }) {
  return (
    <div className="col" style={{ gap: 12 }}>
      <div>
        <h3 style={{ color: componentColor(comp.component) }}>{comp.component}</h3>
        <div className="small dim">
          {fmtBytes(comp.n_bytes)} · {fmtCount(comp.n_elements)} params · roles: {comp.roles.join(", ") || "—"} <Prov p="inferred" />
        </div>
      </div>
      <TensorList names={comp.tensors} tensors={tensors} query={query} />
    </div>
  );
}

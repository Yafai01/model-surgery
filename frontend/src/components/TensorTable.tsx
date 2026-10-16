import { useMemo, useState } from "react";
import { componentColor, download, fmtBytes, fmtInt, shapeStr } from "../format";
import type { ScanResult, TensorRecord } from "../types";
import { Prov } from "./common";

type SortKey = "index" | "name" | "layer" | "component" | "dtype" | "n_elements" | "n_bytes" | "offset";
const PAGE = 100;

export default function TensorTable({ scan }: { scan: ScanResult }) {
  const [query, setQuery] = useState("");
  const [component, setComponent] = useState("");
  const [dtype, setDtype] = useState("");
  const [layer, setLayer] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "index", dir: 1 });
  const [page, setPage] = useState(0);

  const indexed = useMemo(() => scan.tensors.map((t, i) => ({ ...t, index: i })), [scan.tensors]);
  const components = useMemo(() => [...new Set(scan.tensors.map((t) => t.component))].sort(), [scan.tensors]);
  const dtypes = useMemo(() => [...new Set(scan.tensors.map((t) => t.dtype))].sort(), [scan.tensors]);
  const layers = useMemo(() => [...new Set(scan.tensors.map((t) => t.layer).filter((l): l is number => l !== null))].sort((a, b) => a - b), [scan.tensors]);

  const rows = useMemo(() => {
    const ql = query.trim().toLowerCase();
    const filtered = indexed.filter(
      (t) =>
        (!ql || t.name.toLowerCase().includes(ql)) &&
        (!component || t.component === component) &&
        (!dtype || t.dtype === dtype) &&
        (!layer || (layer === "none" ? t.layer === null : t.layer === Number(layer))),
    );
    const k = sort.key;
    return filtered.sort((a, b) => {
      const va = a[k as keyof typeof a] ?? -1;
      const vb = b[k as keyof typeof b] ?? -1;
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
  }, [indexed, query, component, dtype, layer, sort]);

  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const cur = Math.min(page, pages - 1);
  const shown = rows.slice(cur * PAGE, cur * PAGE + PAGE);
  const totalBytes = rows.reduce((a, t) => a + (t.n_bytes ?? 0), 0);
  const orderNote = scan.tensors[0]?.shape_order === "ggml" ? "GGML order: first dimension is contiguous (ne0)" : "row-major order (PyTorch convention)";

  const th = (key: SortKey, label: string, num = false) => (
    <th className={`sortable ${num ? "num" : ""}`} aria-sort={sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
      <button
        className="th-button"
        onClick={() => {
          setSort((s) => ({ key, dir: s.key === key ? ((-s.dir) as 1 | -1) : 1 }));
          setPage(0);
        }}
      >
        {label} {sort.key === key ? (sort.dir === 1 ? "▲" : "▼") : ""}
      </button>
    </th>
  );

  const exportFiltered = () => {
    const head = "name,shape,dtype,n_elements,n_bytes,layer,component,role";
    const esc = (s: string) => (/^[=+\-@]/.test(s) ? "'" + s : s).replace(/"/g, '""');
    const body = rows.map((t: TensorRecord) =>
      [`"${esc(t.name)}"`, `"${t.shape.join("x")}"`, t.dtype, t.n_elements, t.n_bytes ?? "", t.layer ?? "", t.component, t.role ?? ""].join(","),
    );
    download(`${scan.name}.tensors-filtered.csv`, [head, ...body].join("\n"), "text/csv");
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Tensors</h2>
          <p>
            {fmtInt(scan.tensors.length)} tensors. Name, shape, dtype, size and offset <Prov p="verified" /> · layer & component <Prov p="inferred" /> ·
            shapes in {orderNote}.
          </p>
        </div>
      </div>
      <div className="card">
        <div className="row" style={{ marginBottom: 12 }}>
          <input
            type="text"
            placeholder="Search tensor names…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(0);
            }}
            style={{ flex: 1, minWidth: 220 }}
            aria-label="Search tensors"
          />
          <select value={component} onChange={(e) => (setComponent(e.target.value), setPage(0))} aria-label="Component filter">
            <option value="">all components</option>
            {components.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <select value={dtype} onChange={(e) => (setDtype(e.target.value), setPage(0))} aria-label="dtype filter">
            <option value="">all dtypes</option>
            {dtypes.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
          <select value={layer} onChange={(e) => (setLayer(e.target.value), setPage(0))} aria-label="Layer filter">
            <option value="">all layers</option>
            <option value="none">non-layer</option>
            {layers.map((l) => (
              <option key={l} value={l}>
                layer {l}
              </option>
            ))}
          </select>
          <button className="btn sm" onClick={exportFiltered} disabled={!rows.length}>
            Export filtered CSV
          </button>
        </div>
        <div className="row small dim" style={{ marginBottom: 8 }}>
          {fmtInt(rows.length)} match{rows.length === 1 ? "" : "es"} · {fmtBytes(totalBytes)}
          <span className="spacer" />
          <div className="pager">
            <button className="btn sm" disabled={cur === 0} onClick={() => setPage(cur - 1)}>
              ‹
            </button>
            page {cur + 1} / {pages}
            <button className="btn sm" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)}>
              ›
            </button>
          </div>
        </div>
        <div className="table-wrap" style={{ maxHeight: "calc(100vh - 330px)" }}>
          <table>
            <thead>
              <tr>
                {th("index", "#", true)}
                {th("name", "Name")}
                {th("layer", "Layer", true)}
                {th("component", "Component")}
                {th("dtype", "dtype")}
                <th>Shape</th>
                {th("n_elements", "Elements", true)}
                {th("n_bytes", "Bytes", true)}
                {th("offset", "Offset", true)}
              </tr>
            </thead>
            <tbody>
              {shown.map((t) => (
                <tr key={t.name}>
                  <td className="num faint">{t.index}</td>
                  <td className="mono small break">{t.name}</td>
                  <td className="num">{t.layer ?? "—"}</td>
                  <td className="nowrap">
                    <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: componentColor(t.component), marginRight: 6 }} />
                    {t.component}
                    {t.role && <span className="faint small"> · {t.role}</span>}
                  </td>
                  <td className="mono">{t.dtype}</td>
                  <td className="mono small nowrap">{shapeStr(t.shape)}</td>
                  <td className="num">{fmtInt(t.n_elements)}</td>
                  <td className="num">{fmtBytes(t.n_bytes)}</td>
                  <td className="num mono small faint">{t.offset?.toLocaleString() ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

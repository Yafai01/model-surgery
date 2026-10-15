import type { ReactNode } from "react";
import type { Provenance } from "../types";

const PROV_TITLE: Record<Provenance, string> = {
  verified: "Read directly from the file (header field, metadata key or tensor descriptor)",
  derived: "Computed deterministically from verified values",
  inferred: "Heuristic: guessed from naming conventions or assumptions",
};

export function Prov({ p }: { p: Provenance | null | undefined }) {
  if (!p) return null;
  return (
    <span className={`badge prov ${p}`} title={PROV_TITLE[p]}>
      {p}
    </span>
  );
}

export function ProvLegend() {
  return (
    <div className="row small dim">
      <Prov p="verified" /> from file
      <Prov p="derived" /> computed from verified values
      <Prov p="inferred" /> heuristic
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

export function Spinner() {
  return <span className="spinner" aria-label="loading" />;
}

export function Callout({ kind = "info", title, children }: { kind?: "info" | "warn" | "bad" | "ok"; title?: string; children: ReactNode }) {
  return (
    <div className={`callout ${kind === "info" ? "" : kind}`}>
      {title && <strong>{title} </strong>}
      {children}
    </div>
  );
}

export function EstimateBanner({ children }: { children?: ReactNode }) {
  return (
    <div className="estimate-banner" role="note">
      <span aria-hidden>≈</span>
      <span>{children ?? "Estimate — derived from file metadata and stated assumptions. Not a measurement."}</span>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="card empty">
      <h3>{title}</h3>
      {children}
    </div>
  );
}

export function List({ items, max = 50 }: { items: string[]; max?: number }) {
  if (!items.length) return null;
  return (
    <ul>
      {items.slice(0, max).map((x, i) => (
        <li key={i}>{x}</li>
      ))}
      {items.length > max && <li className="faint">… {items.length - max} more</li>}
    </ul>
  );
}

export function fmtBytes(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const neg = n < 0;
  let v = Math.abs(n);
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${neg ? "−" : ""}${i === 0 ? v.toFixed(0) : v.toFixed(digits)} ${units[i]}`;
}

export function fmtCount(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  const tiers: [number, string][] = [
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  for (const [d, s] of tiers) if (n >= d) return `${(n / d).toFixed(2)}${s}`;
  return String(n);
}

export function fmtInt(n: number | null | undefined): string {
  return n === null || n === undefined ? "—" : n.toLocaleString();
}

export function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : String(v);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

export function fmtTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function shapeStr(shape: number[]): string {
  return `[${shape.join(", ")}]`;
}

// Component palette shared by the 2D and 3D architecture views.
export const COMPONENT_COLORS: Record<string, string> = {
  embedding: "#a78bfa",
  attention: "#38bdf8",
  ffn: "#2dd4bf",
  experts: "#34d399",
  router: "#fbbf24",
  norm: "#94a3b8",
  ssm: "#f472b6",
  output: "#fb923c",
  other: "#64748b",
};

export function componentColor(c: string): string {
  return COMPONENT_COLORS[c] ?? COMPONENT_COLORS.other;
}

export function download(filename: string, text: string, mime = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

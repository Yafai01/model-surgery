// Scan history for the browser edition, kept in this browser's IndexedDB (never sent anywhere).
// Falls back to an in-memory map when IndexedDB is unavailable (private mode, blocked storage).

import type { ScanResult, ScanSummary } from "../types";

const DB = "model-surgery";
const STORE = "scans";
const memory = new Map<string, ScanResult>();

function open(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> {
  const db = await open();
  if (!db) return undefined;
  return new Promise((resolve) => {
    try {
      const r = fn(db.transaction(STORE, mode).objectStore(STORE));
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => resolve(undefined);
    } catch {
      resolve(undefined);
    }
  });
}

export async function saveScan(s: ScanResult): Promise<void> {
  memory.set(s.id, s);
  await tx("readwrite", (st) => st.put(s));
}

export async function getScan(id: string): Promise<ScanResult | undefined> {
  return memory.get(id) ?? (await tx<ScanResult>("readonly", (st) => st.get(id)));
}

export async function deleteScan(id: string): Promise<boolean> {
  memory.delete(id);
  await tx("readwrite", (st) => st.delete(id));
  return true;
}

export async function listScans(): Promise<ScanSummary[]> {
  const stored = (await tx<ScanResult[]>("readonly", (st) => st.getAll())) ?? [];
  const all = new Map<string, ScanResult>(stored.map((s) => [s.id, s]));
  for (const [k, v] of memory) all.set(k, v);
  return [...all.values()]
    .sort((a, b) => b.scanned_at.localeCompare(a.scanned_at))
    .map((s) => ({
      id: s.id,
      scanned_at: s.scanned_at,
      path: s.path,
      kind: s.kind,
      name: s.name,
      total_size_bytes: s.total_size_bytes,
      n_params: (s.hyperparameters.find((f) => f.key === "n_params")?.value as number) ?? null,
      quant_label: s.quantization.label,
      valid: s.integrity.valid,
    }));
}

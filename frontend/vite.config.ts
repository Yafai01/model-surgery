import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `vite preview` serves the same security headers as production (vercel.json), so CSP problems show up locally.
function productionHeaders(): Record<string, string> {
  try {
    const cfg = JSON.parse(readFileSync(new URL("../vercel.json", import.meta.url), "utf8"));
    const all = cfg.headers?.find((h: { source: string }) => h.source === "/(.*)")?.headers ?? [];
    return Object.fromEntries(all.map((h: { key: string; value: string }) => [h.key, h.value]));
  } catch {
    return {};
  }
}

// In development the Vite server proxies /api to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: false } },
  },
  preview: {
    headers: productionHeaders(),
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 800,
  },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the Vite server proxies /api to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: false } },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 800,
  },
});

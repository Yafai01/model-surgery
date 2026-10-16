import { defineConfig } from "@playwright/test";

// Browser tests use the installed Google Chrome (no Playwright browser download needed).
// Run `npm run build && npm run build:browser` first (or `npm run test:e2e`, which does both).
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  workers: 1,
  reporter: [["list"]],
  use: { channel: process.env.E2E_CHANNEL ?? "chrome", trace: "retain-on-failure" },
  projects: [
    { name: "browser-edition", testMatch: /browser\.spec\.ts/, use: { baseURL: "http://127.0.0.1:4174" } },
    { name: "local-app", testMatch: /local\.spec\.ts/, use: { baseURL: "http://127.0.0.1:8799" } },
  ],
  webServer: [
    {
      command: "npx vite preview --mode browser --outDir dist-browser --port 4174 --strictPort --host 127.0.0.1",
      url: "http://127.0.0.1:4174",
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: "node e2e/serve-local.mjs",
      url: "http://127.0.0.1:8799/api/health",
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});

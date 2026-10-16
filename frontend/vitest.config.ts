import { defineConfig } from "vitest/config";

// Unit and integration tests for the in-page engine. Synthetic fixtures only: no model downloads.
export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
    testTimeout: 60_000,
  },
});

// Local app: the real FastAPI backend serving the built frontend, with a throwaway data directory.
// Real-model inference tests run only when both are set:
//   E2E_LLAMA_CPP_DIR = folder with llama-server   E2E_GGUF = a small GGUF (e.g. gemma-3-1b Q4_K_M)
import { expect, test } from "@playwright/test";
import { fixture, watchErrors } from "./helpers";

const tiny = fixture("local-tiny.gguf", { nLayers: 4 });
const REAL = process.env.E2E_GGUF;
const RUNTIME = !!process.env.E2E_LLAMA_CPP_DIR;

async function scanPath(page: import("@playwright/test").Page, path: string) {
  await page.goto("/");
  await page.getByLabel("Model path").fill(path);
  await page.getByRole("button", { name: "Scan", exact: true }).click();
}

test("scans a local file by path and survives a bad path", async ({ page }) => {
  const w = watchErrors(page, [/Failed to load resource/]);
  await scanPath(page, tiny + ".missing");
  await expect(page.getByRole("alert")).toContainText(/does not exist|not found|no such/i);
  await page.getByLabel("Model path").fill(tiny);
  await page.getByRole("button", { name: "Scan", exact: true }).click();
  await expect(page.locator(".topbar-title")).toHaveText("tiny-llama");
  await page.getByRole("navigation", { name: "Main", exact: true }).getByRole("button", { name: "Architecture" }).click();
  await expect(page.locator(".layer-row:not(.global-row)")).toHaveCount(4);
  w.assertClean();
});

test("deleting the open model returns to History and disables model pages", async ({ page }) => {
  await scanPath(page, tiny);
  await expect(page.locator(".topbar-title")).toHaveText("tiny-llama");
  const nav = page.getByRole("navigation", { name: "Main", exact: true });
  await nav.getByRole("button", { name: "History" }).click();
  // Other tests scan the same file, so several rows share the name (and possibly the same timestamp):
  // remove all of them; the open model is among them.
  page.on("dialog", (d) => d.accept());
  const remove = page.getByRole("button", { name: /Remove tiny-llama from history/ });
  while ((await remove.count()) > 0) {
    const n = await remove.count();
    await remove.first().click();
    await expect(remove).toHaveCount(n - 1);
  }
  await expect(page.getByText("No model loaded")).toBeVisible();
  await expect(nav.getByRole("button", { name: "Performance" })).toBeDisabled();
});

test("settings shows runtime detection and measured system memory", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("navigation", { name: "Main", exact: true }).getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("cell", { name: "llama-server (inference)" })).toBeVisible();
  await expect(page.getByText(/total · .* available/)).toBeVisible();
  await page.getByRole("button", { name: "Re-detect" }).click();
  await expect(page.getByRole("button", { name: "Re-detect" })).toBeEnabled({ timeout: 60_000 });
});

test("inference page states what the runtime can and cannot show", async ({ page }) => {
  await scanPath(page, tiny);
  await page.getByRole("navigation", { name: "Main", exact: true }).getByRole("button", { name: "Inference" }).click();
  await expect(page.locator(".cap.off", { hasText: "Layer-by-layer forward trace" })).toBeVisible();
  await expect(page.locator(".cap.off", { hasText: "Attention maps" })).toBeVisible();
  if (!RUNTIME) await expect(page.getByText("No runtime available.")).toBeVisible();
});

test.describe("real model inference", () => {
  test.skip(!REAL || !RUNTIME, "set E2E_GGUF and E2E_LLAMA_CPP_DIR to run");
  test.setTimeout(240_000);

  test("loads, streams tokens, stops on request and records measurements", async ({ page }) => {
    const w = watchErrors(page);
    await scanPath(page, REAL!);
    await expect(page.getByText("integrity ok")).toBeVisible();
    const nav = page.getByRole("navigation", { name: "Main", exact: true });
    await nav.getByRole("button", { name: "Inference" }).click();
    await page.getByLabel("Context length").fill("2048");
    await page.getByLabel("GPU offload").selectOption("cpu");
    await page.getByRole("button", { name: "Load model" }).click();
    await expect(page.getByText("runtime: ready")).toBeVisible({ timeout: 180_000 });
    await expect(page.getByText(/Buffers allocated by llama.cpp/)).toBeVisible();

    // a short deterministic generation
    await page.getByLabel("User message").fill("Name three primary colors.");
    await page.getByLabel("Max new tokens").fill("16");
    await page.getByLabel("Temperature").fill("0");
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.locator(".stat", { hasText: "Decode speed" }).locator(".stat-value")).toHaveText(/tok\/s/, { timeout: 120_000 });
    await expect(page.locator(".gen-out .tok")).toHaveCount(16);
    await page.locator(".gen-out .tok").first().click();
    await expect(page.locator(".token-detail table tbody tr").first()).toBeVisible();
    await page.getByText(/Prompt as evaluated/).click();
    await expect(page.locator(".ptok").first()).toBeVisible();

    // stop a long generation
    await page.getByLabel("Max new tokens").fill("2000");
    await page.getByLabel("User message").fill("Write a very long story about a lighthouse.");
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.locator(".gen-out .tok").nth(5)).toBeAttached({ timeout: 60_000 });
    await page.getByRole("button", { name: "■ Stop" }).click();
    await expect(page.getByText("stopped by user")).toBeVisible();
    const n = await page.locator(".gen-out .tok").count();
    expect(n).toBeLessThan(2000);
    await expect(page.getByRole("button", { name: "Generate" })).toBeEnabled();

    // measured vs estimated on the Performance page
    await nav.getByRole("button", { name: "Performance" }).click();
    await expect(page.getByRole("columnheader", { name: "Allocated (reported by llama.cpp)" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Generation runs" })).toBeVisible();
    await expect(page.locator("table", { hasText: "Decode tok/s" }).locator("tbody tr")).toHaveCount(1);

    await nav.getByRole("button", { name: "Inference" }).click();
    await page.getByRole("button", { name: "Unload model" }).click();
    await expect(page.getByRole("button", { name: "Load model" })).toBeVisible();
    w.assertClean();
  });
});

// Browser edition (the hosted site): everything runs in the page; files are read with the File API.
// Served by `vite preview`, which applies the production Content-Security-Policy from vercel.json.
import { expect, test } from "@playwright/test";
import { fixture, garbage, watchErrors } from "./helpers";

const tiny = fixture("tiny-llama.gguf", { nLayers: 6 });

test("starts in the light theme with model-dependent pages disabled", async ({ page }) => {
  const w = watchErrors(page);
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  const nav = page.getByRole("navigation", { name: "Main", exact: true });
  await expect(nav.getByRole("button", { name: "Architecture" })).toBeDisabled();
  await expect(nav.getByRole("button", { name: "Settings" })).toBeEnabled();
  w.assertClean();
});

test("imports a GGUF, explores the architecture in 2D and 3D, and searches tensors", async ({ page }) => {
  const w = watchErrors(page);
  await page.goto("/");
  await page.locator('input[type="file"]').first().setInputFiles(tiny);
  await expect(page.locator(".topbar-title")).toHaveText("tiny-llama");
  await expect(page.getByText("integrity ok")).toBeVisible();

  const nav = page.getByRole("navigation", { name: "Main", exact: true });
  await nav.getByRole("button", { name: "Architecture" }).click();
  await expect(page.getByRole("heading", { name: "Architecture explorer" })).toBeVisible();
  await expect(page.locator(".layer-row:not(.global-row)")).toHaveCount(6);

  // search + jump to the first match
  await page.getByLabel("Search tensors").fill("blk.4.attn_k");
  await expect(page.getByRole("status").filter({ hasText: "tensor(s)" })).toHaveText(/1 tensor\(s\) in 1 layer\(s\)/);
  await page.getByLabel("Search tensors").press("Enter");
  await expect(page.locator(".details-panel h3").first()).toHaveText("blk.4");
  await expect(page.locator(".layer-row.match-row")).toHaveCount(1);

  // layer navigation by button and keyboard
  await page.getByRole("button", { name: "Next layer" }).click();
  await expect(page.locator(".details-panel h3").first()).toHaveText("blk.5");
  await page.locator(".layer-row.selected").focus();
  await page.keyboard.press("ArrowUp");
  await expect(page.locator(".details-panel h3").first()).toHaveText("blk.4");

  // dataflow for the selected layer, with illustrative residual nodes
  await expect(page.getByRole("heading", { name: /Layer 4 dataflow/ })).toBeVisible();
  await expect(page.locator(".flow-node.illustrative").first()).toBeVisible();

  // 3D: renders, fit/reset work, keyboard and selection stay in sync, orbit drag does not select
  await page.getByRole("radio", { name: "3D" }).click();
  const canvas = page.locator(".canvas-3d canvas");
  await expect(canvas).toBeVisible();
  await page.getByRole("button", { name: "Fit selection" }).click();
  await page.getByRole("button", { name: "Fit all / reset" }).click();
  await page.locator(".canvas-3d").focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.locator(".details-panel h3").first()).toHaveText("blk.5");
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 80, box.y + box.height / 2 + 20, { steps: 5 });
  await page.mouse.up();
  await expect(page.locator(".details-panel h3").first()).toHaveText("blk.5");
  await expect(page.locator(".label-3d").first()).toBeAttached();

  // isolate a component, then back to 2D (3D unmounts and disposes its context)
  await page.getByRole("button", { name: "attention", pressed: true }).click();
  await expect(page.getByRole("button", { name: "attention", pressed: false })).toBeVisible();
  await page.getByRole("radio", { name: "2D" }).click();
  await expect(canvas).toHaveCount(0);
  w.assertClean();
});

test("the architecture graph export is JSON with provenance", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').first().setInputFiles(tiny);
  await page.getByRole("navigation", { name: "Main", exact: true }).getByRole("button", { name: "Architecture" }).click();
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: /Graph JSON/ }).click()]);
  const chunks: Buffer[] = [];
  for await (const c of await dl.createReadStream()) chunks.push(c as Buffer);
  const graph = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  expect(graph.schema).toBe("model-surgery/architecture-graph@1");
  expect(graph.layers).toHaveLength(6);
  expect(graph.layers[0].nodes.find((n: { kind: string }) => n.kind === "residual_add").illustrative).toBe(true);
});

test("malformed and unsupported files produce honest errors", async ({ page }) => {
  const w = watchErrors(page);
  await page.goto("/");
  await page.locator('input[type="file"]').first().setInputFiles(garbage("broken.gguf"));
  await expect(page.getByRole("alert")).toContainText(/GGUF magic/);
  await page.locator('input[type="file"]').first().setInputFiles(garbage("weights.bin"));
  await expect(page.getByRole("alert")).toContainText(/pickle/i);
  w.assertClean();
});

test("inference is honestly unavailable in the browser edition", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').first().setInputFiles(tiny);
  await page.getByRole("navigation", { name: "Main", exact: true }).getByRole("button", { name: "Inference" }).click();
  await expect(page.getByText("No runtime available.")).toBeVisible();
  await expect(page.getByText(/cannot execute GGUF or safetensors/)).toBeVisible();
  await expect(page.locator(".cap.on")).toHaveCount(0);
});

test("core workflows make no requests outside the page's origin", async ({ page }) => {
  const external: string[] = [];
  await page.route("**/*", (route) => {
    const u = new URL(route.request().url());
    if (u.hostname !== "127.0.0.1") {
      external.push(u.href);
      return route.abort();
    }
    return route.continue();
  });
  const w = watchErrors(page);
  await page.goto("/");
  await page.locator('input[type="file"]').first().setInputFiles(tiny);
  const nav = page.getByRole("navigation", { name: "Main", exact: true });
  for (const name of ["Architecture", "Tensors", "Metadata", "Performance", "Compare & export", "History", "Settings", "Overview"]) {
    await nav.getByRole("button", { name, exact: true }).click();
    await expect(page.locator(".content")).not.toBeEmpty();
  }
  await nav.getByRole("button", { name: "Architecture" }).click();
  await page.getByRole("radio", { name: "3D" }).click();
  await expect(page.locator(".canvas-3d canvas")).toBeVisible();
  expect(external).toEqual([]);
  w.assertClean();
});

test("phone-width layout has working navigation and no horizontal scroll", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".sidebar")).toBeHidden();
  const mnav = page.getByRole("navigation", { name: "Main (compact)" });
  await expect(mnav).toBeVisible();
  await page.locator('input[type="file"]').first().setInputFiles(tiny);
  await mnav.getByRole("button", { name: "Architecture" }).click();
  await expect(page.getByRole("heading", { name: "Architecture explorer" })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("theme preference applies and persists", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("navigation", { name: "Main", exact: true }).getByRole("button", { name: "Settings" }).click();
  await page.getByLabel("Theme").selectOption("dark");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

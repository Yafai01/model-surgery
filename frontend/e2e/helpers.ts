import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Page, expect } from "@playwright/test";
import { ggufBytes, llamaLike } from "../src/test/fixtures";

export const dir = mkdtempSync(join(tmpdir(), "ms-e2e-"));

/** Writes a synthetic llama-style GGUF (headers + zero data) and returns its absolute path. */
export function fixture(name: string, opts: Parameters<typeof llamaLike>[0] = {}): string {
  const { kv, tensors } = llamaLike(opts);
  const p = join(dir, name);
  writeFileSync(p, ggufBytes(kv, tensors));
  return p;
}

export function garbage(name: string): string {
  const p = join(dir, name);
  writeFileSync(p, "this is not a model file");
  return p;
}

/** Collects uncaught page errors and console errors; call assertClean() at the end of a test. */
export function watchErrors(page: Page, allow: RegExp[] = []) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error" && !allow.some((r) => r.test(m.text()))) errors.push(`console: ${m.text()}`);
  });
  return { errors, assertClean: () => expect(errors, errors.join("\n")).toEqual([]) };
}

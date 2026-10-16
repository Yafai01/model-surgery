// Starts the local Model Surgery server for the e2e tests with a fresh, throwaway data directory,
// so tests never touch the user's real history (~/.model_surgery).
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const dataDir = resolve("node_modules/.tmp/e2e-data");
rmSync(dataDir, { recursive: true, force: true });
mkdirSync(dataDir, { recursive: true });
const bin = [process.env.MODEL_SURGERY_BIN, resolve("../.venv/Scripts/model-surgery.exe"), resolve("../.venv/bin/model-surgery")].find((p) => p && existsSync(p));
if (!bin) {
  console.error("model-surgery executable not found; create ../.venv (see README) or set MODEL_SURGERY_BIN");
  process.exit(1);
}
const args = ["--data-dir", dataDir];
if (process.env.E2E_LLAMA_CPP_DIR) args.push("--llama-cpp-dir", process.env.E2E_LLAMA_CPP_DIR);
args.push("serve", "--port", "8799");
const child = spawn(bin, args, { stdio: "inherit", env: { ...process.env, MODEL_SURGERY_FRONTEND_DIST: resolve("dist") } });
const stop = () => child.kill();
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
process.on("exit", stop);
child.on("exit", (code) => process.exit(code ?? 0));

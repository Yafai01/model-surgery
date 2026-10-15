"""Stand-in for llama-quantize used by the job tests. Behaviour is driven by env vars."""

import os
import shutil
import sys
import time

args = [a for a in sys.argv[1:] if a != "--allow-requantize"]
src, out, qtype = args[0], args[1], args[2]
n = 5
for i in range(1, n + 1):
    print(f"[{i:4d}/{n:4d}] tensor{i}.weight - [ 64, 64, 1, 1], type = f16, converting to {qtype.lower()}", flush=True)
    time.sleep(float(os.environ.get("FAKE_DELAY", "0")))

mode = os.environ.get("FAKE_MODE", "ok")
if mode == "fail":
    print("llama_model_quantize: error: simulated failure", flush=True)
    with open(out, "wb") as fh:
        fh.write(b"partial")
    sys.exit(1)
if mode == "garbage":
    with open(out, "wb") as fh:
        fh.write(b"not a gguf file")
    sys.exit(0)
shutil.copyfile(os.environ["FAKE_OUTPUT_SRC"], out)

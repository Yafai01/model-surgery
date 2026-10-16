"""Stand-in for llama-server used by test_runtime.py.

Speaks the subset of the HTTP API that model_surgery.runtime uses and prints
the load-time log lines it parses. Behaviour is selected with environment
variables:
  FAKE_SERVER_MODE   = ok (default) | crash (exit 1 while loading) | hang (never becomes healthy)
  FAKE_SERVER_DELAY  = seconds between streamed tokens (default 0)
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]


def arg(name, default=None):
    return args[args.index(name) + 1] if name in args else default


PORT = int(arg("--port"))
KEY = arg("--api-key")
N_CTX = int(arg("-c", "4096"))
NGL = int(arg("-ngl", "0"))
MODE = os.environ.get("FAKE_SERVER_MODE", "ok")
DELAY = float(os.environ.get("FAKE_SERVER_DELAY", "0"))
WORDS = ["Hello", ",", " world", "!", " This", " is", " a", " fake", " model", "."]

print("load_tensors: offloaded %d/5 layers to GPU" % NGL, flush=True)
print("load_tensors:   CPU_Mapped model buffer size =   12.50 MiB", flush=True)
print("llama_context: n_ctx         = %d" % N_CTX, flush=True)
print("llama_kv_cache:        CPU KV buffer size =     2.00 MiB", flush=True)
print("sched_reserve:        CPU compute buffer size =     3.25 MiB", flush=True)
if MODE == "crash":
    print("llama_model_load: error loading model: fake failure", flush=True)
    sys.exit(1)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            self._json(401, {"error": {"message": "Invalid API Key"}})
            return False
        return True

    def do_GET(self):
        if self.path == "/health":
            if MODE == "hang":
                return self._json(503, {"error": {"message": "Loading model"}})
            return self._json(200, {"status": "ok"})
        self._json(404, {})

    def do_POST(self):
        if not self._authed():
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path == "/tokenize":
            words = body["content"].split()
            toks = ([{"id": 1, "piece": "<s>"}] if body.get("add_special") else []) + [{"id": 100 + i, "piece": w} for i, w in enumerate(words)]
            return self._json(200, {"tokens": toks if body.get("with_pieces") else [t["id"] for t in toks]})
        if self.path == "/apply-template":
            text = "".join(f"<|{m['role']}|>{m['content']}" for m in body["messages"]) + "<|assistant|>"
            return self._json(200, {"prompt": text})
        if self.path == "/completion":
            n = body.get("n_predict", 4)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i in range(n):
                    w = WORDS[i % len(WORDS)]
                    ev = {"content": w, "tokens": [200 + i], "stop": False}
                    if body.get("n_probs"):
                        ev["completion_probabilities"] = [{"id": 200 + i, "token": w, "logprob": -0.1, "top_logprobs": [{"id": 200 + i, "token": w, "logprob": -0.1}, {"id": 999, "token": " alt", "logprob": -2.5}]}]
                    self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                    self.wfile.flush()
                    if DELAY:
                        time.sleep(DELAY)
                done = {
                    "content": "", "stop": True, "stop_type": "limit", "tokens_predicted": n, "tokens_evaluated": len(body["prompt"]),
                    "tokens_cached": 0, "truncated": False, "generation_settings": {"temperature": body.get("temperature"), "n_probs": body.get("n_probs")},
                    "timings": {"cache_n": 0, "prompt_n": len(body["prompt"]), "prompt_ms": 5.0, "prompt_per_second": 200.0, "predicted_n": n, "predicted_ms": 10.0 * n, "predicted_per_second": 100.0},
                }
                self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return
        self._json(404, {})


srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
print("main: server is listening on http://127.0.0.1:%d" % PORT, flush=True)
srv.serve_forever()

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import hf_llama_dir, llama_like_gguf

import model_surgery.runtime as runtime_mod
from model_surgery.api import create_app
from model_surgery.config import Settings
from model_surgery.history import History
from model_surgery.runtime import GenerateRequest, RuntimeBlocked, RuntimeManager, RuntimeStartRequest
from model_surgery.scanner import scan_path
from model_surgery.system import SystemMemory
from model_surgery.tooling import ServerTool

FAKE = Path(__file__).with_name("fake_llama_server.py")
H = {"X-Model-Surgery": "1"}


def fake_popen(argv, *a, **kw):
    return subprocess.Popen([sys.executable, str(FAKE)] + list(argv[1:]), *a, **kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_SERVER_MODE", raising=False)
    monkeypatch.delenv("FAKE_SERVER_DELAY", raising=False)
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    tool = ServerTool(available=True, path=str(tmp_path / "llama-server.exe"), version="test")
    mgr = RuntimeManager(settings, tool_fn=lambda refresh=False: tool, popen=fake_popen, load_timeout=20)
    model = llama_like_gguf(tmp_path / "tiny.gguf")
    scan = scan_path(str(model), settings)
    yield mgr, scan, settings, tmp_path
    mgr.shutdown()


def wait_state(mgr, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        st = mgr.status()
        if st.state != "starting":
            return st
        time.sleep(0.05)
    raise AssertionError("runtime did not leave 'starting'")


def test_unavailable_tool_blocks_start(tmp_path):
    s = Settings(data_dir=tmp_path / "d")
    s.ensure_dirs()
    mgr = RuntimeManager(s, tool_fn=lambda refresh=False: ServerTool(available=False, detail="llama-server not found"))
    scan = scan_path(str(llama_like_gguf(tmp_path / "m.gguf")), s)
    assert mgr.status().state == "unavailable"
    with pytest.raises(RuntimeBlocked, match="not available"):
        mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)


def test_start_parses_runtime_log_and_reports_capabilities(env):
    mgr, scan, *_ = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id, context_length=1024, n_gpu_layers=3), scan)
    st = wait_state(mgr)
    assert st.state == "ready", st.error
    assert st.n_ctx_runtime == 1024
    assert (st.gpu_layers_offloaded, st.total_layers) == (3, 5)
    kinds = {(a.kind, a.device): a.bytes for a in st.allocations}
    assert kinds[("model", "CPU_Mapped")] == int(12.5 * 2**20)
    assert kinds[("KV", "CPU")] == 2 * 2**20
    assert st.load_seconds is not None and st.load_seconds >= 0
    assert st.capabilities["token_ids"] and not st.capabilities["layer_trace"] and not st.capabilities["attention_maps"]
    assert "no API for intermediate activations" in st.trace_note


def test_generate_streams_runtime_tokens_and_records_measurement(env):
    mgr, scan, *_ = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    assert wait_state(mgr).state == "ready"
    events = list(mgr.generate(GenerateRequest(prompt="say hi", n_predict=5, n_probs=2, system="be brief")).events())
    kinds = [e["type"] for e in events]
    assert kinds[0] == "prompt" and kinds[-1] == "done" and kinds.count("token") == 5
    prompt = events[0]
    assert prompt["chat_template_applied"] and prompt["formatted"].startswith("<|system|>be brief<|user|>say hi")
    assert prompt["tokens"][0]["id"] == 100  # chat prompts are tokenized without extra special tokens
    tok = events[1]
    assert tok["content"] == "Hello" and tok["ids"] == [200] and tok["probs"][0]["top"][1]["token"] == " alt"
    done = events[-1]
    assert done["timings"]["predicted_per_second"] == 100.0 and done["tokens_predicted"] == 5
    assert done["wall"]["total_ms"] >= done["wall"]["first_token_ms"] >= 0
    (m,) = mgr.measurements()
    assert m.decode_tokens_per_s == 100.0 and m.generated_tokens == 5 and m.scan_id == scan.id


def test_raw_prompt_gets_special_tokens(env):
    mgr, scan, *_ = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    wait_state(mgr)
    events = list(mgr.generate(GenerateRequest(prompt="a b", chat=False, n_predict=1)).events())
    assert events[0]["tokens"][0]["piece"] == "<s>" and not events[0]["chat_template_applied"]


def test_generate_requires_ready_runtime_and_is_exclusive(env, monkeypatch):
    mgr, scan, *_ = env
    with pytest.raises(RuntimeBlocked, match="start the runtime"):
        mgr.generate(GenerateRequest(prompt="x"))
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    wait_state(mgr)
    g = mgr.generate(GenerateRequest(prompt="x"))
    with pytest.raises(RuntimeBlocked, match="already running"):
        mgr.generate(GenerateRequest(prompt="y"))
    list(g.events())
    mgr.generate(GenerateRequest(prompt="z"))  # slot is free again


def test_cancel_stops_stream_quickly(env, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_DELAY", "0.2")
    mgr, scan, *_ = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    wait_state(mgr)
    g = mgr.generate(GenerateRequest(prompt="x", n_predict=100))
    seen, t0 = [], time.time()
    for ev in g.events():
        seen.append(ev["type"])
        if seen.count("token") == 2:
            assert mgr.cancel_generation()
    assert seen[-1] == "cancelled" and "done" not in seen
    assert time.time() - t0 < 5
    assert not mgr.status().generating and mgr.measurements() == []


def test_context_overflow_is_reported(env):
    mgr, scan, *_ = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id, context_length=128), scan)
    wait_state(mgr)
    events = list(mgr.generate(GenerateRequest(prompt="word " * 100, chat=False, n_predict=64)).events())
    assert events[-1]["type"] == "error" and "exceeds the loaded context of 128" in events[-1]["message"]


def test_crash_during_load_is_reported(env, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_MODE", "crash")
    mgr, scan, *_ = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    st = wait_state(mgr)
    assert st.state == "failed" and "fake failure" in st.error


def test_load_timeout_kills_server(env, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_MODE", "hang")
    mgr, scan, *_ = env
    mgr.load_timeout = 1.5
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    proc = mgr._proc
    st = wait_state(mgr, timeout=20)
    assert st.state == "failed" and "did not become ready" in st.error
    assert proc.poll() is not None


def test_stop_terminates_process_and_switching_models_replaces_it(env):
    mgr, scan, settings, tmp = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    wait_state(mgr)
    first = mgr._proc
    scan2 = scan_path(str(llama_like_gguf(tmp / "other.gguf")), settings)
    mgr.start(RuntimeStartRequest(scan_id=scan2.id), scan2)
    assert first.poll() is not None  # previous model was unloaded
    assert wait_state(mgr).scan_id == scan2.id
    second = mgr._proc
    assert mgr.stop().state == "stopped"
    assert second.poll() is not None


def test_only_local_valid_unchanged_gguf_can_be_loaded(env):
    mgr, scan, settings, tmp = env
    hf = scan_path(str(hf_llama_dir(tmp / "hf")), settings)
    with pytest.raises(RuntimeBlocked, match="local GGUF"):
        mgr.start(RuntimeStartRequest(scan_id=hf.id), hf)
    remote = scan.model_copy(update={"path": "hf://org/model/m.gguf"})
    with pytest.raises(RuntimeBlocked, match="local GGUF"):
        mgr.start(RuntimeStartRequest(scan_id=scan.id), remote)
    with open(scan.files[0].path, "ab") as f:
        f.write(b"\0" * 32)
    with pytest.raises(RuntimeBlocked, match="changed since it was scanned"):
        mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)


def test_memory_guard_blocks_unless_acknowledged(env, monkeypatch):
    mgr, scan, *_ = env
    monkeypatch.setattr(runtime_mod, "system_memory", lambda: SystemMemory(total_bytes=1 << 30, available_bytes=1024, source="test"))
    with pytest.raises(RuntimeBlocked, match="exceeds the .* available"):
        mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    assert mgr._proc is None
    mgr.start(RuntimeStartRequest(scan_id=scan.id, allow_memory_risk=True), scan)
    assert wait_state(mgr).state == "ready"


def test_http_api_streams_ndjson(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_SERVER_MODE", raising=False)
    monkeypatch.delenv("FAKE_SERVER_DELAY", raising=False)
    settings = Settings(data_dir=tmp_path / "data")
    tool = ServerTool(available=True, path=str(tmp_path / "llama-server.exe"), version="test")
    mgr = RuntimeManager(settings, tool_fn=lambda refresh=False: tool, popen=fake_popen, load_timeout=20)
    model = llama_like_gguf(tmp_path / "m.gguf")
    with TestClient(create_app(settings, runtime_manager=mgr), base_url="http://127.0.0.1") as c:
        scan = c.post("/api/scan", json={"path": str(model)}, headers=H).json()
        assert c.post("/api/runtime/start", json={"scan_id": scan["id"]}).status_code == 403  # CSRF header required
        assert c.post("/api/runtime/generate", json={"prompt": "x"}, headers=H).status_code == 409
        assert c.post("/api/runtime/start", json={"scan_id": scan["id"]}, headers=H).status_code == 200
        wait_state(mgr)
        with c.stream("POST", "/api/runtime/generate", json={"prompt": "hi", "n_predict": 3}, headers=H) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
            types = [json.loads(line)["type"] for line in r.iter_lines() if line]
        assert types == ["prompt", "token", "token", "token", "done"]
        assert len(c.get("/api/runtime/measurements").json()) == 1
        tk = c.post("/api/runtime/tokenize", json={"text": "a b c"}, headers=H).json()
        assert tk["n_tokens"] == 4
        assert c.post("/api/runtime/stop", headers=H).json()["state"] == "stopped"
    assert mgr._proc is None


def test_runtime_key_is_never_exposed(env):
    mgr, scan, *_ = env
    mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
    wait_state(mgr)
    dumped = mgr.status().model_dump_json()
    assert mgr._key and mgr._key not in dumped
    # the argv is logged nowhere in status either
    assert "--api-key" not in dumped


def test_system_endpoint_reports_measured_memory(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path / "d")), base_url="http://127.0.0.1")
    j = c.get("/api/system").json()
    assert "memory" in j and j["memory"]["source"]
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        assert j["memory"]["total_bytes"] > 0 and j["memory"]["available_bytes"] > 0


def test_history_is_untouched_by_runtime(env):
    mgr, scan, settings, _ = env
    History(settings.db_path)  # runtime keeps no on-disk state besides its process
    assert not any(p.suffix == ".partial" for p in settings.outputs_dir.iterdir())


HOST_SCRIPT = r"""
import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from helpers import llama_like_gguf
from model_surgery.config import Settings
from model_surgery.runtime import RuntimeManager, RuntimeStartRequest
from model_surgery.scanner import scan_path
from model_surgery.tooling import ServerTool
tmp = Path(sys.argv[2])
s = Settings(data_dir=tmp / "data"); s.ensure_dirs()
fake = str(Path(sys.argv[1]) / "fake_llama_server.py")
popen = lambda argv, *a, **kw: subprocess.Popen([sys.executable, fake] + list(argv[1:]), *a, **kw)
mgr = RuntimeManager(s, tool_fn=lambda r=False: ServerTool(available=True, path="llama-server.exe"), popen=popen)
scan = scan_path(str(llama_like_gguf(tmp / "m.gguf")), s)
mgr.start(RuntimeStartRequest(scan_id=scan.id), scan)
while mgr.status().state == "starting":
    time.sleep(0.05)
print(mgr._proc.pid, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="kill-on-close job objects are Windows-only")
def test_llama_server_dies_with_model_surgery(tmp_path):
    import ctypes

    host = subprocess.Popen([sys.executable, "-c", HOST_SCRIPT, str(Path(__file__).parent), str(tmp_path)], stdout=subprocess.PIPE, text=True)
    child_pid = int(host.stdout.readline())
    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x00100000, False, child_pid)  # SYNCHRONIZE
    assert h, "server child should be running"
    try:
        assert k32.WaitForSingleObject(ctypes.c_void_p(h), 200) == 0x102  # still alive (WAIT_TIMEOUT)
        host.kill()  # TerminateProcess: no atexit, no lifespan shutdown
        host.wait(10)
        assert k32.WaitForSingleObject(ctypes.c_void_p(h), 10_000) == 0  # the OS killed the server with its host
    finally:
        k32.CloseHandle(ctypes.c_void_p(h))

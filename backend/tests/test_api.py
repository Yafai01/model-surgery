import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import hf_llama_dir, llama_like_gguf

from model_surgery.api import create_app
from model_surgery.config import Settings

H = {"X-Model-Surgery": "1"}


@pytest.fixture
def client(tmp_path):
    s = Settings(data_dir=tmp_path / "data", extra_allowed_hosts=["testserver"])
    return TestClient(create_app(s))


def test_host_header_guard(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path / "d")), base_url="http://evil.example")
    assert c.get("/api/health").status_code == 421
    c2 = TestClient(create_app(Settings(data_dir=tmp_path / "d2")), base_url="http://localhost:8765")
    assert c2.get("/api/health").status_code == 200


def test_state_changing_requests_need_header(client, tmp_path):
    p = llama_like_gguf(tmp_path / "m.gguf")
    assert client.post("/api/scan", json={"path": str(p)}).status_code == 403
    assert client.post("/api/scan", json={"path": str(p)}, headers=H).status_code == 200


def test_no_cors_headers(client):
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    r = client.options("/api/scan", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_scan_history_export_roundtrip(client, tmp_path):
    p = llama_like_gguf(tmp_path / "m.gguf")
    r = client.post("/api/scan", json={"path": str(p)}, headers=H).json()
    sid = r["id"]
    assert [s["id"] for s in client.get("/api/scans").json()] == [sid]
    assert client.get(f"/api/scans/{sid}").json()["name"] == "tiny-llama"
    j = client.get(f"/api/scans/{sid}/export?format=json")
    assert j.headers["content-disposition"].endswith('.scan.json"') and j.json()["id"] == sid
    csv = client.get(f"/api/scans/{sid}/export?format=csv").text
    assert csv.splitlines()[0].startswith("name,shape") and "blk.0.attn_q.weight" in csv
    md = client.get(f"/api/scans/{sid}/export?format=md").text
    assert md.startswith("# Model Surgery report") and "not measurements" in md.lower()
    assert client.delete(f"/api/scans/{sid}", headers=H).status_code == 200
    assert client.get(f"/api/scans/{sid}").status_code == 404


def test_csv_formula_injection_neutralised(client, tmp_path):
    from helpers import gguf_bytes

    p = tmp_path / "evil.gguf"
    p.write_bytes(gguf_bytes([], [("=HYPERLINK(\"http://x\")", [32], "F32")]))
    sid = client.post("/api/scan", json={"path": str(p)}, headers=H).json()["id"]
    csv = client.get(f"/api/scans/{sid}/export?format=csv").text
    assert "'=HYPERLINK" in csv
    assert not any(line.lstrip('"').startswith("=") for line in csv.splitlines())


def test_scan_errors_are_400(client, tmp_path):
    r = client.post("/api/scan", json={"path": str(tmp_path / "nope.gguf")}, headers=H)
    assert r.status_code == 400 and "does not exist" in r.json()["detail"]
    pk = tmp_path / "pytorch_model.bin"
    pk.write_bytes(b"\x80\x04")
    r = client.post("/api/scan", json={"path": str(pk)}, headers=H)
    assert r.status_code == 400 and "pickle" in r.json()["detail"]


def test_memory_and_devices(client, tmp_path):
    sid = client.post("/api/scan", json={"path": str(llama_like_gguf(tmp_path / "m.gguf"))}, headers=H).json()["id"]
    devs = client.get("/api/devices").json()
    assert any(d["id"] == "jetson-orin-nano-8gb" for d in devs)
    m = client.post(f"/api/scans/{sid}/memory", json={"context_length": 2048, "device_id": "jetson-orin-nano-8gb", "weight_format": "q4_0"}, headers=H).json()
    assert m["is_measurement"] is False
    assert m["selected_weights"]["id"] == "q4_0"
    assert {c["key"] for c in m["inference"]} == {"weights", "kv_cache", "compute", "runtime"}
    assert m["fit"]["device_id"] == "jetson-orin-nano-8gb"
    bad = client.post(f"/api/scans/{sid}/memory", json={"context_length": 0}, headers=H)
    assert bad.status_code == 422


def test_compare(client, tmp_path):
    a = client.post("/api/scan", json={"path": str(llama_like_gguf(tmp_path / "a.gguf"))}, headers=H).json()["id"]
    b = client.post("/api/scan", json={"path": str(llama_like_gguf(tmp_path / "b.gguf", wtype="Q8_0", ftype=7))}, headers=H).json()["id"]
    c = client.get(f"/api/compare?a={a}&b={b}").json()
    assert c["tensors_dtype_changed"] == 2 * 7 + 2
    assert any(r["label"] == "Label" and r["differs"] for r in c["rows"])
    md = client.get(f"/api/compare.md?a={a}&b={b}").text
    assert md.startswith("# Model comparison")


def test_fs_listing(client, tmp_path):
    llama_like_gguf(tmp_path / "m.gguf")
    hf_llama_dir(tmp_path / "hfmodel", shards=1)
    (tmp_path / "notes.txt").write_text("x")
    r = client.get("/api/fs/list", params={"path": str(tmp_path)}).json()
    names = {e["name"]: e for e in r["entries"]}
    assert names["m.gguf"]["kind"] == "gguf"
    assert names["hfmodel"]["is_model_dir"] is True
    assert "notes.txt" not in names
    assert client.get("/api/fs/list", params={"path": "relative"}).status_code == 400
    roots = client.get("/api/fs/roots").json()
    assert any(r["label"] == "Model Surgery outputs" for r in roots)


def test_fs_listing_respects_allowed_roots(tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()
    c = TestClient(create_app(Settings(data_dir=tmp_path / "d", allowed_roots=[ok], extra_allowed_hosts=["testserver"])))
    assert c.get("/api/fs/list", params={"path": str(ok)}).status_code == 200
    assert c.get("/api/fs/list", params={"path": str(tmp_path)}).status_code == 403


def test_tools_endpoint_never_takes_paths(client):
    r = client.get("/api/tools").json()
    assert set(r) == {"quantize", "convert", "checked_at", "configuration_help"}
    # The OpenAPI schema must not expose any field that names an executable or command.
    schema = client.get("/api/openapi.json").json()
    body_fields = set()
    for comp in schema["components"]["schemas"].values():
        if comp.get("title", "").endswith("Request"):
            body_fields |= set(comp.get("properties", {}))
    assert not body_fields & {"command", "cmd", "exe", "executable", "argv", "tool", "tool_path", "shell", "args"}


@pytest.mark.skipif(not os.environ.get("MODEL_SURGERY_TEST_GGUF"), reason="set MODEL_SURGERY_TEST_GGUF to a real .gguf to run")
def test_real_gguf(client):
    p = Path(os.environ["MODEL_SURGERY_TEST_GGUF"])
    r = client.post("/api/scan", json={"path": str(p)}, headers=H).json()
    assert r["integrity"]["valid"], r["integrity"]["errors"]
    assert sum(t["n_bytes"] for t in r["tensors"]) <= p.stat().st_size


def _as_remote(scan_json: dict) -> dict:
    scan_json = dict(scan_json)
    scan_json["path"] = "hf://owner/model@abc123"
    scan_json["files"] = [{**f, "path": "hf://owner/model@abc123/" + f["path"].replace("\\", "/").split("/")[-1]} for f in scan_json["files"]]
    scan_json["hub"] = {"repo": "owner/model", "downloads": 5}
    return scan_json


def test_import_remote_scan(client, tmp_path):
    local = client.post("/api/scan", json={"path": str(llama_like_gguf(tmp_path / "m.gguf"))}, headers=H).json()
    remote = _as_remote(local)
    remote["id"] = "remote1"
    assert client.post("/api/scans/import", json=remote).status_code == 403  # CSRF header required
    r = client.post("/api/scans/import", json=remote, headers=H)
    assert r.status_code == 200 and r.json()["hub"]["repo"] == "owner/model"
    assert client.get("/api/scans/remote1").json()["path"].startswith("hf://")
    m = client.post("/api/scans/remote1/memory", json={"device_id": "jetson-orin-nano-8gb"}, headers=H)
    assert m.status_code == 200


def test_import_rejects_local_paths(client, tmp_path):
    local = client.post("/api/scan", json={"path": str(llama_like_gguf(tmp_path / "m.gguf"))}, headers=H).json()
    local["id"] = "spoof"
    assert client.post("/api/scans/import", json=local, headers=H).status_code == 400
    mixed = _as_remote(local)
    mixed["files"][0]["path"] = "C:/Windows/System32/x.gguf"
    assert client.post("/api/scans/import", json=mixed, headers=H).status_code == 400
    assert client.post("/api/scans/import", content=b"{not json", headers={**H, "Content-Type": "application/json"}).status_code == 422


def test_remote_scans_cannot_be_quantized(client, tmp_path):
    local = client.post("/api/scan", json={"path": str(llama_like_gguf(tmp_path / "m.gguf"))}, headers=H).json()
    remote = _as_remote(local)
    remote["id"] = "remote2"
    client.post("/api/scans/import", json=remote, headers=H)
    r = client.post("/api/quantize/plan", json={"scan_id": "remote2", "operation": "quantize", "target": "Q4_0"}, headers=H)
    assert r.status_code == 403 and "Hugging Face Hub" in r.json()["detail"]

"""Regression tests for the defects recorded in docs/INITIAL_AUDIT.md (B-numbers refer to that file)."""

from pathlib import Path

from fastapi.testclient import TestClient
from helpers import T, gguf_bytes, hf_llama_dir, llama_like_gguf, safetensors_bytes

from model_surgery.api import create_app
from model_surgery.architecture import classify
from model_surgery.config import Settings
from model_surgery.devices import DeviceProfile
from model_surgery.history import History
from model_surgery.jobs import validate_output
from model_surgery.memory import MemoryRequest, estimate_memory
from model_surgery.scanner import scan_path

H = {"X-Model-Surgery": "1"}


def _settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


def test_b2_non_numeric_gguf_hyperparameters_are_ignored(tmp_path):
    kv = [
        ("general.architecture", T["str"], "llama"),
        ("llama.block_count", T["bool"], True),
        ("llama.attention.head_count", T["str"], "ab"),
        ("llama.embedding_length", T["u32"], 64),
    ]
    p = tmp_path / "bad.gguf"
    p.write_bytes(gguf_bytes(kv, [("token_embd.weight", [64, 32], "F32"), ("blk.0.attn_q.weight", [64, 64], "F32")]))
    s = _settings(tmp_path)
    scan = scan_path(str(p), s)
    assert scan.fact("n_heads") is None
    n_layers = scan.fact("n_layers")
    assert n_layers is not None and n_layers.provenance == "inferred"  # the bool was ignored; count comes from tensors
    assert any("non-numeric" in n for n in scan.architecture.notes)
    est = estimate_memory(scan, MemoryRequest(context_length=100_000, flash_attention=False))  # must not raise or explode
    assert est.total_bytes > 0


def test_b3_parameter_count_beyond_sqlite_integer_is_stored(tmp_path):
    s = _settings(tmp_path)
    scan = scan_path(str(llama_like_gguf(tmp_path / "m.gguf")), s)
    fact = scan.fact("n_params")
    fact.value = 2**70
    h = History(s.db_path)
    h.save_scan(scan)
    (row,) = h.list_scans()
    assert row.n_params is None and h.get_scan(scan.id).fact_value("n_params") == 2**70


def test_b4_export_of_non_latin1_name(tmp_path):
    kv = [("general.architecture", T["str"], "llama"), ("general.name", T["str"], "Qwen中文模型")]
    p = tmp_path / "cjk.gguf"
    p.write_bytes(gguf_bytes(kv, [("token_embd.weight", [8, 8], "F32")]))
    c = TestClient(create_app(_settings(tmp_path)), base_url="http://127.0.0.1")
    sid = c.post("/api/scan", json={"path": str(p)}, headers=H).json()["id"]
    for fmt in ("json", "csv", "md"):
        r = c.get(f"/api/scans/{sid}/export?format={fmt}")
        assert r.status_code == 200
        cd = r.headers["content-disposition"]
        assert cd.isascii() and "filename*=UTF-8''Qwen%E4%B8%AD" in cd


def test_b5_import_cannot_overwrite_existing_scan(tmp_path):
    c = TestClient(create_app(_settings(tmp_path)), base_url="http://127.0.0.1")
    local = c.post("/api/scan", json={"path": str(llama_like_gguf(tmp_path / "m.gguf"))}, headers=H).json()
    remote = {**local, "path": "hf://o/m", "files": [{**f, "path": "hf://o/m/m.gguf"} for f in local["files"]], "hub": {"repo": "o/m"}}
    r = c.post("/api/scans/import", json=remote, headers=H)
    assert r.status_code == 200 and r.json()["id"] != local["id"]
    assert not c.get(f"/api/scans/{local['id']}").json()["path"].startswith("hf://")  # original untouched
    big = c.post("/api/scans/import", content=b"x", headers={**H, "Content-Type": "application/json", "Content-Length": str(10**9)})
    assert big.status_code == 413


def test_b6_convert_validation_ignores_vision_tower(tmp_path):
    s = _settings(tmp_path)
    src_path = tmp_path / "src.gguf"
    base = llama_like_gguf(tmp_path / "base.gguf")
    out = llama_like_gguf(tmp_path / "out.gguf")
    # source = same language model + a large vision tower stack
    kv = [("general.architecture", T["str"], "llama"), ("llama.block_count", T["u32"], 2)]
    base_scan = scan_path(str(base), s)
    tensors = [(t.name, t.shape, t.dtype) for t in base_scan.tensors]
    tensors += [(f"vision_tower.layers.{i}.mlp.fc1.weight", [64, 64], "F16") for i in range(2)]
    src_path.write_bytes(gguf_bytes(kv, tensors))
    src = scan_path(str(src_path), s)
    report = validate_output("convert", out, src, None, 0)
    check = next(c for c in report.checks if c.name == "parameter count")
    assert check.passed and "vision/audio/auxiliary" in check.detail


def test_b7_b10_b11_component_classification():
    assert classify("backbone.layers.1.mixer.q_proj.weight").component == "attention"
    assert classify("backbone.layers.1.mixer.up_proj.weight").component == "ffn"
    assert classify("backbone.layers.0.mixer.in_proj.weight").component == "ssm"
    assert classify("model.layers.3.mlp.shared_experts.gate_proj.weight").component == "ffn"
    assert classify("model.layers.3.mlp.experts.7.gate_proj.weight").component == "experts"
    assert classify("model.layers.0.mamba.A_log").component == "ssm"
    assert classify("model.layers.0.mamba.in_proj.weight").component == "ssm"
    assert classify("blk.0.ssm_conv1d.weight").component == "ssm"


def test_b8_layer_count_comes_from_language_stack(tmp_path):
    tensors = {f"language_model.model.layers.{i}.self_attn.q_proj.weight": ("F32", [64, 64]) for i in range(4)}
    for i in range(3):
        for j in range(6):
            tensors[f"vision_tower.encoder.layers.{i}.t{j}.weight"] = ("F32", [4])
    p = tmp_path / "mm.safetensors"
    p.write_bytes(safetensors_bytes(tensors))
    scan = scan_path(str(p), _settings(tmp_path))
    assert scan.fact_value("n_layers") == 4


def test_b9_deepseek_mla_head_dims(tmp_path):
    d = hf_llama_dir(tmp_path / "ds", config_extra={"qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128, "model_type": "deepseek_v3"})
    scan = scan_path(str(d), _settings(tmp_path))
    assert scan.fact_value("head_dim") == 192 and scan.fact_value("head_dim_v") == 128


def test_b12_outputs_dir_is_always_allowed(tmp_path):
    s = Settings(data_dir=tmp_path / "data", allowed_roots=[tmp_path / "models"])
    s.ensure_dirs()
    (s.outputs_dir / "job1").mkdir()
    assert s.is_allowed(s.outputs_dir / "job1")
    assert not s.is_allowed(tmp_path / "elsewhere")


def test_b13_all_layers_on_gpu_is_not_partial_offload(tmp_path):
    scan = scan_path(str(llama_like_gguf(tmp_path / "m.gguf")), _settings(tmp_path))
    total = estimate_memory(scan, MemoryRequest(context_length=512, device_id=None, custom_device=None)).total_bytes
    gpu = DeviceProfile(id="g", name="g", memory_kind="discrete_gpu", capacity_bytes=total - 1, reserved_bytes=0, runtime_overhead_bytes=200 * 1024**2, host_ram_bytes=1 << 34)
    fit = estimate_memory(scan, MemoryRequest(context_length=512, device_id=None, custom_device=gpu)).fit
    assert fit.gpu_layers == fit.total_layers
    assert fit.verdict == "tight" and not any("the rest run from system RAM" in n for n in fit.notes)


def test_b14_tool_refresh_requires_post_with_header(tmp_path):
    c = TestClient(create_app(_settings(tmp_path)), base_url="http://127.0.0.1")
    assert c.post("/api/tools/refresh").status_code == 403
    assert c.post("/api/tools/refresh", headers=H).status_code == 200

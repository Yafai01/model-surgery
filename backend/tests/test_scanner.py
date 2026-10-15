import pytest
from helpers import T, gguf_bytes, hf_llama_dir, llama_like_gguf, safetensors_bytes

from model_surgery.config import Settings
from model_surgery.scanner import ScanError, scan_path


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


def facts(r):
    return {f.key: (f.value, f.provenance) for f in r.hyperparameters}


def test_gguf_scan_verified_facts(tmp_path, settings):
    r = scan_path(str(llama_like_gguf(tmp_path / "m.gguf")), settings)
    f = facts(r)
    assert r.kind == "gguf" and r.integrity.valid
    assert f["n_layers"] == (2, "verified")
    assert f["n_kv_heads"] == (2, "verified")
    assert f["head_dim"] == (16, "derived")
    assert f["attention_type"][0].startswith("GQA")
    assert f["vocab_size"] == (256, "verified")
    assert f["n_params"][1] == "derived"
    assert r.quantization.label == "F16" and r.quantization.label_provenance == "verified"
    stack = r.architecture.stacks[0]
    assert stack.primary and stack.n_layers == 2 and r.architecture.uniform_layers
    comps = {c.component for c in stack.layers[0].components}
    assert comps == {"attention", "ffn", "norm"}


def test_gguf_without_metadata_infers_from_shapes(tmp_path, settings):
    p = llama_like_gguf(tmp_path / "m.gguf", with_metadata=False, n_layers=3, hidden=64, n_heads=4, n_kv=1)
    r = scan_path(str(p), settings)
    f = facts(r)
    assert f["n_layers"] == (3, "inferred")
    assert f["hidden_size"] == (64, "inferred")
    assert f["vocab_size"] == (256, "inferred")
    assert "n_heads" not in f  # cannot be derived from shapes honestly
    assert r.quantization.label_provenance == "inferred"


def test_metadata_tensor_mismatch_is_reported(tmp_path, settings):
    p = llama_like_gguf(tmp_path / "m.gguf", n_layers=2)
    raw = p.read_bytes().replace(b"llama.block_count", b"llama.block_count", 1)
    # rewrite block_count value 2 -> 5
    i = raw.index(b"llama.block_count") + len(b"llama.block_count") + 4
    raw = raw[:i] + (5).to_bytes(4, "little") + raw[i + 4 :]
    p.write_bytes(raw)
    r = scan_path(str(p), settings)
    assert any("declares 5 layers" in n for n in r.architecture.notes)


def test_tied_embeddings_inferred(tmp_path, settings):
    r = scan_path(str(llama_like_gguf(tmp_path / "m.gguf", tied=True)), settings)
    assert facts(r)["tied_embeddings"] == (True, "inferred")


def test_hf_scan(tmp_path, settings):
    d = hf_llama_dir(tmp_path / "hf", shards=2)
    r = scan_path(str(d), settings)
    f = facts(r)
    assert r.kind == "hf_safetensors" and r.integrity.valid, r.integrity.errors
    assert f["architecture"] == ("llama", "verified")
    assert f["model_class"] == ("LlamaForCausalLM", "verified")
    assert f["n_kv_heads"] == (2, "verified")
    assert f["tied_embeddings"] == (False, "verified")
    assert r.quantization.label == "BF16"
    roles = {f.role for f in r.files}
    assert {"config", "safetensors index", "safetensors shard"} <= roles
    # every tensor's file_index must point at a shard
    for t in r.tensors:
        assert r.files[t.file_index].role == "safetensors shard"


def test_selecting_shard_scans_directory(tmp_path, settings):
    d = hf_llama_dir(tmp_path / "hf", shards=1)
    r = scan_path(str(d / "model.safetensors"), settings)
    assert r.kind == "hf_safetensors"
    assert "scanned the whole directory" in r.architecture.notes[0]


def test_hf_packed_quant_checkpoint(tmp_path, settings):
    d = hf_llama_dir(tmp_path / "hf", shards=1, config_extra={"quantization_config": {"quant_method": "gptq", "bits": 4}})
    r = scan_path(str(d), settings)
    assert r.quantization.packed_checkpoint and r.quantization.label == "gptq 4-bit"
    assert facts(r)["n_params_logical"][1] == "inferred"


def test_hf_multimodal_text_config(tmp_path, settings):
    d = hf_llama_dir(tmp_path / "hf", shards=1)
    import json

    cfg = json.loads((d / "config.json").read_text())
    cfg = {"model_type": "llava", "architectures": ["LlavaForConditionalGeneration"], "text_config": cfg}
    (d / "config.json").write_text(json.dumps(cfg))
    r = scan_path(str(d), settings)
    f = facts(r)
    assert f["n_layers"][0] == 2 and "text_config" in r.fact("n_layers").source


def test_single_safetensors(tmp_path, settings):
    p = tmp_path / "x.safetensors"
    p.write_bytes(safetensors_bytes({"model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]), "model.embed_tokens.weight": ("F16", [32, 8])}))
    r = scan_path(str(p), settings)
    assert r.kind == "safetensors"
    assert facts(r)["hidden_size"] == (8, "inferred")


@pytest.mark.parametrize("name", ["pytorch_model.bin", "model.pt", "model.ckpt", "weights.pth"])
def test_pickle_files_refused(tmp_path, settings, name):
    p = tmp_path / name
    p.write_bytes(b"\x80\x04\x95 pretend pickle")
    with pytest.raises(ScanError, match="never opens pickle"):
        scan_path(str(p), settings)


def test_gguf_extension_without_magic(tmp_path, settings):
    p = tmp_path / "fake.gguf"
    p.write_bytes(b"MZ" + b"\0" * 64)
    with pytest.raises(ScanError, match="magic"):
        scan_path(str(p), settings)


def test_onnx_unsupported(tmp_path, settings):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"\0" * 16)
    with pytest.raises(ScanError, match="ONNX"):
        scan_path(str(p), settings)


def test_relative_and_missing_paths(tmp_path, settings):
    with pytest.raises(ScanError, match="absolute"):
        scan_path("relative/model.gguf", settings)
    with pytest.raises(ScanError, match="does not exist"):
        scan_path(str(tmp_path / "nope.gguf"), settings)
    with pytest.raises(ScanError):
        scan_path("C:\\x\x00y", settings)


def test_allowed_roots(tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    s = Settings(data_dir=tmp_path / "data", allowed_roots=[allowed])
    p = llama_like_gguf(tmp_path / "m.gguf")
    with pytest.raises(ScanError, match="allowed roots"):
        scan_path(str(p), s)
    p2 = llama_like_gguf(allowed / "m.gguf")
    assert scan_path(str(p2), s).integrity.valid


def test_split_gguf(tmp_path, settings):
    kv1 = [("general.architecture", T["str"], "llama"), ("split.count", T["u16"], 2), ("split.no", T["u16"], 0), ("split.tensors.count", T["i32"], 2)]
    kv2 = [("split.count", T["u16"], 2), ("split.no", T["u16"], 1), ("split.tensors.count", T["i32"], 2)]
    (tmp_path / "m-00001-of-00002.gguf").write_bytes(gguf_bytes(kv1, [("token_embd.weight", [32, 4], "F16")]))
    (tmp_path / "m-00002-of-00002.gguf").write_bytes(gguf_bytes(kv2, [("output.weight", [32, 4], "F16")]))
    r = scan_path(str(tmp_path / "m-00001-of-00002.gguf"), settings)
    assert r.integrity.valid, r.integrity.errors
    assert len(r.files) == 2 and len(r.tensors) == 2
    (tmp_path / "m-00002-of-00002.gguf").unlink()
    r = scan_path(str(tmp_path / "m-00001-of-00002.gguf"), settings)
    assert any("missing split shard" in e for e in r.integrity.errors)


def test_corrupt_file_reports_integrity(tmp_path, settings):
    p = llama_like_gguf(tmp_path / "m.gguf", truncate_data=100)
    r = scan_path(str(p), settings)
    assert not r.integrity.valid

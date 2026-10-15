import pytest
from helpers import llama_like_gguf

from model_surgery.config import Settings
from model_surgery.devices import DeviceProfile, GiB
from model_surgery.memory import MemoryRequest, _use_more_bits, estimate_memory, estimate_tensor_bytes, weight_estimates
from model_surgery.scanner import scan_path


@pytest.fixture
def scan(tmp_path):
    return scan_path(str(llama_like_gguf(tmp_path / "m.gguf", n_layers=4, hidden=256, n_heads=8, n_kv=2, ffn=512, vocab=512)), Settings(data_dir=tmp_path))


def comp(est, key):
    return next(c for c in est.inference if c.key == key)


def test_estimates_are_never_measurements(scan):
    est = estimate_memory(scan, MemoryRequest())
    assert est.is_measurement is False and "not measurements" in est.disclaimer
    assert all(w.provenance in ("derived", "inferred") for w in est.weights)


def test_weight_table_ordering_and_overheads(scan):
    w = {x.id: x for x in weight_estimates(scan)}
    assert w["current"].bytes == sum(t.n_bytes for t in scan.tensors)
    assert w["fp32"].bytes > w["fp16"].bytes > w["q8_0"].bytes > w["q4_0"].bytes
    # Q8_0 carries a 2-byte scale per 32 weights: 8.5 bits per quantized weight
    assert 8.4 < w["q8_0"].bits_per_weight < 9.0
    assert w["fp16"].type_mix["F32"] == 4 * 2 + 1  # 1-D norms stay F32


def test_kv_cache_formula(scan):
    est = estimate_memory(scan, MemoryRequest(context_length=1000, batch_size=2, kv_cache_type="f16"))
    kv = comp(est, "kv_cache")
    head_dim = 256 // 8
    assert kv.bytes == 4 * 1000 * 2 * 2 * (head_dim + head_dim) * 2
    q8 = comp(estimate_memory(scan, MemoryRequest(context_length=1000, batch_size=2, kv_cache_type="q8_0")), "kv_cache")
    assert q8.bytes == int(kv.bytes / 2 * 34 / 32)


def test_flash_attention_removes_score_matrix(scan):
    on = comp(estimate_memory(scan, MemoryRequest(context_length=8192, flash_attention=True)), "compute").bytes
    off = comp(estimate_memory(scan, MemoryRequest(context_length=8192, flash_attention=False)), "compute").bytes
    assert off - on == 8 * 512 * 8192 * 4


def test_warnings(scan):
    est = estimate_memory(scan, MemoryRequest(context_length=100_000, kv_cache_type="q4_0", flash_attention=False))
    text = " ".join(est.warnings)
    assert "exceeds the model's training context" in text and "flash attention" in text


def test_device_fit_verdicts(scan):
    big = estimate_memory(scan, MemoryRequest(device_id="jetson-orin-nano-8gb"))
    assert big.fit and big.fit.verdict == "fits" and big.fit.decode_ceiling_tokens_per_s
    tiny = DeviceProfile(id="t", name="tiny", memory_kind="unified", capacity_bytes=10_000_000, reserved_bytes=0, runtime_overhead_bytes=0)
    est = estimate_memory(scan, MemoryRequest(custom_device=tiny))
    assert est.fit.verdict == "does_not_fit" and est.fit.headroom_bytes < 0


def test_partial_offload(scan):
    w = next(x for x in weight_estimates(scan) if x.id == "current").bytes
    gpu = DeviceProfile(
        id="g", name="g", memory_kind="discrete_gpu", capacity_bytes=int(w * 0.8), reserved_bytes=0,
        runtime_overhead_bytes=0, host_ram_bytes=16 * GiB,
    )
    est = estimate_memory(scan, MemoryRequest(custom_device=gpu, context_length=256))
    assert est.fit.verdict == "partial_offload"
    assert 0 < est.fit.gpu_layers < est.fit.total_layers == 4


def test_use_more_bits_matches_llama_cpp():
    assert [i for i in range(24) if _use_more_bits(i, 24)] == [0, 1, 2, 5, 8, 11, 14, 17, 20, 21, 22, 23]


def test_q4_k_m_rules_and_fallbacks(tmp_path):
    # hidden 896 is not a multiple of 256 -> Q4_K falls back to Q5_0 and Q6_K to Q8_0, as in llama.cpp
    s = scan_path(str(llama_like_gguf(tmp_path / "q.gguf", n_layers=8, hidden=896, n_heads=14, n_kv=2, ffn=1024, vocab=512)), Settings(data_dir=tmp_path))
    types = {k: v[0] for k, v in estimate_tensor_bytes(s, "Q4_K_M").items()}
    assert types["output.weight"] == "Q8_0"
    assert types["blk.0.attn_q.weight"] == "Q5_0"
    assert types["blk.0.attn_v.weight"] == "Q8_0"  # more bits -> Q6_K -> fallback Q8_0
    assert types["blk.2.attn_v.weight"] == "Q5_0"  # use_more_bits(2, 8) is false
    assert types["blk.0.ffn_down.weight"] == "Q6_K"  # ne0=1024 divisible by 256
    assert types["blk.0.attn_norm.weight"] == "F32"


def test_packed_checkpoint_uses_logical_params(tmp_path):
    from helpers import hf_llama_dir

    d = hf_llama_dir(tmp_path / "hf", shards=1, config_extra={"quantization_config": {"quant_method": "awq", "bits": 4}})
    s = scan_path(str(d), Settings(data_dir=tmp_path))
    ws = weight_estimates(s)
    assert all(w.id != "current" for w in ws)
    assert all(w.provenance == "inferred" and "logical parameter" in w.method for w in ws)

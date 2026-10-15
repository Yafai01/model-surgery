import struct

import pytest
from helpers import T, gguf_bytes, llama_like_gguf, tensor_size

from model_surgery.formats import FormatError
from model_surgery.formats.gguf import GGUFLimits, read_gguf


def test_parses_valid_file(tmp_path):
    p = llama_like_gguf(tmp_path / "m.gguf")
    g = read_gguf(p)
    assert g.valid, g.errors
    assert g.version == 3 and g.byte_order == "little" and g.alignment == 32
    assert g.get("llama.block_count") == 2
    assert g.get_array_len("tokenizer.ggml.tokens") == 256
    assert len(g.tensors) == 1 + 2 * 9 + 2
    q = next(t for t in g.tensors if t.name == "blk.0.attn_q.weight")
    assert q.shape == [64, 64] and q.ggml_type.name == "F16" and q.n_bytes == 64 * 64 * 2


def test_quantized_tensor_sizes(tmp_path):
    p = tmp_path / "q.gguf"
    p.write_bytes(gguf_bytes([], [("a", [256, 4], "Q4_K"), ("b", [64, 2], "Q8_0"), ("c", [32], "Q4_0")]))
    g = read_gguf(p)
    assert g.valid, g.errors
    sizes = {t.name: t.n_bytes for t in g.tensors}
    assert sizes == {"a": 4 * 144, "b": 2 * 2 * 34, "c": 18}


def test_big_endian(tmp_path):
    p = llama_like_gguf(tmp_path / "be.gguf", big_endian=True)
    g = read_gguf(p)
    assert g.byte_order == "big" and g.valid
    assert g.get("llama.embedding_length") == 64
    assert any("big-endian" in w for w in g.warnings)


def test_rejects_bad_magic(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"GGML" + b"\0" * 100)
    with pytest.raises(FormatError, match="magic"):
        read_gguf(p)


def test_rejects_v1(tmp_path):
    p = tmp_path / "v1.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<I", 1) + b"\0" * 32)
    with pytest.raises(FormatError, match="v1"):
        read_gguf(p)


@pytest.mark.parametrize("cut", [45, 60, 200])
def test_truncated_header_raises(tmp_path, cut):
    full = gguf_bytes([("general.architecture", T["str"], "llama")], [("a", [32, 2], "F32")])
    header_len = len(full) - 32 * 2 * 4  # data section
    p = tmp_path / "t.gguf"
    p.write_bytes(full[: max(8, header_len - cut)])
    with pytest.raises(FormatError):
        read_gguf(p)


def test_missing_data_section_is_integrity_error(tmp_path):
    full = gguf_bytes([("general.architecture", T["str"], "llama")], [("a", [32, 2], "F32")])
    p = tmp_path / "t.gguf"
    p.write_bytes(full[: len(full) - 32 * 2 * 4 - 1])  # header intact, padding + data missing
    g = read_gguf(p)
    assert not g.valid


def test_truncated_data_is_integrity_error(tmp_path):
    p = tmp_path / "t.gguf"
    p.write_bytes(gguf_bytes([], [("a", [32, 4], "F32")], truncate_data=16))
    g = read_gguf(p)
    assert not g.valid
    assert any("exceeds data section" in e for e in g.errors)


def test_huge_counts_rejected(tmp_path):
    p = tmp_path / "h.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 10**12, 0))
    with pytest.raises(FormatError, match="tensor count"):
        read_gguf(p)
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 10**9))
    with pytest.raises(FormatError, match="metadata count"):
        read_gguf(p)


def test_huge_string_length_rejected(tmp_path):
    p = tmp_path / "s.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**60) + b"x" * 16)
    with pytest.raises(FormatError):
        read_gguf(p)


def test_array_length_bounded_by_file(tmp_path):
    body = struct.pack("<Q", 3) + b"key" + struct.pack("<IIQ", T["arr"], T["u32"], 10**7)
    p = tmp_path / "a.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + body + b"\0" * 64)
    with pytest.raises(FormatError, match="truncated"):
        read_gguf(p)


def test_array_limit(tmp_path):
    body = struct.pack("<Q", 3) + b"key" + struct.pack("<IIQ", T["arr"], T["u8"], 100)
    p = tmp_path / "a.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + body + b"\0" * 100)
    with pytest.raises(FormatError, match="exceeds limit"):
        read_gguf(p, GGUFLimits(max_array_len=10))


def test_duplicate_key_rejected(tmp_path):
    p = tmp_path / "d.gguf"
    p.write_bytes(gguf_bytes([("a", T["u32"], 1), ("a", T["u32"], 2)], []))
    with pytest.raises(FormatError, match="duplicate metadata key"):
        read_gguf(p)


def test_duplicate_tensor_rejected(tmp_path):
    p = tmp_path / "d.gguf"
    p.write_bytes(gguf_bytes([], [("w", [32], "F32"), ("w", [32], "F32")]))
    with pytest.raises(FormatError, match="duplicate tensor"):
        read_gguf(p)


def test_bad_n_dims(tmp_path):
    bo = "<"
    info = struct.pack(bo + "Q", 1) + b"w" + struct.pack(bo + "I", 9)
    p = tmp_path / "n.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + info + b"\0" * 200)
    with pytest.raises(FormatError, match="dimensions"):
        read_gguf(p)


def test_unknown_type_is_integrity_error(tmp_path):
    p = tmp_path / "u.gguf"
    p.write_bytes(gguf_bytes([], [("w", [32], "F32")], type_ids={"w": 4}))  # 4 = removed Q4_2
    g = read_gguf(p)
    assert not g.valid and any("unknown or removed" in e for e in g.errors)


def test_block_alignment_of_rows(tmp_path):
    p = tmp_path / "r.gguf"
    # declare a Q4_0 tensor whose row (ne0=48) is not a multiple of 32
    raw = gguf_bytes([], [("w", [64], "Q4_0")])
    raw = raw.replace(struct.pack("<Q", 64), struct.pack("<Q", 48), 1)
    p.write_bytes(raw)
    g = read_gguf(p)
    assert any("not divisible" in e for e in g.errors)


def test_misaligned_and_overlapping_offsets(tmp_path):
    p = tmp_path / "o.gguf"
    p.write_bytes(gguf_bytes([], [("a", [32, 2], "F32"), ("b", [32, 2], "F32")], offsets={"b": 8}))
    g = read_gguf(p)
    assert any("not aligned" in e for e in g.errors)
    p.write_bytes(gguf_bytes([], [("a", [32, 2], "F32"), ("b", [32, 2], "F32")], offsets={"b": 32}))
    g = read_gguf(p)
    assert any("overlap" in e for e in g.errors)


def test_offset_beyond_file(tmp_path):
    p = tmp_path / "f.gguf"
    p.write_bytes(gguf_bytes([], [("a", [32], "F32")], offsets={"a": 2**40}))
    g = read_gguf(p)
    assert not g.valid


def test_invalid_alignment(tmp_path):
    p = tmp_path / "al.gguf"
    p.write_bytes(gguf_bytes([("general.alignment", T["u32"], 48)], []))
    with pytest.raises(FormatError, match="alignment"):
        read_gguf(p)


def test_custom_alignment(tmp_path):
    p = tmp_path / "al.gguf"
    p.write_bytes(gguf_bytes([("general.alignment", T["u32"], 64)], [("a", [32], "F32"), ("b", [32], "F32")], alignment=64))
    g = read_gguf(p)
    assert g.valid, g.errors and g.alignment == 64
    assert g.data_offset % 64 == 0


def test_long_string_truncated_for_display(tmp_path):
    p = tmp_path / "l.gguf"
    p.write_bytes(gguf_bytes([("tokenizer.chat_template", T["str"], "x" * 20000)], []))
    v = read_gguf(p).metadata["tokenizer.chat_template"]
    assert v.truncated and len(v.value) == 16384 and v.length == 20000


def test_invalid_utf8_replaced(tmp_path):
    raw = gguf_bytes([("k", T["str"], "ab")], [])
    raw = raw.replace(b"ab", b"\xff\xfe", 1)
    p = tmp_path / "u8.gguf"
    p.write_bytes(raw)
    assert "�" in read_gguf(p).metadata["k"].value


def test_large_numeric_array_skipped_not_loaded(tmp_path):
    p = tmp_path / "n.gguf"
    p.write_bytes(gguf_bytes([("tokenizer.ggml.scores", T["arr"], (T["f32"], [0.5] * 5000))], []))
    v = read_gguf(p).metadata["tokenizer.ggml.scores"]
    assert v.length == 5000 and len(v.value) == 32 and v.truncated


def test_tensor_size_helper_matches_reader():
    assert tensor_size("Q6_K", [256, 2]) == 2 * 210

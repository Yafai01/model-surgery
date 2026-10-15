import json
import struct

import pytest
from helpers import hf_llama_dir, safetensors_bytes

from model_surgery.formats import FormatError, UnsupportedFormatError
from model_surgery.formats.hf import read_hf_dir
from model_surgery.formats.safetensors_reader import read_safetensors


def test_valid_safetensors(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(safetensors_bytes({"w": ("F16", [4, 8]), "b": ("F32", [8])}, {"format": "pt"}))
    sf = read_safetensors(p)
    assert sf.valid, sf.errors
    assert sf.metadata == {"format": "pt"}
    w = next(t for t in sf.tensors if t.name == "w")
    assert w.n_bytes == 64 and w.n_elements == 32


def test_header_length_beyond_file(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(struct.pack("<Q", 10_000) + b"{}")
    with pytest.raises(FormatError, match="exceeds file size"):
        read_safetensors(p)


def test_header_length_limit(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(struct.pack("<Q", 2**40) + b"{}")
    with pytest.raises(FormatError, match="limit"):
        read_safetensors(p)


def test_duplicate_json_keys(tmp_path):
    raw = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    p = tmp_path / "a.safetensors"
    p.write_bytes(safetensors_bytes({}, raw_header=raw) + b"\0" * 4)
    with pytest.raises(FormatError, match="duplicate"):
        read_safetensors(p)


def test_not_json(tmp_path):
    p = tmp_path / "a.safetensors"
    p.write_bytes(safetensors_bytes({}, raw_header=b"{not json"))
    with pytest.raises(FormatError, match="JSON"):
        read_safetensors(p)


@pytest.mark.parametrize(
    "desc,match",
    [
        ({"dtype": "F99", "shape": [1], "data_offsets": [0, 4]}, "unknown dtype"),
        ({"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}, "invalid shape"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}, "invalid data_offsets"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}, "byte range"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [0, 4000]}, "byte range"),
        ({"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}, "invalid shape"),
    ],
)
def test_bad_descriptors_are_errors(tmp_path, desc, match):
    p = tmp_path / "a.safetensors"
    p.write_bytes(safetensors_bytes({}, header_override={"w": desc}) + b"\0" * 8)
    sf = read_safetensors(p)
    assert not sf.valid and any(match in e for e in sf.errors)


def test_out_of_bounds_and_overlap(tmp_path):
    hdr = {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}
    p = tmp_path / "a.safetensors"
    p.write_bytes(safetensors_bytes({}, header_override=hdr) + b"\0" * 8)
    sf = read_safetensors(p)
    errs = " ".join(sf.errors)
    assert "overlaps" in errs and "beyond data section" in errs


def test_hf_dir_sharded(tmp_path):
    d = hf_llama_dir(tmp_path / "m", shards=3)
    hf = read_hf_dir(d)
    assert not hf.errors, hf.errors
    assert len(hf.shards) == 3 and hf.index_path is not None
    assert hf.config["model_type"] == "llama"


def test_hf_index_path_traversal_rejected(tmp_path):
    d = hf_llama_dir(tmp_path / "m", shards=2)
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    first = next(iter(idx["weight_map"]))
    idx["weight_map"][first] = "../../evil.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(FormatError, match="unsafe shard filename"):
        read_hf_dir(d)


def test_hf_index_missing_shard_and_mismatch(tmp_path):
    d = hf_llama_dir(tmp_path / "m", shards=2)
    (d / "model-00002-of-00002.safetensors").unlink()
    hf = read_hf_dir(d)
    assert any("missing shard" in e for e in hf.errors)
    assert any("not in any shard" in e for e in hf.errors)


def test_hf_pickle_only_is_unsupported(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "pytorch_model.bin").write_bytes(b"\x80\x04 not really a pickle")
    with pytest.raises(UnsupportedFormatError, match="will not open"):
        read_hf_dir(d)


def test_hf_pickle_listed_never_opened(tmp_path):
    d = hf_llama_dir(tmp_path / "m", shards=1)
    (d / "pytorch_model.bin").write_bytes(b"\x80\x04 untrusted")
    hf = read_hf_dir(d)
    assert [u.path.name for u in hf.unparsed] == ["pytorch_model.bin"]
    assert "never opened" in hf.unparsed[0].reason


def test_hf_config_duplicate_keys(tmp_path):
    d = hf_llama_dir(tmp_path / "m", shards=1)
    (d / "config.json").write_text('{"a": 1, "a": 2}')
    with pytest.raises(FormatError, match="duplicate"):
        read_hf_dir(d)


def test_hf_config_too_large(tmp_path, monkeypatch):
    import model_surgery.formats.hf as hfmod

    d = hf_llama_dir(tmp_path / "m", shards=1)
    monkeypatch.setattr(hfmod, "MAX_JSON_BYTES", 10)
    with pytest.raises(FormatError, match="refusing"):
        read_hf_dir(d)
